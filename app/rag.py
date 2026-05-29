"""RAG 知识库核心：文档切分、本地向量化、Chroma 检索，以及基于检索结果的生成。"""

import os

# 国内访问 huggingface.co 常不稳定，默认改走镜像站下载 embedding 模型。
# 必须在 import transformers 之前设置才生效。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import uuid
from pathlib import Path

import chromadb
import torch
from openai import OpenAI
from transformers import AutoModel, AutoTokenizer

# ---- 配置（可按需改）----
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_DIR = DATA_DIR / "chroma"          # 向量库存盘位置
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"    # 本地中文 embedding 模型（首次会自动下载约 100MB）
DEEPSEEK_MODEL = "deepseek-chat"
CHUNK_SIZE = 400                          # 每段大约多少字
CHUNK_OVERLAP = 80                        # 相邻段重叠多少字（避免把一句话从中间切断丢上下文）
TOP_K = 4                                 # 每次问答检索几段

# bge 系列模型的官方建议：查询(query)前要加这句指令，文档段落(passage)则不用。
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class LocalEmbedder:
    """直接用 transformers 加载 bge 模型做向量化（绕开 sentence-transformers 的版本坑）。
    bge 系列用 [CLS] 向量 + L2 归一化，再用余弦相似度比较。"""

    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: list[str]) -> list[list[float]]:
        batch = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        out = self.model(**batch)
        emb = out.last_hidden_state[:, 0]  # 取 [CLS] 位置的向量（bge 推荐做法）
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)  # L2 归一化
        return emb.tolist()


class RagEngine:
    def __init__(self):
        # 加载本地 embedding 模型（启动时加载一次，常驻内存）
        self.embedder = LocalEmbedder(EMBED_MODEL)

        # 连接本地向量库（存成文件，不需要单独装数据库服务）
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            name="docs",
            metadata={"hnsw:space": "cosine"},  # 用余弦相似度衡量"语义接近程度"
        )

        # DeepSeek 用 OpenAI 兼容接口，延迟创建：没 key 时也能用上传/检索，调用时才报错
        self._llm = None

    def _get_llm(self) -> OpenAI:
        if self._llm is None:
            key = os.environ.get("DEEPSEEK_API_KEY")
            if not key:
                raise RuntimeError("未配置 DEEPSEEK_API_KEY，请在项目根目录 .env 填入后重启服务")
            self._llm = OpenAI(api_key=key, base_url="https://api.deepseek.com")
        return self._llm

    # ---------- 1. 切分 ----------
    def _chunk(self, text: str) -> list[str]:
        text = text.replace("\r\n", "\n").strip()
        if not text:
            return []
        chunks, start = [], 0
        while start < len(text):
            end = start + CHUNK_SIZE
            piece = text[start:end]
            # 尽量在换行处断开，读起来更完整
            if end < len(text):
                nl = piece.rfind("\n")
                if nl > CHUNK_SIZE * 0.5:
                    piece = piece[: nl + 1]
                    end = start + len(piece)
            chunks.append(piece.strip())
            start = end - CHUNK_OVERLAP  # 回退一点，制造重叠
        return [c for c in chunks if c]

    # ---------- 2+3. 向量化并入库 ----------
    def ingest_text(self, text: str, source: str) -> int:
        chunks = self._chunk(text)
        if not chunks:
            return 0
        embeddings = self.embedder.encode(chunks)
        self.collection.add(
            ids=[str(uuid.uuid4()) for _ in chunks],
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{"source": source, "chunk": i} for i in range(len(chunks))],
        )
        return len(chunks)

    def ingest_file(self, path: Path) -> int:
        """支持 .txt / .md / .pdf"""
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        else:  # txt / md / 其它纯文本
            text = path.read_text(encoding="utf-8", errors="ignore")
        return self.ingest_text(text, source=path.name)

    # ---------- 4. 检索 ----------
    def retrieve(self, question: str, k: int = TOP_K) -> list[dict]:
        q_emb = self.embedder.encode([QUERY_INSTRUCTION + question])
        res = self.collection.query(query_embeddings=q_emb, n_results=k)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        out = []
        for doc, meta, dist in zip(docs, metas, dists):
            out.append(
                {
                    "text": doc,
                    "source": meta.get("source", "?"),
                    "score": round(1 - dist, 3),  # 余弦距离转相似度，越接近 1 越相关
                }
            )
        return out

    # ---------- 5. 生成 ----------
    def answer(self, question: str) -> dict:
        hits = self.retrieve(question)
        if not hits:
            return {
                "answer": "知识库还是空的，请先上传文档。",
                "sources": [],
            }

        context = "\n\n".join(
            f"【资料{i+1}·来自 {h['source']}】\n{h['text']}" for i, h in enumerate(hits)
        )

        # 关键：系统提示里强制"只能基于资料回答、没有就说不知道"——这是 RAG 不瞎编的核心
        system_prompt = (
            "你是一个严谨的知识库问答助手。你只能依据下面提供的【参考资料】回答用户问题。"
            "如果参考资料中没有足以回答问题的信息，必须如实回答："
            "「根据现有资料，我无法回答这个问题。」"
            "绝对不要编造、不要使用资料之外的知识。回答用中文，简洁准确。"
        )
        user_prompt = f"【参考资料】\n{context}\n\n【用户问题】\n{question}"

        resp = self._get_llm().chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,  # 低温度，让回答更稳定、不发散
        )
        answer_text = resp.choices[0].message.content

        return {
            "answer": answer_text,
            "sources": [
                {"source": h["source"], "score": h["score"], "snippet": h["text"][:120]}
                for h in hits
            ],
        }

    # ---------- 辅助 ----------
    def status(self) -> dict:
        count = self.collection.count()
        # 取出所有 metadata 统计有哪些来源文档
        sources = set()
        if count:
            got = self.collection.get(include=["metadatas"])
            for m in got.get("metadatas", []):
                sources.add(m.get("source", "?"))
        return {"chunks": count, "docs": sorted(sources)}

    def reset(self):
        self.client.delete_collection("docs")
        self.collection = self.client.get_or_create_collection(
            name="docs", metadata={"hnsw:space": "cosine"}
        )
