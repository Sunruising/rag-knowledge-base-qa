"""
FastAPI 后端：把 RAG 能力包成网页可调用的接口，并托管前端页面。

启动后访问 http://127.0.0.1:8000 即可使用。
"""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()  # 读取 .env 里的 DEEPSEEK_API_KEY

from app.rag import RagEngine, DATA_DIR  # noqa: E402  (load_dotenv 必须先执行)

app = FastAPI(title="RAG 知识库问答 Demo")

STATIC_DIR = Path(__file__).resolve().parent / "static"
DOCS_DIR = DATA_DIR / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# RAG 引擎只初始化一次（加载 embedding 模型较慢，常驻）
engine: RagEngine | None = None


@app.on_event("startup")
def _startup():
    global engine
    engine = RagEngine()


class AskBody(BaseModel):
    question: str


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status():
    return engine.status()


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "没有文件名")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".txt", ".md", ".pdf"):
        raise HTTPException(400, "只支持 .txt / .md / .pdf")
    save_path = DOCS_DIR / file.filename
    save_path.write_bytes(await file.read())
    chunks = engine.ingest_file(save_path)
    return {"filename": file.filename, "chunks": chunks}


@app.post("/api/ask")
def ask(body: AskBody):
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "问题不能为空")
    return engine.answer(q)


@app.post("/api/reset")
def reset():
    engine.reset()
    return {"ok": True}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
