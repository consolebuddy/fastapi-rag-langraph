from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from .db import init_db
from .models import User, Session, Message
from .schemas import RegisterRequest, TokenResponse, ChatRequest, ChatResponse
from .auth import hash_pw, verify_pw, sign
from .deps import auth_user, db_sess
from rag.retriever import answer_with_rag
from .logging_conf import init_logging

app = FastAPI(title="UltraSafe Chatbot + Agents")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def _startup():
    init_logging()
    await init_db()

@app.post("/auth/register", response_model=TokenResponse)
async def register(body: RegisterRequest, db: AsyncSession = Depends(db_sess)):
    exists = (await db.execute(select(User).where(User.name==body.name))).scalar_one_or_none()
    if exists:
        raise HTTPException(400, "User exists")
    user = User(name=body.name, password_hash=hash_pw(body.password))
    db.add(user)
    await db.commit()
    return TokenResponse(access_token=sign(user.name))

@app.post("/auth/login", response_model=TokenResponse)
async def login(body: RegisterRequest, db: AsyncSession = Depends(db_sess)):
    user = (await db.execute(select(User).where(User.name==body.name))).scalar_one_or_none()
    if not user or not verify_pw(body.password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")
    return TokenResponse(access_token=sign(user.name))

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user: User = Depends(auth_user), db: AsyncSession = Depends(db_sess)):
    # Ensure session exists
    sess = (await db.execute(select(Session).where(Session.session_id==req.session_id).limit(1))).scalar_one_or_none()
    if not sess:
        sess = Session(session_id=req.session_id, user_id=user.id)
        db.add(sess)
        await db.commit()
        await db.refresh(sess)

    db.add(Message(role="user", content=req.message, session_pk=sess.id))
    await db.commit()

    ans, cits = await answer_with_rag(req.message)

    db.add(Message(role="assistant", content=ans, session_pk=sess.id))
    await db.commit()

    return ChatResponse(answer=ans, citations=cits)

@app.get("/history/{session_id}")
async def history(session_id: str, user: User = Depends(auth_user), db: AsyncSession = Depends(db_sess)):
    sess = (await db.execute(select(Session).where(Session.session_id==session_id))).scalar_one_or_none()
    if not sess:
        return []
    msgs = (await db.execute(select(Message).where(Message.session_pk==sess.id))).scalars().all()
    return [{"role": m.role, "content": m.content} for m in msgs]

# Test B endpoint (LangGraph orchestration)
from agents.graph import run_research

@app.post("/agents/research")
async def agents_research(payload: dict, user: User = Depends(auth_user)):
    """payload: {"question": str, "max_papers": int}
    """
    report = await run_research(payload.get("question", ""), payload.get("max_papers"))
    return report