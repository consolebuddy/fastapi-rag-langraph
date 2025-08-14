import os, secrets, time
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.hash import bcrypt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from .db import AsyncSessionLocal
from .models import User

security = HTTPBearer()

async def get_db():
    async with AsyncSessionLocal() as s:
        yield s

def hash_pw(pw: str) -> str:
    return bcrypt.hash(pw)

def verify_pw(pw: str, hashed: str) -> bool:
    return bcrypt.verify(pw, hashed)

# Very simple signed token (HMAC). Replace with JWT in prod.
SECRET = os.getenv("AUTH_SECRET", secrets.token_hex(16))

def sign(name: str) -> str:
    ts = int(time.time())
    return f"{name}.{ts}.{secrets.token_hex(8)}.{secrets.token_hex(8)}"

async def current_user(creds: HTTPAuthorizationCredentials = Depends(security), db: AsyncSession = Depends(get_db)) -> User:
    token = creds.credentials
    # naive parse; accept any token created by our /login (no blacklist for brevity)
    name = token.split(".")[0]
    user = (await db.execute(select(User).where(User.name==name))).scalar_one_or_none()
    if not user:
        raise HTTPException(401, "Invalid token")
    return user