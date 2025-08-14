from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from .auth import current_user, get_db
from .models import User

async def auth_user(user: User = Depends(current_user)) -> User:
    return user

async def db_sess(db: AsyncSession = Depends(get_db)) -> AsyncSession:
    return db