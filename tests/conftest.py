import asyncio, os, pytest
from httpx import AsyncClient
from fastapi import FastAPI

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop

@pytest.fixture()
async def client():
    os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    from app.main import app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac