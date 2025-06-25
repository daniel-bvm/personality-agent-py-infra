import fastapi
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
from app.configs import settings
from app.apis import api_router
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    server_app = fastapi.FastAPI()

    server_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    server_app.include_router(api_router)

    @server_app.get("/health")
    async def healthcheck():
        return {"status": "ok", "message": "Yo, I am alive"}

    uvicorn.run(server_app, host=settings.host, port=settings.port)

if __name__ == '__main__':
    main()