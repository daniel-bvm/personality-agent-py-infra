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

    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)

    config = uvicorn.Config(
        server_app,
        loop=event_loop,
        host=settings.host,
        port=settings.port,
        log_level="warning",
        timeout_keep_alive=300,
        workers=32
    )

    server = uvicorn.Server(config)
    event_loop.run_until_complete(server.serve())

if __name__ == '__main__':
    main()