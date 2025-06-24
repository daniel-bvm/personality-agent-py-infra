from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json
import asyncio
import uvicorn
import os 
import logging 
import random
from fastapi.responses import RedirectResponse
from typing import AsyncGenerator

HOSTNAME = os.getenv("HOSTNAME")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

server_app = FastAPI()

logger.info(f"Hostname: {HOSTNAME}")

async def random_chunking(s: str) -> AsyncGenerator[str, None]:
    i = 0
    
    while i < len(s):
        next_i = i + random.randint(1, 10)
        yield s[i:next_i]
        i = next_i

        await asyncio.sleep(random.uniform(0.1, 0.5))


@server_app.post("/prompt")
async def prompt() -> StreamingResponse:
    async def generator():
        async for chunk in random_chunking(
            random.choice([
                "Hi, this is Dr. Moon, how can I help you today?", 
                "I'm Dr. Moon, your AI healthcare assistant. How can I assist you today?", 
                "Hello! I'm Dr. Moon, your AI healthcare assistant. How can I help you?",
                "Sorry, I'm not sure I understand. Can you please rephrase your question?",
            ])
        ):
            yield 'data: ' + json.dumps({
                "id": "1",
                "object": "chat.completion.chunk",
                "created": 1719206400,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk},
                        "role": "assistant"
                    }
                ]
            }) + '\n\n'

    return StreamingResponse(generator(), media_type="text/event-stream")

@server_app.get("/vibe-agent/{agent_id}")
async def vibe_agent(agent_id: str):
    return {
        "container_name": HOSTNAME,
        "container_id": HOSTNAME,
        "port": int(os.getenv("PORT", 8010)),
        "status": "running",
        "name": "Dr. Moon",
        "description": "**Dr. Moon** is an AI-powered healthcare assistant designed to provide medical information, analyze X-ray images, and facilitate medical research. The goal of this agent is to assist healthcare professionals and patients with various medical inquiries and diagnostic support.",
        "meta_data": {
            "nft_token_image": "https://cdn.eternalai.org/homepage/cryptoagents/528/9550.png"
        }
    }
    
@server_app.post("/agent-router/prompt")
async def agent_router_prompt(url: str) -> StreamingResponse:
    return await prompt()

@server_app.get("/health")
async def health():
    return {"status": "ok"}

def main():
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)

    config = uvicorn.Config(
        server_app,
        loop=event_loop,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8010)),
        timeout_keep_alive=300,
        workers=1
    )

    server = uvicorn.Server(config)
    event_loop.run_until_complete(server.serve())

if __name__ == "__main__":
    main()