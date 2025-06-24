from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from app.handlers import handle_request
from app.oai_models import ChatCompletionRequest, ChatCompletionStreamResponse, random_uuid
import time
from typing import AsyncGenerator
import logging

logger = logging.getLogger(__name__)
api_router = APIRouter()

@api_router.post("/prompt")
async def chat_completions(request: ChatCompletionRequest):
    enqueued = time.time()
    ttft, tps, n_tokens = float("inf"), None, 0
    req_id = request.request_id or f"req-{random_uuid()}"

    if request.stream:
        generator = handle_request(request)

        async def to_bytes(gen: AsyncGenerator) -> AsyncGenerator[bytes, None]:
            nonlocal ttft, tps, n_tokens

            async for chunk in gen:
                current_time = time.time()

                n_tokens += 1
                ttft = min(ttft, current_time - enqueued)
                tps = n_tokens / (current_time - enqueued)

                if isinstance(chunk, ChatCompletionStreamResponse):
                    data = chunk.model_dump_json()
                    yield "data: " + data + "\n\n"

            logger.info(f"Request {req_id} - TTFT: {ttft:.2f}s, TPS: {tps:.2f} tokens/s")
            yield "data: [DONE]\n\n"

        return StreamingResponse(to_bytes(generator), media_type="text/event-stream")
    
    else:
        async for chunk in handle_request(request):
            current_time = time.time()

            n_tokens += 1
            ttft = min(ttft, current_time - enqueued)
            tps = n_tokens / (current_time - enqueued)

        logger.info(f"Request {req_id} - TTFT: {ttft:.2f}s, TPS: {tps:.2f} tokens/s")
        return JSONResponse(chunk.model_dump())
