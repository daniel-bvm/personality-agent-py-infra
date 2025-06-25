from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse
from app.handlers import handle_request, LabRequest
from app.oai_models import ChatCompletionRequest, ChatCompletionStreamResponse, random_uuid
import time
from typing import AsyncGenerator
import logging

logger = logging.getLogger(__name__)
api_router = APIRouter()

@api_router.post("/prompt")
async def chat_completions(request: ChatCompletionRequest, original_request: Request):
    enqueued = time.time()
    ttft, tps, n_tokens = float("inf"), None, 0
    req_id = request.request_id or f"req-{random_uuid()}"

    orig_data = await original_request.json()
    lab_req_payload = (
        LabRequest.model_validate(orig_data)
        if orig_data.get("personality")
        else None
    )

    if request.stream:
        generator = handle_request(request, lab_req_payload)

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
        async for chunk in handle_request(request, lab_req_payload):
            current_time = time.time()

            n_tokens += 1
            ttft = min(ttft, current_time - enqueued)
            tps = n_tokens / (current_time - enqueued)

        logger.info(f"Request {req_id} - TTFT: {ttft:.2f}s, TPS: {tps:.2f} tokens/s")
        return JSONResponse(chunk.model_dump())

@api_router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    return await chat_completions(request)

@api_router.post("/cancel")
async def cancel():
    # TODO: implement cancel
    return JSONResponse({"status": "ok"})