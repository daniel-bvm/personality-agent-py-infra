from app.oai_models import (
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    ChatCompletionStreamResponse, 
    ErrorResponse, 
    random_uuid
)

from app.oai_streaming import create_streaming_response, ChatCompletionResponseBuilder
from app.tools import compose as compose_mcp, get_bio, get_a2a_toolcalls, handle_a2a_call
from app.utils import (
    refine_mcp_response, 
    convert_mcp_tools_to_openai_format, 
    execute_openai_compatible_toolcall,
    refine_chat_history,
    refine_assistant_message,
    get_newest_message,
    wrap_chunk
)
from typing import Optional, Any, AsyncGenerator
from app.configs import settings, NOTIFICATION_TEMPLATES
import json
import logging
import random
import re
from cryptoagents_a2a_devkit import get_agent_detail

logger = logging.getLogger(__name__)

async def get_system_prompt(newest_message: Optional[str]) -> str:
    system_prompt = settings.agent_system_prompt

    if newest_message is None:
        return system_prompt

    memory = await get_bio(newest_message)
    memory_str = ""

    for m in memory:
        memory_str += f"- {m}\n"

    if len(memory) > 0:
        logger.info(f"Memory:\n{memory_str}")
        system_prompt += f"\n{system_prompt}\n\nBio:\n{memory_str}"
    
    return system_prompt

    
async def handle_request(request: ChatCompletionRequest) -> AsyncGenerator[ChatCompletionStreamResponse | ChatCompletionResponse, None]:
    messages = request.messages
    assert len(messages) > 0, "No messages in the request"
 
    system_prompt = await get_system_prompt(get_newest_message(messages))
    messages: list[dict[str, Any]] = refine_chat_history(messages, system_prompt)

    tools = await compose_mcp._mcp_list_tools()
    oai_tools = convert_mcp_tools_to_openai_format(tools)

    finished = False
    n_calls, max_calls = 0, 25
    a2a_call_pattern = re.compile(r"^call_(\d+)$", re.IGNORECASE)
    colaborators = settings.agent_collaborators

    colaborators_chat_histories: dict[str, list[dict[str, str]]] = {
        str(colab): []
        for colab in colaborators
    }

    while not finished:
        completion_builder = ChatCompletionResponseBuilder()
        requires_toolcall = n_calls < max_calls
        toolcalls = oai_tools + await get_a2a_toolcalls()

        payload = dict(
            messages=messages,
            tools=toolcalls,
            tool_choice="auto",
            model=settings.llm_model_id
        )

        if not requires_toolcall:
            payload.pop("tools")
            payload.pop("tool_choice")

        logger.info(f"Payload - URL: {settings.llm_base_url}, API Key: {'*' * len(settings.llm_api_key)}, Model: {settings.llm_model_id}")
        streaming_iter = create_streaming_response(
            settings.llm_base_url,
            settings.llm_api_key,
            **payload
        )

        async for chunk in streaming_iter:
            completion_builder.add_chunk(chunk)

            if chunk.choices[0].delta.content:
                yield chunk

        completion = await completion_builder.build()
        messages.append(refine_assistant_message(completion.choices[0].message))

        for call in (completion.choices[0].message.tool_calls or []):
            n_calls += 1

            _id, _name, _args = call.id, call.function.name, call.function.arguments
            _args: dict = json.loads(_args)
            match = a2a_call_pattern.match(_name)

            if match:
                _result, agent_id = "", str(match.group(1))
                message_arg = _args.get("message", "")

                async for chunk in handle_a2a_call(agent_id, message_arg, colaborators_chat_histories[agent_id]):
                    yield chunk
                    _result += chunk.choices[0].delta.content

                colaborators_chat_histories[agent_id].extend([
                    {"role": "user", "content": message_arg},
                    {"role": "assistant", "content": refine_mcp_response(_result)}
                ])

            else:
                _result = await execute_openai_compatible_toolcall(_name, _args, compose_mcp)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": _id,
                    "content": refine_mcp_response(_result)
                }
            )


        finished = len((completion.choices[0].message.tool_calls or [])) == 0

    yield completion