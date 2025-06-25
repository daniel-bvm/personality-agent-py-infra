from app.oai_models import (
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    ChatCompletionStreamResponse, 
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
from app.configs import settings, BASE_SYSTEM_PROMPT, Dependency
import json
import logging
import re
import os
from pydantic import BaseModel

class LabRequest(BaseModel):
    personality: str = ""
    dependencies: list[Dependency] = []
    thinking: bool = True

logger = logging.getLogger(__name__)

async def get_system_prompt(newest_message: Optional[str], personality: str = "", enable_memory: bool = True) -> str:
    memory = await get_bio(newest_message)
    memory_str = "\n".join([f"- {m}" for m in memory])
    return BASE_SYSTEM_PROMPT.format(
        personality=personality or settings.agent_personality, 
        bio=memory_str if enable_memory else ""
    )

async def handle_request(request: ChatCompletionRequest, lab: Optional[LabRequest] = None) -> AsyncGenerator[ChatCompletionStreamResponse | ChatCompletionResponse, None]:
    messages = request.messages
    assert len(messages) > 0, "No messages in the request"
 
    logger.info(f"Lab: {lab}")
    system_prompt = await get_system_prompt(
        get_newest_message(messages), 
        personality=lab.personality if lab else settings.agent_personality,
        enable_memory=lab is None
    )

    messages: list[dict[str, Any]] = refine_chat_history(messages, system_prompt)

    tools = await compose_mcp._mcp_list_tools()
    oai_tools = convert_mcp_tools_to_openai_format(tools)

    finished = False
    n_calls, max_calls = 0, 25
    a2a_call_pattern = re.compile(r"^call_(\d+)$", re.IGNORECASE)
    dependencies = lab.dependencies if lab is not None else settings.agent_dependencies

    a2a_chat_histories: dict[str, list[dict[str, str]]] = {
        str(dependency.id): []
        for dependency in dependencies
        if dependency.id
    }

    while not finished:
        completion_builder = ChatCompletionResponseBuilder()
        requires_toolcall = n_calls < max_calls
        a2a_toolcalls = await get_a2a_toolcalls(dependencies)
        toolcalls = oai_tools + a2a_toolcalls

        payload = dict(
            messages=messages,
            tools=toolcalls,
            tool_choice="auto",
            model=settings.llm_model_id
        )

        if not requires_toolcall:
            payload.pop("tools")
            payload.pop("tool_choice")

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

                logger.info(f"A2A call: {_name}; Message: {message_arg}")
                async for chunk in handle_a2a_call(agent_id, message_arg, a2a_chat_histories[agent_id]):
                    yield chunk
                    _result += chunk.choices[0].delta.content

                _result = refine_mcp_response(_result)
                a2a_chat_histories[agent_id].extend([
                    {"role": "user", "content": message_arg},
                    {"role": "assistant", "content": _result}
                ])

            else:
                if _name != "bio_action":
                    yield wrap_chunk(random_uuid(), f"<action>Executing <b>{_name}</b></action>", "assistant")
                    yield wrap_chunk(random_uuid(), f"<details>\n<summary>Arguments:</summary>\n```json\n{json.dumps(_args, indent=2)}\n```\n</details>", "assistant")

                else:
                    yield wrap_chunk(random_uuid(), f"<action>Memory updated!</action>", "assistant")


                if lab is not None and _name == "bio_action":
                    # skip memory update for lab req
                    _result = "Memory updated!"

                else:
                    _result = refine_mcp_response(await execute_openai_compatible_toolcall(_name, _args, compose_mcp))

                if _name != "bio_action":
                    yield wrap_chunk(random_uuid(), f"<details>\n<summary>Result:</summary>\n```json\n{json.dumps(_result, indent=2)}\n```\n</details>", "assistant")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": _id,
                    "content": _result
                }
            )

        finished = len((completion.choices[0].message.tool_calls or [])) == 0

    os.makedirs("logs", exist_ok=True)
    with open(f"logs/messages-{request.request_id}.json", "w") as f:
        json.dump(messages, f, indent=2)

    yield completion