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
from app.configs import settings, BASE_SYSTEM_PROMPT
import json
import logging
import re
import os

logger = logging.getLogger(__name__)

async def get_system_prompt(newest_message: Optional[str]) -> str:
    memory = await get_bio(newest_message)
    memory_str = "\n".join([f"- {m}" for m in memory])
    return BASE_SYSTEM_PROMPT.format(personality=settings.agent_personality, bio=memory_str)
    
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
        str(colab["id"]): []
        for colab in colaborators
        if colab.get("id")
    }

    while not finished:
        completion_builder = ChatCompletionResponseBuilder()
        requires_toolcall = n_calls < max_calls
        a2a_toolcalls = await get_a2a_toolcalls()
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

                async for chunk in handle_a2a_call(agent_id, message_arg, colaborators_chat_histories[agent_id]):
                    yield chunk
                    _result += chunk.choices[0].delta.content

                _result = refine_mcp_response(_result)
                colaborators_chat_histories[agent_id].extend([
                    {"role": "user", "content": message_arg},
                    {"role": "assistant", "content": _result}
                ])

            else:
                if _name != "bio_action":
                    yield wrap_chunk(random_uuid(), f"<action>Executing <b>{_name}</b></action>", "assistant")
                    yield wrap_chunk(random_uuid(), f"<details>\n<summary>Arguments:</summary>\n```json\n{json.dumps(_args, indent=2)}\n```\n</details>", "assistant")

                else:
                    yield wrap_chunk(random_uuid(), f"<action>Memory updated!</action>", "assistant")

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