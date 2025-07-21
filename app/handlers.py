from app.oai_models import (
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    ChatCompletionStreamResponse, 
    ErrorResponse,
    random_uuid,
    LabRequest,
    ChatCompletionAdditionalParameters
)
import asyncio

from app.oai_streaming import create_streaming_response, ChatCompletionResponseBuilder
from app.tools import compose as compose_mcp, get_bio, get_a2a_toolcalls, handle_a2a_call, extract_keywords, search_web_dummy
from app.utils import (
    refine_mcp_response, 
    convert_mcp_tools_to_openai_format, 
    execute_openai_compatible_toolcall,
    refine_chat_history,
    refine_assistant_message,
    get_newest_message,
    wrap_chunk,
    create_rich_user_message,
    AgentResourceManager,
    sync2async
)
from typing import Optional, Any, AsyncGenerator
from app.configs import settings, BASE_SYSTEM_PROMPT
import json
import logging
import re
import os

logger = logging.getLogger(__name__)

async def get_system_prompt(newest_message: Optional[str], personality: str = "", enable_memory: bool = True) -> str:
    keywords = await sync2async(extract_keywords)(newest_message)
    memory = await get_bio(newest_message)
    search_fn = sync2async(search_web_dummy)

    memory_str = "\n".join([f"- {m}" for m in memory])

    sys_prompt = BASE_SYSTEM_PROMPT.format(
        personality=personality or settings.agent_personality, 
        bio=memory_str if enable_memory else ""
    )

    results = await asyncio.gather(*[search_fn(kw) for kw in keywords], return_exceptions=True)
    refs = ''

    max_results = 10

    if len(keywords) > 0:
        per_kw = (max_results + len(keywords) - 1) // len(keywords)
        refs += f"\n## References\n"

        unique_urls = set([])
        
        found_any = False

        for kw, result in zip(keywords, results):
            if not isinstance(result, list) or len(result) == 0:
                continue

            select = min(len(result), per_kw, max_results - len(unique_urls))

            if select < 1:
                continue

            for r in result[:select]:
                if r['url'] in unique_urls:
                    continue

                unique_urls.add(r['url'])

                logger.info(f"Reference for {kw}: {r['title']} - {r['description']} - {r['url']}")

                refs += f"- {r['title']}\n"
                refs += f"  - {r['description']}\n"
                refs += f"  - {r['url']}\n\n"

                found_any = True

        if found_any:
            return sys_prompt + '\n\n' + refs + '\n\nhint: use scrape tool to explore more information'
    
    return sys_prompt
    
from typing import Callable

async def wrapstream(
    streaming_iter: AsyncGenerator[ChatCompletionStreamResponse | ErrorResponse, None], 
    callback: Callable[[ChatCompletionStreamResponse | ErrorResponse], None]
):
    async for chunk in streaming_iter:
        callback(chunk)

        if chunk.choices[0].delta.content:
            yield chunk

async def handle_request(
    request: ChatCompletionRequest, 
    event: asyncio.Event,
    lab: Optional[LabRequest] = None, 
    additional_parameters: Optional[ChatCompletionAdditionalParameters] = None,
) -> AsyncGenerator[ChatCompletionStreamResponse | ChatCompletionResponse, None]:
    messages = request.messages
    assert len(messages) > 0, "No messages in the request"

    arm = AgentResourceManager()
    logger.info(f"Lab: {lab}")

    system_prompt = await get_system_prompt(
        get_newest_message(messages), 
        personality=lab.personality if lab else settings.agent_personality,
        enable_memory=lab is None
    )

    messages: list[dict[str, Any]] = refine_chat_history(messages, system_prompt, arm)

    tools = await compose_mcp._mcp_list_tools()
    oai_tools = convert_mcp_tools_to_openai_format(tools)

    finished = False
    n_calls, max_calls = 0, 25
    a2a_call_pattern = re.compile(r"^call_(\d+)$", re.IGNORECASE)
    dependencies = lab.dependencies if lab is not None else settings.agent_dependencies

    a2a_chat_histories: dict[str, list[dict[str, Any]]] = {}

    while not finished and not event.is_set():
        completion_builder = ChatCompletionResponseBuilder()
        requires_toolcall = n_calls < max_calls
        a2a_toolcalls = await get_a2a_toolcalls(dependencies)
        toolcalls = oai_tools + a2a_toolcalls

        payload = dict(
            messages=messages,
            tools=toolcalls,
            tool_choice="auto",
            model=settings.llm_model_id,
            **(
                additional_parameters.model_dump() 
                if additional_parameters and 'api.openai.com' not in settings.llm_base_url 
                else {}
            )
        )

        if not requires_toolcall:
            payload.pop("tools")
            payload.pop("tool_choice")

        streaming_iter = create_streaming_response(
            settings.llm_base_url,
            settings.llm_api_key,
            **payload
        )

        # need to reveal resource
        async for chunk in arm.handle_streaming_response(wrapstream(streaming_iter, completion_builder.add_chunk)):
            if event.is_set():
                logger.info(f"[main] Event signal received, stopping the request")
                break

            yield chunk

        completion = await completion_builder.build()
        messages.append(refine_assistant_message(completion.choices[0].message))

        for call in (completion.choices[0].message.tool_calls or []):
            n_calls += 1

            _id, _name, _args = call.id, call.function.name, call.function.arguments
            _args: dict = json.loads(_args)
            a2a_call_match = a2a_call_pattern.match(_name)

            if a2a_call_match:
                _result, agent_id = "", str(a2a_call_match.group(1))
                message_arg = _args.get("message", "")

                if agent_id not in a2a_chat_histories:
                    a2a_chat_histories[agent_id] = []

                a2a_chat_histories[agent_id].append(create_rich_user_message(message_arg, arm))

                logger.info(f"A2A call: {_name}; Message: {message_arg}")
                async for chunk in handle_a2a_call(agent_id, event, a2a_chat_histories[agent_id], additional_parameters):
                    if event.is_set():
                        break

                    if chunk.choices[0].delta.content:
                        _result += chunk.choices[0].delta.content
                        yield chunk

                _result = refine_mcp_response(_result, arm, skip_embed_resource=True) # remain the resource in the assistant message
                a2a_chat_histories[agent_id].append({"role": "assistant", "content": _result})

                _result = refine_mcp_response(_result, arm) # remove the resource in the assistant, tool message
                logger.info(f"A2A result: {_result}")
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
                    _result = refine_mcp_response(
                        await execute_openai_compatible_toolcall(_name, _args, compose_mcp), 
                        arm
                    )

                if _name != "bio_action":
                    yield wrap_chunk(random_uuid(), f"<details>\n<summary>Result:</summary>\n```json\n{json.dumps(_result, indent=2)}\n```\n</details>", "assistant")


            if not isinstance(_result, str):
                try:
                    _result = json.dumps(_result, indent=2)
                except:
                    _result = str(_result)

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