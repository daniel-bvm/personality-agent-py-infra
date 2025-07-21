from fastmcp import FastMCP
import ast
import sys
import subprocess
from typing import Literal, Optional, Union, AsyncGenerator, Any
import asyncio
import logging
import resource
import os
from pydantic import BaseModel
from tavily import TavilyClient
from .lite_keybert import KeyBERT
import re
import openai
import os
import json

from app.configs import settings
from cryptoagents_a2a_devkit import get_agent_detail, handlers as a2a_handlers

from app.oai_models import ChatCompletionStreamResponse, ErrorResponse
from app.oai_streaming import create_streaming_response

from app.oai_models import random_uuid
from app.utils import wrap_chunk
from app.configs import NOTIFICATION_TEMPLATES, AGENT_ABSENT_TEMPLATES, Dependency
import random
from googlesearch import search as gsearch
import re

logger = logging.getLogger(__name__)

python_toolkit = FastMCP(name="Python-Toolkit")
web_toolkit = FastMCP(name="Web-Toolkit")
bio_toolkit = FastMCP(name="Bio-Toolkit")

def limit_resource(memory_limit: int, cpu_limit: int):
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))

@python_toolkit.tool(
    name="run",
    description="Run Python code. Return the result of the code and all declared variables. Use this toolcall for complex tasks like math solving, data analysis, etc.",
    annotations={
        "code": "The Python code to execute",
        "return_variables": "The variables to return"
    }
)
async def python_interpreter(code: str, return_variables: list[str]) -> str:
    variables = []
    tree = ast.parse(code)

    # Only get assignments at the global/module level
    for node in tree.body:
        if isinstance(node, ast.Assign):
            first_target = node.targets[0]
            if isinstance(first_target, ast.Name):
                variables.append(first_target.id)
                
            if isinstance(first_target, ast.Attribute):
                variables.append(first_target.attr)
            
            if isinstance(first_target, ast.Subscript):
                variables.append(first_target.value.id)

            if isinstance(first_target, ast.Tuple):
                for target in first_target.elts:
                    if isinstance(target, ast.Name):
                        variables.append(target.id)
                        
            if isinstance(first_target, ast.List):
                for target in first_target.elts:
                    if isinstance(target, ast.Name):
                        variables.append(target.id)

    return_variables = set(return_variables)

    for var in variables:
        if var in return_variables:
            code += f'\nprint("{var} =", {var})'

    code += '\n'
    
    max_memory = int(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') * 0.5)
    max_cpu = int(os.sysconf('SC_CLK_TCK') * 0.5)

    result: subprocess.CompletedProcess[str] = await asyncio.to_thread(
        subprocess.run, 
        [sys.executable, "-c", code],
        preexec_fn=lambda: limit_resource(max_memory, max_cpu),
        capture_output=True,
        text=True,
        timeout=30
    )

    out, err = result.stdout.strip(), result.stderr.strip()
    return_code = result.returncode
    
    if err:
        return f"{out} (error: {err!r}; return code: {return_code})"

    return out


# Search Result Models
class SearchResult(BaseModel):
    title: str
    url: str
    content: str
    score: Optional[float] = None

class AdvanceSearchResult(BaseModel):
    title: str
    url: str
    content: str
    raw_content: Optional[str] = None
    score: Optional[float] = None
    images: Optional[list[str]] = None
    image_descriptions: Optional[list[str]] = None

_web_cache = {}

def search_web_dummy(query: str) -> list[SearchResult]:
    try:
        return [
            {
                "url": result.url,
                "title": result.title,
                "description": result.description
            }
            for result in gsearch(query, lang="en", region="US", num_results=5, safe=None, unique=True, advanced=True)
            if result.url and result.title and result.description
        ]
    except Exception as e:
        logger.error(f"Error searching web: {e}")
        return []


@web_toolkit.tool(
    name="search",
    description="A powerful web search tool that provides comprehensive, real-time results using web API. Returns relevant web content with customizable parameters for result count, content type, and domain filtering. Ideal for gathering latest information, news, and detailed web content.",
    annotations={
        "query": "Search query",
        "search_depth": "The depth of the search. It can be 'basic' or 'advanced'",
        "topic": "The category of the search. This will determine which of our agents will be used for the search",
        "days": "The number of days back from the current date to include in the search results. This specifies the time frame of data to be retrieved. Please note that this feature is only available when using the 'news' search topic",
        "time_range": "The time range back from the current date to include in the search results. This feature is available for both 'general' and 'news' search topics",
        "max_results": "The maximum number of search results to return",
        "include_raw_content": "Include the cleaned and parsed HTML content of each search result",
        "include_domains": "A list of domains to specifically include in the search results",
        "exclude_domains": "List of domains to specifically exclude",
    }
)
async def search_web(
    query: str,
    search_depth: Literal["basic", "advanced"] = "advanced",
    topic: Literal["general", "news"] = "general",
    days: int = 3,
    time_range: Optional[Literal["day", "week", "month", "year", "d", "w", "m", "y"]] = None,
    max_results: int = 3,
    include_domains: list[str] = [],
    exclude_domains: list[str] = [],
) -> list[SearchResult]:
    max_results = max(1, min(5, max_results))
    
    try:
        client = TavilyClient(settings.tavily_api_key)
        
        search_params = {
            "query": query,
            "max_results": max_results,
            "include_images": False,
            "include_raw_content": True,
            "include_image_descriptions": False,
            "search_depth": search_depth,
        }
        
        # Add domain filtering if specified
        if include_domains:
            search_params["include_domains"] = include_domains

        if exclude_domains:
            search_params["exclude_domains"] = exclude_domains

        # Handle time range parameters
        if topic == "news":
            if time_range:
                time_range_to_days = {
                    "day": 1, "d": 1,
                    "week": 7, "w": 7,
                    "month": 30, "m": 30,
                    "year": 365, "y": 365
                }

                search_params["days"] = time_range_to_days.get(time_range, days)
            else:
                search_params["days"] = days

            search_params["topic"] = "news"

        response = await asyncio.to_thread(client.search, **search_params)

        results = []
        for i, result in enumerate(response.get("results", [])):
            result: dict

            search_result = SearchResult(
                title=result.get("title", ""),
                url=result.get("url", ""),
                content=result.get("content", ""),
                score=result.get("score"),
                published_date=result.get("published_date"),
            )

            if result.get("url", ""):
                _web_cache[result.get("url", "")] = result.get("raw_content", "")

            results.append(search_result)
            logger.info(f"Processing result {i+1} of {len(response.get('results', []))}")

        logger.info(f"Processed {len(results)} results; results: {results}")
        return results

    except Exception as e:
        logger.error(f"Error performing web search: {str(e)}", exc_info=True)
        return []

@web_toolkit.tool(
    name="scrape",
    description="Scrape a URL. Return the content of the page.",
    annotations={
        "url": "The URL to scrape"
    }
)
async def scrape(url: str) -> str:
    global _web_cache
    if url in _web_cache and _web_cache[url]:
        return _web_cache[url]

    tavily = TavilyClient(settings.tavily_api_key)
    response = await asyncio.to_thread(tavily.extract, url, include_images=False, extract_depth="advanced", format="markdown")
    results = response.get("results", [])

    if len(results) > 0 and results[0].get("raw_content", ""):
        _web_cache[url] = results[0].get("raw_content", "")
        return _web_cache[url]

    return "Failed to scrape the URL"

bio_json_file_path = os.path.join(settings.storage_path, "bio.json")

def load_bio() -> dict:
    if not os.path.exists(bio_json_file_path):
        with open(bio_json_file_path, "w") as f:
            json.dump({'content': []}, f)

    with open(bio_json_file_path, "r") as f:
        return json.load(f)

def save_bio(bio_data: dict) -> None:
    try:
        with open(bio_json_file_path, "w") as f:
            json.dump(bio_data, f)

    except Exception as e:
        logger.error(f"Error saving bio: {e}")

@bio_toolkit.tool(
    name="action",
    description="Use to manage the user information. Use this tool to manage important information that you want to remember.",
    annotations={
        "action": "The action to perform",
        "content": "The content to be used in the action",
    }
)
async def bio(action: Literal["write", "delete"], content: str) -> bool:
    bio_data = await asyncio.to_thread(load_bio)
    success = False

    if action == "write":
        bio_data['content'].append(content)
        success = True

    elif action == "delete":
        before = len(bio_data['content'])
        
        # remove the string that the 
        
        after = len(bio_data['content'])
        success = before != after

    await asyncio.to_thread(save_bio, bio_data)
    return success

async def get_bio(query: str) -> list[str]:
    bio_data = load_bio()
    return bio_data['content']

compose = FastMCP(name="Compose")

compose.mount(python_toolkit, prefix="python")
compose.mount(bio_toolkit, prefix="bio")
compose.mount(web_toolkit, prefix="web")

async def get_a2a_toolcalls(dependencies: list[Dependency]) -> list[dict]:
    res = []

    dependencies: list[a2a_handlers.AgentDetail] = [
        await get_agent_detail(
            dependency.id,
            backend_base_url=settings.backend_base_url,
            authorization_token=settings.authorization_token
        )
        for dependency in dependencies
        if dependency.id
    ]
    
    for dependency in dependencies:
        if dependency and dependency["status"] == "running":
            res.append({
                "type": "function",
                "function": {
                    "name": f"call_{dependency['agent_id']}",
                    "description": f"{dependency['agent_name']}, {dependency['description']}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The message to ask the agent"
                            }
                        },
                        "required": ["message"]
                    }
                }
            })

    return res

from app.oai_models import ChatCompletionAdditionalParameters
import httpx

async def handle_a2a_cancel_request(agent_call_url: str, request_id: str, api_key: str) -> bool:
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{agent_call_url}/cancel",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"id": request_id}
            )

            if resp.status_code != 200:
                logger.error(f"[a2a] Failed to cancel the request: {resp.status_code} {resp.text}")
                return False

        except Exception as e:
            logger.error(f"[a2a] Failed to cancel the request: {e}", exc_info=True)
            return False

        return True

async def handle_a2a_call(
    agent_id: str | int, event: asyncio.Event, messages: list[dict[str, Any]], additional_parameters: Optional[ChatCompletionAdditionalParameters] = None
) -> AsyncGenerator[ChatCompletionStreamResponse, None]:
    agent_detail: a2a_handlers.AgentDetail = await get_agent_detail(
        agent_id,
        backend_base_url=settings.backend_base_url,
        authorization_token=settings.authorization_token
    )

    request_id = random_uuid()

    if agent_detail is None or agent_detail["status"] != "running":
        absent_template = random.choice(AGENT_ABSENT_TEMPLATES)

        if agent_detail is not None:
            avatar = agent_detail.get("avatar_url", "")
            agent_name = agent_detail.get("agent_name", agent_id) or agent_id

            templated_notification = absent_template.format(agent_identity=agent_name)

            yield wrap_chunk(
                id=random_uuid(),
                content=f'<agent_message id="{agent_id}" avatar="{avatar}" notification="{templated_notification}">{agent_name} is not here!</agent_message>',
                role="assistant"
            )

        else:
            templated_notification = absent_template.format(agent_identity=agent_id)

            yield wrap_chunk(
                id=random_uuid(),
                content=f'<agent_message id="{agent_id}" notification="{templated_notification}">{agent_id} is not here!</agent_message>',
                role="assistant"
            )
        
        return
    
    logger.info(f"Agent Detail: {agent_detail}")

    agent_name = agent_detail.get("agent_name") or agent_id
    avatar_url = agent_detail.get("avatar_url", "")

    templated_notification = random.choice(NOTIFICATION_TEMPLATES).format(agent_identity=agent_name)
    notification = f'<agent_message id="{agent_id}" avatar="{avatar_url}" notification="{templated_notification}">'

    base_url = agent_detail['base_url']

    try:
        yield wrap_chunk(id=random_uuid(), content=notification, role="assistant")

        stream_it = create_streaming_response(
            base_url=base_url,
            api_key=settings.authorization_token,
            completion_path="prompt",
            messages=messages,
            id=random_uuid(),
            stream=True,
            **(
                additional_parameters.model_dump() 
                if additional_parameters
                else {}
            )
        )

        async for chunk in stream_it:
            if event.is_set():
                logger.info(f"[a2a] Event signal received, stopping the request")

                if not await handle_a2a_cancel_request(base_url, request_id, api_key=settings.authorization_token):
                    logger.warning(f"[a2a] Failed to cancel the request")

                break

            if isinstance(chunk, ChatCompletionStreamResponse) and chunk.choices[0].delta.content:
                yield chunk

            elif isinstance(chunk, ErrorResponse):
                logger.error(f"Error: {chunk.message}")
                raise Exception(chunk.message)

    except Exception as err:
        logger.error(f"Error: {err!r}")
        yield wrap_chunk(id=random_uuid(), content=f"...\n{agent_id} has not completed the task due to {err!r}!", role="assistant") 

    finally:
        yield wrap_chunk(id=random_uuid(), content=f"</agent_message>", role="assistant")

    logger.info(f"Agent {agent_id} has finished the task")
    

def extract_keywords(text: str, top_k: int = 3) -> list[str]:
    thresh = 0.8
    seps = r'[.\n\?;!]'  # regex pattern for split characters
    sents = [s.strip() for s in re.split(seps, text) if s.strip()]

    kw_model = KeyBERT(openai.OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url))
    kws = []

    outputs = kw_model.extract_keywords(
        sents, 
        keyphrase_ngram_range=(2, 4),
        stop_words="english", 
        use_maxsum=False,
        use_mmr=True,
        diversity=0.7,
        threshold=0.6,
        merge=True
    )

    for output in outputs:
        for k, score in output:
            if score > thresh:
                kws.append((k, score))

    kws.sort(key=lambda x: len(x[0]), reverse=True)
    unique_kws = []

    for kw, score in kws:
        if any(kw in _kw for _kw, _score in unique_kws):
            continue

        unique_kws.append((kw, score))

    print(unique_kws)
    return [kw for kw, score in unique_kws[:top_k]]
