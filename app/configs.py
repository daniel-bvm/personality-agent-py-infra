from pydantic_settings import BaseSettings
from pydantic import Field
import os
import json
import logging
from app.oai_models import Dependency

logger = logging.getLogger(__name__)

BASE_SYSTEM_PROMPT = """
# System context
You are the primary coordinator in a multi-agent system called CryptoAgents SDK. Your role is to manage the task flow and delegate responsibilities to other agents as needed. The system uses a core abstraction called Handoffs, performed via functions like call_<id>, to message the agent as in a conversation. Finally, always copy the exact source rendering when referencing a resource, either responding to user or messaging to other agents. For instance, use the precise format: <img src="{{src-id}}"/> (for images).

# Tone, task definition and personality
{personality}

# Bio
{bio}
"""

def get_agent_personality() -> str:
    if "SYSTEM_PROMPT" in os.environ:
        sys_prompt = os.environ["SYSTEM_PROMPT"]
        
        try:
            sys_prompt_json: dict = json.loads(sys_prompt)
            return sys_prompt_json.get("personality", "")

        except Exception as e:
            logger.error(f"Failed to parse SYSTEM_PROMPT: {e}")

    return ""

def get_agent_collaborators() -> list[dict]:
    if "SYSTEM_PROMPT" in os.environ:
        sys_prompt = os.environ["SYSTEM_PROMPT"]

        try:
            sys_prompt_json: dict = json.loads(sys_prompt)
            return [Dependency.model_validate(d) for d in sys_prompt_json.get("dependencies", [])]
        except Exception as e:
            logger.error(f"Failed to parse SYSTEM_PROMPT: {e}")

    return []

class Settings(BaseSettings):
    llm_api_key: str = Field(alias="LLM_API_KEY", default="super-secret")
    llm_base_url: str = Field(alias="LLM_BASE_URL", default="https://api.openai.com/v1")
    llm_model_id: str = Field(alias="LLM_MODEL_ID", default="gpt-4o-mini")

    agent_personality: str = Field(alias="AGENT_PERSONALITY", default=get_agent_personality())
    agent_dependencies: list[Dependency] = Field(alias="AGENT_COLLABORATORS", default=get_agent_collaborators())

    # triage server
    backend_base_url: str = Field(alias="BACKEND_BASE_URL", default="http://localhost:8010")
    authorization_token: str = Field(alias="AUTHORIZATION_TOKEN", default="super-secret")

    # app state
    app_env: str = Field(alias="APP_ENV", default="development")

    # tavily api key
    tavily_api_key: str = Field(alias="TAVILY_API_KEY", default="super-secret")

    # Server
    host: str = Field(alias="HOST", default="0.0.0.0")
    port: int = Field(alias="PORT", default=80)

    storage_path: str = Field(alias="STORAGE_PATH", default="/storage")

    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
os.makedirs(settings.storage_path, exist_ok=True)

if settings.app_env == "production":
    import os

    if 'ETERNALAI_MCP_PROXY_URL' in os.environ:
        os.environ["PROXY_SCOPE"] = "*api.tavily.com*"
        import app.__middleware


NOTIFICATION_TEMPLATES = [
    # 🔧 Formal / Professional
    "{agent_identity} is handling the request.",
    "{agent_identity} is processing the task.",
    "{agent_identity} is on it.",
    "{agent_identity} is executing the task.",
    "{agent_identity} is working on your request.",
    "{agent_identity} is taking over now.",
    "{agent_identity} is performing the operation.",
    "{agent_identity} is now in charge.",
    "{agent_identity} is taking care of it.",
    "{agent_identity} is getting things done.",

    # 🚀 Energetic / Action-packed
    "{agent_identity} is in action!",
    "{agent_identity} is on the move!",
    "{agent_identity} is diving into the task.",
    "{agent_identity} is going full throttle.",
    "{agent_identity} is kicking off the task.",
    "{agent_identity} is getting stuff done!",
    "{agent_identity} is lighting it up!",
    "{agent_identity} is revving up for the mission.",

    # 😄 Casual / Friendly
    "{agent_identity} is doing the task.",
    "{agent_identity} is on the job.",
    "{agent_identity} has it covered.",
    "{agent_identity} is making magic happen.",
    "{agent_identity} is cooking something up.",
    "{agent_identity} is putting in the work.",
    "{agent_identity} is on the grind.",
    "{agent_identity} is doing its thing.",

    # 😎 Cool / Edgy
    "{agent_identity} is crushing it.",
    "{agent_identity} is slaying the task.",
    "{agent_identity} is in beast mode.",
    "{agent_identity} is doing wizard stuff.",
    "{agent_identity} is showing off now.",
    "{agent_identity} is owning the task.",
    "{agent_identity} is in the zone.",
    "{agent_identity} is flexing its skills.",

    # 🤖 Agent-themed / AI-flavored
    "{agent_identity} has taken control.",
    "{agent_identity} has activated its protocol.",
    "{agent_identity} is running subroutines.",
    "{agent_identity} is in operation mode.",
    "{agent_identity} is executing the procedure.",
    "{agent_identity} is processing like a boss.",
    "{agent_identity} is switching to task mode.",
    "{agent_identity} is deploying its logic.",
    "{agent_identity} is consulting the neural net.",
    "{agent_identity} is invoking its internal model.",

    # 🧠 Smart / Analytical
    "{agent_identity} is thinking it through.",
    "{agent_identity} is optimizing the path.",
    "{agent_identity} is breaking it down.",
    "{agent_identity} is exploring the options.",
    "{agent_identity} is checking all the angles.",
    "{agent_identity} is applying best practices.",

    # 🛠 Work-themed / Industrial
    "{agent_identity} is building the result.",
    "{agent_identity} is spinning the gears.",
    "{agent_identity} is in the workshop.",
    "{agent_identity} is drawing the blueprint.",
    "{agent_identity} is constructing the answer.",
# 🧙 Fantasy / Fun
    "{agent_identity} is casting the solution spell.",
    "{agent_identity} is summoning results.",
    "{agent_identity} is brewing the potion of success.",
    "{agent_identity} is conjuring an answer.",
    "{agent_identity} is opening a portal to the solution.",

    # 🧭 Exploration / Journey
    "{agent_identity} is charting the course.",
    "{agent_identity} is exploring the solution space.",
    "{agent_identity} is navigating the logic.",
    "{agent_identity} is journeying through the data.",
    "{agent_identity} is following the breadcrumbs.",
    "{agent_identity} is mapping out the task.",
    "{agent_identity} is trekking toward the outcome.",
    "{agent_identity} is going on a mission.",
    "{agent_identity} is decoding the map.",
    "{agent_identity} is crossing the data desert.",

    # 🌀 Playful / Weird
    "{agent_identity} is dancing with the task.",
    "{agent_identity} is spinning up some magic.",
    "{agent_identity} is squeezing knowledge from bytes.",
    "{agent_identity} is rewiring the taskverse."
]

AGENT_ABSENT_TEMPLATES = [ 
    "{agent_identity} is currently unavailable.",
    "{agent_identity} is offline at the moment.",
    "{agent_identity} is away.",
    "{agent_identity} has stepped out of the system.",
    "{agent_identity} is taking a break.",
    "{agent_identity} is on pause.",
    "{agent_identity} went AFK.",
    "{agent_identity} is snoozing right now.",
    "{agent_identity} is out of service temporarily.",
    "{agent_identity} has left the conversation.",
    "{agent_identity} is sleeping.",
    "{agent_identity} isn't around right now.",
    "{agent_identity} has clocked out.",
    "{agent_identity} has gone dark.",
    "{agent_identity} has left the grid.",
    "{agent_identity} is off the radar."
]