import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Export it before starting the app.")

DEFAULT_MODELS = {
    "chat": os.getenv("MODEL_CHAT", "gpt-4.1-2025-04-14"),
    "summarize": os.getenv("MODEL_SUMMARIZE", "gpt-4.1-mini"),
    "research": os.getenv("MODEL_RESEARCH", "gpt-4o"),
    "embed": os.getenv("MODEL_EMBED", "text-embedding-3-large"),
    "research_plain": os.getenv("MODEL_RESEARCH_PLAIN", "gpt-4.1-2025-04-14"),
}
# Token tracking only - no pricing needed
OPENAI_RESPONSE_MODEL = os.getenv(
    "OPENAI_RESPONSE_MODEL", "o4-mini"
)
# o4-mini-deep-research-2025-06-26
OPTIMIZER_MODEL = os.getenv("OPTIMIZER_MODEL", "gpt-4o-mini")


TOOLS = {"web_search_preview": {"type": "web_search_preview"}}


_raw_max = int(os.getenv("MAX_TOKENS_FOR_TRIM", "180000"))
# Clamp to a sane range to avoid exceeding model context windows
MAX_TOKENS_FOR_TRIM = max(1000, min(_raw_max, 200000))
SUMMARY_TRIGGER_COUNT = int(os.getenv("SUMMARY_TRIGGER_COUNT", "10"))
MAX_TOKENS_FOR_SUMMARY = 180000


POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")

# Deep Research Configuration
CONV_TYPE_DEEP_RESEARCH = "deep-research"
OPENAI_WEBHOOK_SECRET = os.getenv("OPENAI_WEBHOOK_SECRET")
if not OPENAI_WEBHOOK_SECRET:
    raise RuntimeError(
        "OPENAI_WEBHOOK_SECRET is not set. Export it before starting the app."
    )
