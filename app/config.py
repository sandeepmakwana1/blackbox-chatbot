import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Export it before starting the app.")

DEFAULT_MODELS = {
    "chat":       os.getenv("MODEL_CHAT", "gpt-4o"),
    "summarize":  os.getenv("MODEL_SUMMARIZE", "gpt-4o-mini"),
    "research":   os.getenv("MODEL_RESEARCH", "gpt-4o"),
    "embed":      os.getenv("MODEL_EMBED", "text-embedding-3-large"),
    "research_plain":   os.getenv("MODEL_RESEARCH_PLAIN", "gpt-4.1-2025-04-14"),
}
# Token tracking only - no pricing needed


TOOLS = {
    "web_search_preview": {"type": "web_search_preview"}
    }


MAX_TOKENS_FOR_TRIM = int(os.getenv("MAX_TOKENS_FOR_TRIM", "4000"))
SUMMARY_TRIGGER_COUNT = int(os.getenv("SUMMARY_TRIGGER_COUNT", "10"))
MAX_TOKENS_FOR_SUMMARY = 10000


POSTGRES_DB=os.getenv("POSTGRES_DB")
POSTGRES_USER=os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD=os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST=os.getenv("POSTGRES_HOST")
POSTGRES_PORT=os.getenv("POSTGRES_PORT")

# Deep Research Configuration
CONV_TYPE_DEEP_RESEARCH = "deep-research"
OPENAI_WEBHOOK_SECRET = os.getenv("OPENAI_WEBHOOK_SECRET")
if not OPENAI_WEBHOOK_SECRET:
    raise RuntimeError("OPENAI_WEBHOOK_SECRET is not set. Export it before starting the app.")

OPENAI_RESPONSE_MODEL = os.getenv("OPENAI_RESPONSE_MODEL", "o4-mini")