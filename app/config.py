import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Export it before starting the app.")

DEFAULT_MODELS = {
    "chat":       os.getenv("MODEL_CHAT", "gpt-4o"),
    "summarize":  os.getenv("MODEL_SUMMARIZE", "gpt-4o-mini"),
    "research":   os.getenv("MODEL_RESEARCH", "gpt-4o"),
    "embed":      os.getenv("MODEL_EMBED", "text-embedding-3-large"),
}
# Token tracking only - no pricing needed

MAX_TOKENS_FOR_TRIM = int(os.getenv("MAX_TOKENS_FOR_TRIM", "4000"))
SUMMARY_TRIGGER_COUNT = int(os.getenv("SUMMARY_TRIGGER_COUNT", "10"))
MAX_TOKENS_FOR_SUMMARY = 10000



