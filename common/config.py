import os
from typing import Any, Dict

from common.parameters import get_parameter


def get_elastic_config() -> Dict[str, Any]:
    """Read Elastic APM settings from environment & SSM."""
    try:
        server_url = get_parameter("APM_SERVER_URL")
        secret_token = get_parameter("APM_SECRET_TOKEN")
        api_key = get_parameter("ELASTIC_APM_API_KEY")
    except Exception:
        raise ValueError(f" Error : Elastic credentials not found")

    return {
        "SERVICE_NAME": os.getenv("APM_SERVICE_NAME", "blackbox-aws-services"),
        "SERVER_URL": server_url,
        "ENVIRONMENT": os.getenv("SSM_PREFIX", "dev").strip("blackbox-"),
        "SECRET_TOKEN": secret_token,
        "API_KEY": api_key,
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        "CAPTURE_BODY": os.getenv("ELASTIC_APM_CAPTURE_BODY", "errors"),
        "TRANSACTION_SAMPLE_RATE": float(
            os.getenv("ELASTIC_APM_TRANSACTION_SAMPLE_RATE", "1.0")
        ),
    }
