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


S3_PATH_TEMPLATES: Dict[str, str] = {
    "rfp_text": "{source_id}/rfp_text.json",
    "prompt_data": "{source_id}/prompt_data.json",
    "validation_result": "{source_id}/stage/validation/validation_results.json",
    "deep_research": "{source_id}/stage/deep_research/deep_research.json",
    "deep_research_queries": "{source_id}/stage/deep_research/deep_research_queries.json",
    "company_data": "{source_id}/stage/pre_steps/company_data.json",
    "user_summary": "{source_id}/stage/pre_steps/user_summary.json",
    "system_summary": "{source_id}/stage/pre_steps/system_summary.json",
    "user_preference": "{source_id}/stage/user_preferences/user_preferences.json",
    "table_of_content": "{source_id}/stage/pre_steps/table_of_content.json",
    "agency_references": "{source_id}/stage/pre_steps/agency_references.json",
    "cost_summary": "{source_id}/stage/costing/cost_summary.json",
    "rfp_license": "{source_id}/stage/costing/rfp_license.json",
    "hourly_wages": "{source_id}/stage/costing/hourly_wages.json",
    "rfp_infrastructure": "{source_id}/stage/costing/rfp_infrastructure.json",
    "cost_field_name": "{source_id}/stage/costing/cost_field_name.json",
    "cost_user_preference": "{source_id}/stage/costing/cost_user_preference.json",
    "content": "{source_id}/stage/content/content.json",
    "long_term_memory": "{source_id}/stage/content/long_term_memory.json",
}

S3_KEY_PREFIX = "proposals/"
