import json
import os
from datetime import datetime
from common.datastore import store_s3_data, S3StoreService
from common.config import S3_PATH_TEMPLATES, S3_KEY_PREFIX

PRICING = {
    "gpt-5": {"input": 1.25, "cached": 0.125, "output": 10.00},
    "o3": {"input": 2.00, "cached": 0.50, "output": 8.00},
    "o4-mini": {"input": 1.10, "cached": 0.275, "output": 4.40},
    "gpt-4o": {"input": 2.50, "cached": 1.25, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.40, "cached": 0.10, "output": 1.60},
    "gemini-3-pro": {"input": 2.00, "cached": 0.20, "output": 12.00},
    "gemini-2.5-pro": {"input": 1.25, "cached": 0.125, "output": 10.00},
    "text-embedding-3-small": {"input": 0.02, "cached": 0.00, "output": 0.00},
}


def calculate_and_store_usage(logger, cb, source_id, stage_name, request_id):
    """Calculates cost and stores usage JSON. Self-contained to avoid config errors."""
    try:
        model_name = os.environ.get("model", "gpt-5")

        # 1. Match rates
        base_model = (
            "gpt-5"
            if "gpt-5" in model_name
            else ("gpt-4o" if "gpt-4o" in model_name else "gpt-4.1-mini")
        )
        rates = PRICING.get(base_model, PRICING["gpt-5"])

        # 2. Extract specific 2026 usage details
        prompt_tokens = cb.prompt_tokens
        completion_tokens = cb.completion_tokens
        cached_tokens = getattr(cb, "prompt_tokens_details", {}).get("cached_tokens", 0)
        reasoning_tokens = getattr(cb, "completion_tokens_details", {}).get(
            "reasoning_tokens", 0
        )

        # 3. Calculate actual USD cost
        cost = (
            ((prompt_tokens - cached_tokens) / 1000000 * rates["input"])
            + (cached_tokens / 1000000 * rates["cached"])
            + (completion_tokens / 1000000 * rates["output"])
        )

        token_metrics = {
            "source_id": source_id,
            "stage": stage_name,
            "model": model_name,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_cost_usd": round(cost, 6),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # 4. Resolve path with safety to prevent NoneType error
        current_date = datetime.utcnow().strftime("%Y-%m-%d")
        bucket_name = S3StoreService.get_bucket_name()

        template = S3_PATH_TEMPLATES.get("llm_usage_log")

        if not template:
            relative_key = f"usage/dt={current_date}/{source_id}_{stage_name}.json"
        else:
            relative_key = template.format(
                date_str=current_date,
                source_id=source_id,
                stage_name=stage_name,
                request_id=request_id,
            )

        full_s3_key = f"{relative_key}"

        # 5. Save to S3
        success = store_s3_data(full_s3_key, bucket_name, token_metrics)

        if success:
            logger.info(f"USAGE_LOGGED: s3://{bucket_name}/{full_s3_key}")
        else:
            logger.error(f"FAILED_TO_LOG_USAGE: {request_id}")

        return token_metrics

    except Exception as e:
        logger.error(f"Error in usage tracking: {str(e)}")
        return {}


def calculate_and_store_playground_usage(
    logger, cb, source_id, stage_name, request_id, message_text=None
):
    """Append-style usage logger for playground threads, optionally capturing user message text."""
    try:
        model_name = os.environ.get("model", "gpt-5")

        # 1. Match rates
        base_model = (
            "gpt-5"
            if "gpt-5" in model_name
            else ("gpt-4o" if "gpt-4o" in model_name else "gpt-4.1-mini")
        )
        rates = PRICING.get(base_model, PRICING["gpt-5"])

        # 2. Extract specific 2026 usage details
        prompt_tokens = cb.prompt_tokens
        completion_tokens = cb.completion_tokens
        cached_tokens = getattr(cb, "prompt_tokens_details", {}).get("cached_tokens", 0)
        reasoning_tokens = getattr(cb, "completion_tokens_details", {}).get(
            "reasoning_tokens", 0
        )

        # 3. Calculate actual USD cost
        cost = (
            ((prompt_tokens - cached_tokens) / 1000000 * rates["input"])
            + (cached_tokens / 1000000 * rates["cached"])
            + (completion_tokens / 1000000 * rates["output"])
        )

        token_metrics = {
            "source_id": source_id,
            "stage": stage_name,
            "model": model_name,
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_cost_usd": round(cost, 6),
            "timestamp": datetime.utcnow().isoformat(),
        }
        if message_text:
            token_metrics["message"] = str(message_text)[:2000]

        # 4. Resolve path with safety to prevent NoneType error
        current_date = datetime.utcnow().strftime("%Y-%m-%d")
        bucket_name = S3StoreService.get_bucket_name()

        template = S3_PATH_TEMPLATES.get("llm_usage_log")

        if not template:
            relative_key = f"usage/dt={current_date}/{source_id}_{stage_name}"
            if request_id:
                relative_key = f"{relative_key}_{request_id}"
            relative_key = f"{relative_key}.json"
        else:
            relative_key = template.format(
                date_str=current_date,
                source_id=source_id,
                stage_name=stage_name,
                request_id=request_id,
            )

        full_s3_key = f"{relative_key}"

        # Append semantics: load existing list, append, rewrite
        existing = []
        try:
            import boto3
            from botocore.exceptions import ClientError, BotoCoreError

            client = boto3.client("s3")
            obj = client.get_object(Bucket=bucket_name, Key=full_s3_key)
            body = obj["Body"].read().decode("utf-8")
            data = json.loads(body)
            if isinstance(data, list):
                existing = data
        except Exception:
            existing = []

        existing.append(token_metrics)

        success = store_s3_data(full_s3_key, bucket_name, existing)

        if success:
            logger.info(f"USAGE_LOGGED: s3://{bucket_name}/{full_s3_key}")
        else:
            logger.error(f"FAILED_TO_LOG_USAGE: {request_id}")

        return token_metrics

    except Exception as e:
        logger.error(f"Error in playground usage tracking: {str(e)}")
        return {}
