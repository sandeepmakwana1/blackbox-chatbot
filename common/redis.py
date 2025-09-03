import json
import os
from typing import Optional, Union
from dotenv import load_dotenv
from redis_om.connections import get_redis_connection

from common.logging import get_custom_logger
from common.parameters import get_parameter

logger = get_custom_logger("blackbox.common.redis")

load_dotenv()


class RedisManager:
    """
    Singleton manager for Redis connections.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.connection = cls._initialize_connection()
        return cls._instance

    @staticmethod
    def _initialize_connection():
        """
        Initialize a connection to Redis using parameters from SSM.
        """
        try:
            env_endpoint = os.getenv("REDIS_ENDPOINT")
            if env_endpoint:
                endpoint = env_endpoint
                source = "environment variable"
            else:
                try:
                    endpoint = get_parameter("redis-endpoint")
                    source = "SSM Parameter Store"
                except Exception:
                    logger.warning(
                        "Could not find 'redis-endpoint' in SSM. Falling back."
                    )
                    endpoint = "localhost:6379"

            host, port = endpoint.split(":")
            port = int(port)
            use_ssl = True

            logger.info(
                f"Initializing Redis connection to {host}:{port} (SSL: {use_ssl})"
            )
            return get_redis_connection(
                host=host,
                port=port,
                decode_responses=True,
                ssl=use_ssl,
                ssl_cert_reqs=None,
                socket_timeout=10,
                socket_connect_timeout=10,
            )
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {e}", exc_info=True)
            raise

    @classmethod
    def conn(cls):
        return cls().__new__(cls).connection  # singleton instance


class RedisService:
    """
    Service for interacting with Redis by redis_key (string keys).

    Usage:
        # CREATE / SET
        key = "lambda:response:latest"
        RedisService.create(key, "Hello World")

        # READ / GET
        val = RedisService.get(key)
        print("Got:", val)

        # UPDATE (same as create, but returns False if key didn't exist)
        ok = RedisService.update(key, "New Value")
        print("Updated:", ok)

        # DELETE
        gone = RedisService.delete(key)
        print("Deleted:", gone)
    """

    @staticmethod
    def create(redis_key: str, value: str) -> str:
        """
        Create or overwrite a string value at redis_key.
        Returns the key on success.
        """
        RedisManager.conn().set(redis_key, value)
        return redis_key

    @staticmethod
    def get(redis_key: str) -> Optional[str]:
        """
        Get the string value stored at redis_key.
        Returns None if not found.
        """
        return RedisManager.conn().get(redis_key)

    @staticmethod
    def update(redis_key: str, new_value: str) -> bool:
        """
        Overwrite an existing key. Returns True if the key existed.
        """
        r = RedisManager.conn()
        if not r.exists(redis_key):
            return False
        r.set(redis_key, new_value)
        return True

    @staticmethod
    def delete(redis_key: str) -> bool:
        """
        Delete a key. Returns True if a key was deleted.
        """
        return RedisManager.conn().delete(redis_key) > 0

    @staticmethod
    def publish(redis_channel: str, message: dict) -> bool:
        """
        Publish a message to a Redis channel.
        """
        return RedisManager.conn().publish(redis_channel, json.dumps(message))

    @staticmethod
    def get_redis_key(source_id: Union[int, str], stage: str) -> str:
        """
        Get a standardized Redis key for RFP data.

        Args:
            source_id: The source ID (can be integer or string)
            stage: The processing stage name

        Returns:
            Formatted Redis key string
        """
        return f"source_id:{source_id}:stage:{stage}"

    @staticmethod
    def fetch_rfp_data_from_redis(source_id: Union[int, str], stage: str) -> str:
        """
        Fetch RFP data for a given stage from Redis using the source_id.
        """
        logger.info(f"Fetching RFP {stage} from redis for source_id: {source_id}")
        try:
            rfp_text = RedisService.get(f"source_id:{source_id}:stage:{stage}")
            if rfp_text:
                return rfp_text
            logger.warning(f"No valid {stage} found for source_id: {source_id}")
        except Exception as e:
            logger.error(
                f"Failed to fetch RFP {stage} from redis for source_id: {source_id}",
                exc_info=True,
            )
        return ""

    @staticmethod
    def key_exists(redis_key: str) -> bool:
        """
        Check if a key exists in Redis.
        Returns True if the key exists, False otherwise.
        """
        return RedisManager.conn().exists(redis_key) > 0
