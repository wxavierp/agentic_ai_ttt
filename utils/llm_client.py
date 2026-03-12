import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_client() -> OpenAI:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    return OpenAI(
        base_url=endpoint,
        api_key=api_key,
    )


def get_deployment_name() -> str:
    return os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
