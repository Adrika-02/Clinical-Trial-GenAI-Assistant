"""LLM factory for the LangChain agents. Backs onto AWS Bedrock (Claude via
Bedrock) or the direct Anthropic API, selected by the LLM_PROVIDER env var.

The project targets AWS Bedrock as the production LLM backend (see
.env.example), but Bedrock's Claude models require an AWS Marketplace
subscription that itself requires a valid payment instrument on the AWS
account. When that isn't available, LLM_PROVIDER=anthropic routes the exact
same LangChain agent code through the direct Anthropic API instead — a
one-line env change swaps back to Bedrock once billing is sorted, since
every agent only ever calls get_llm() and never touches boto3/Bedrock
directly.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

_llm = None


def _build_bedrock_llm(temperature: float) -> BaseChatModel:
    from langchain_aws import ChatBedrockConverse

    return ChatBedrockConverse(
        model=os.environ["BEDROCK_MODEL_ID"],
        region_name=os.environ["AWS_REGION"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        temperature=temperature,
    )


def _build_anthropic_llm(temperature: float) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"),
        api_key=os.environ["ANTHROPIC_API_KEY"],
        temperature=temperature,
    )


def _build_groq_llm(temperature: float) -> BaseChatModel:
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        api_key=os.environ["GROQ_API_KEY"],
        temperature=temperature,
    )


_PROVIDER_BUILDERS = {
    "bedrock": _build_bedrock_llm,
    "anthropic": _build_anthropic_llm,
    "groq": _build_groq_llm,
}


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """Return a cached chat model configured from .env.

    LLM_PROVIDER selects "bedrock", "anthropic", or "groq". Every agent in
    this project calls only get_llm() and never touches a provider SDK
    directly, so switching providers (e.g. back to Bedrock once AWS
    Marketplace billing is set up) is a one-line .env change.

    Bedrock model access is region-scoped and Claude models frequently
    require a cross-region inference profile ID (e.g.
    'eu.anthropic.claude-sonnet-4-5-20250929-v1:0') rather than a bare
    model ID — see .env.example for how to discover the correct one.
    """
    global _llm
    if _llm is None:
        provider = os.environ.get("LLM_PROVIDER", "bedrock").lower()
        builder = _PROVIDER_BUILDERS.get(provider, _build_bedrock_llm)
        _llm = builder(temperature)
    return _llm


if __name__ == "__main__":
    llm = get_llm()
    resp = llm.invoke("Reply with exactly: Bedrock connection OK")
    print(resp.content)
