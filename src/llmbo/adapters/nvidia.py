import logging

from .openai_compatible import OpenAICompatibleAdapter


class NvidiaAdapter(OpenAICompatibleAdapter):
    """Adapter for NVIDIA Nemotron models in AWS Bedrock."""

    logger = logging.getLogger(f"{__name__}.NvidiaAdapter")
    _provider_name = "NVIDIA"
