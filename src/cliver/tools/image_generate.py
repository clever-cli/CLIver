"""Built-in tool for generating images from text descriptions.

Uses AgentCore.generate() with an image-capable model.
"""

import logging
import os

from cliver.tool import tool

logger = logging.getLogger(__name__)


def _find_image_model(requested: str = "") -> tuple[str | None, str]:
    """Find an image-capable model from config.

    Returns (model_key, available_list).  model_key is None if no image
    models are configured.  available_list is a comma-separated list
    for use in error messages.
    """
    from cliver.config import ConfigManager
    from cliver.util import get_config_dir

    cm = ConfigManager(get_config_dir())
    image_models = cm.config.models.get("image", {})
    available = ", ".join(image_models.keys()) if image_models else ""

    if not image_models:
        return None, available

    if requested and requested in image_models:
        return requested, available

    # Return the first image model as fallback
    return next(iter(image_models.keys()), None), available


@tool(
    name="ImageGenerate",
    description=(
        "Generate an image from a text description and save it to disk. "
        'The `model` parameter is optional — leave it blank ("") to auto-select '
        "the first available image model from the Configured Models list. "
        "Only pass a model name if you verified it exists in Configured Models. "
        "Use `output_dir` to save to a user-specified directory. "
        "Use when the user asks to create, draw, or generate an image."
    ),
)
async def image_generate(
    prompt: str,
    model: str = "",
    output_dir: str = "",
) -> list[dict]:
    """Generate an image from a text description.

    Args:
        prompt: Text description of the image to generate.
        model: Image model name. Pick from the Available Models list (image category).
            Leave empty to auto-select the first image-capable model.
        output_dir: Directory to save generated images.
    """
    from cliver.llm.agent_core import AgentCore
    from cliver.provider.providers import create_provider

    requested = model.strip()
    model_key, available = _find_image_model(requested)
    if not model_key:
        return [{"error": "No image-capable model configured. Add an image model in config.yaml."}]

    # If LLM asked for a model that doesn't exist, auto-select the available one
    if requested and requested not in available.split(", "):
        logger.info("Model '%s' not found, auto-selecting '%s'", requested, model_key)

    from cliver.config import ConfigManager
    from cliver.util import get_config_dir

    cm = ConfigManager(get_config_dir())
    mc = cm.all_models().get(model_key)
    if not mc:
        return [{"error": f"Image model '{model_key}' not found. Available: {available}"}]

    pc = cm.get_provider_config(mc.provider)
    provider = create_provider(
        api_key=pc.get_api_key() if pc else None,
        base_url=mc.get_resolved_url(pc) or None,
        protocol=mc.get_provider_protocol(pc),
        provider_class=mc.provider,
        user_agent=cm.config.user_agent,
    )

    agent_core = AgentCore(provider=provider, model=mc.api_model_name)

    try:
        save_dir = output_dir.strip() or os.path.join(os.getcwd(), ".cliver", "generated-images")
        response = await agent_core.generate(prompt=prompt, media_type="image", output_dir=save_dir)

        if not response.media:
            return [{"text": response.message.text or "Image generation completed but no media was returned."}]

        paths = [m.saved_path or m.data for m in response.media]
        return [{"text": f"Generated {len(paths)} image(s):\n" + "\n".join(paths)}]

    except Exception as e:
        logger.warning("Image generation failed: %s", e)
        return [{"error": f"Error generating image: {e}"}]
