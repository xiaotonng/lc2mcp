"""Image generation tool using OpenAI's gpt-image-1.5."""

from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from openai import OpenAI

from lc2mcp_community.context import ChatContext, get_context


@tool(parse_docstring=True)
def generate_image(
    prompt: str,
    runtime: ToolRuntime[ChatContext],
    size: str = "1024x1024",
) -> str:
    """Generate an image using OpenAI's gpt-image-1.5 model.

    Args:
        prompt: Description of the image to generate
        size: Image size, one of: 1024x1024, 1536x1024, 1024x1536
    """
    ctx = get_context(runtime)
    _ = ctx.user if ctx else None

    valid_sizes = ["1024x1024", "1536x1024", "1024x1536"]
    if size not in valid_sizes:
        size = "1024x1024"

    try:
        client = OpenAI()
        response = client.images.generate(
            model="gpt-image-1.5",
            prompt=prompt,
            size=size,
            n=1,
        )
        return response.data[0].url
    except Exception as e:
        return f"Image generation error: {str(e)}"
