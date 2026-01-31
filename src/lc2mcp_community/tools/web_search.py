"""Web search tool using DuckDuckGo."""

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from lc2mcp_community.context import ChatContext, get_context


@tool(parse_docstring=True)
def web_search(
    query: str,
    runtime: ToolRuntime[ChatContext],
) -> str:
    """Search the web for real-time information.

    Args:
        query: The search query to look up
    """
    from ddgs import DDGS

    ctx = get_context(runtime)
    _ = ctx.user if ctx else None

    try:
        results = DDGS().text(query, max_results=5)
        if not results:
            return "No results found."

        formatted = []
        for r in results:
            formatted.append(f"**{r.get('title', 'Untitled')}**")
            formatted.append(f"{r.get('body', '')}")
            formatted.append(f"URL: {r.get('href', '')}")
            formatted.append("")

        return "\n".join(formatted)
    except Exception as e:
        return f"Search error: {str(e)}"
