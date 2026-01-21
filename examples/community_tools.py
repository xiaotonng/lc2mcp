"""
Example: Using langchain_community tools with MCP.

This demonstrates how to expose standard LangChain community tools
like DuckDuckGo search and Wikipedia as MCP tools.

Requirements:
    pip install langchain-community duckduckgo-search wikipedia
"""

from fastmcp import FastMCP

from lc2mcp import register_tools

# Import standard LangChain tools
try:
    from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
except ImportError:
    print("Please install required packages:")
    print("  pip install langchain-community duckduckgo-search wikipedia")
    exit(1)


mcp = FastMCP("knowledge-server")

# Initialize the tools
search_tool = DuckDuckGoSearchRun()
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Register them all at once
register_tools(mcp, [search_tool, wiki_tool])

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8002)
