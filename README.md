# lc2mcp

[![PyPI version](https://img.shields.io/pypi/v/lc2mcp.svg)](https://pypi.org/project/lc2mcp/)
[![Python versions](https://img.shields.io/pypi/pyversions/lc2mcp.svg)](https://pypi.org/project/lc2mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Convert LangChain tools to FastMCP tools — in one line of code.**

> Stop rewriting your tools. Just adapt them.

`lc2mcp` is a lightweight adapter that converts existing **LangChain tools** into **FastMCP tools**, enabling you to quickly build MCP servers accessible to Claude, Cursor, and any MCP-compatible client.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔄 **Instant Conversion** | One function call to convert any LangChain tool to FastMCP tool |
| 📦 **Ecosystem Access** | Unlock 1000+ LangChain community tools (Search, Wikipedia, SQL, APIs...) |
| 🎯 **Zero Boilerplate** | Automatic Pydantic → JSON Schema conversion |
| 🔐 **Context Injection** | Pass auth, user info, and request context to tools |
| 📊 **Progress & Logging** | Full support for MCP progress notifications and logging |
| 🏷️ **Namespace Support** | Prefix tool names and handle conflicts automatically |

---

## 🚀 Quick Start

### Installation

```bash
pip install lc2mcp
```

### 3 Lines to MCP

```python
from langchain_core.tools import tool
from fastmcp import FastMCP
from lc2mcp import register_tools

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 25°C in {city}"

mcp = FastMCP("weather-server")
register_tools(mcp, [get_weather])  # ← That's it!

if __name__ == "__main__":
    mcp.run()
```

Your tool is now available to Claude, Cursor, and any MCP client.

---

## 🔌 How It Works

```
┌─────────────────┐      ┌─────────────┐      ┌─────────────────┐
│  LangChain Tool │ ───▶ │   lc2mcp    │ ───▶ │  FastMCP Tool   │
│  (@tool, etc.)  │      │  (adapter)  │      │                 │
└─────────────────┘      └─────────────┘      └────────┬────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │  FastMCP Server │
                                              └────────┬────────┘
                                                       │
                                                       ▼
                                          ┌───────────────────────┐
                                          │      MCP Clients      │
                                          │ (Claude, Cursor, ...) │
                                          └───────────────────────┘
```

---

## 📚 Examples

### Using Community Tools

Instantly expose DuckDuckGo search and Wikipedia to MCP clients:

```bash
pip install lc2mcp langchain-community duckduckgo-search wikipedia
```

```python
from fastmcp import FastMCP
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from lc2mcp import register_tools

mcp = FastMCP("knowledge-server")

register_tools(mcp, [
    DuckDuckGoSearchRun(),
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
])

if __name__ == "__main__":
    mcp.run()
```

### With Authentication Context

Inject user authentication and app context into your tools:

```python
from dataclasses import dataclass
from fastmcp import Context, FastMCP
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from lc2mcp import register_tools

@dataclass(frozen=True)
class UserContext:
    user_id: str
    tenant_id: str

@tool
def whoami(runtime: ToolRuntime[UserContext]) -> str:
    """Return the current user."""
    return f"Hello, user {runtime.context.user_id} from {runtime.context.tenant_id}"

def runtime_adapter(mcp_ctx: Context) -> ToolRuntime[UserContext]:
    return ToolRuntime(
        context=UserContext(
            user_id=mcp_ctx.get_state("user_id") or "anonymous",
            tenant_id=mcp_ctx.get_state("tenant_id") or "default",
        ),
        state={}, config={}, stream_writer=lambda x: None,
        tool_call_id=None, store=None,
    )

mcp = FastMCP("auth-server")
register_tools(mcp, [whoami], runtime_adapter=runtime_adapter)

if __name__ == "__main__":
    mcp.run()
```

### With Progress Reporting & Logging

Use MCP context for real-time progress updates and logging:

```python
from fastmcp import Context, FastMCP
from langchain_core.tools import tool
from lc2mcp import register_tools

@tool
async def process_data(data: str, mcp_ctx: Context) -> str:
    """Process data with progress reporting."""
    await mcp_ctx.info(f"Starting: {data}")
    await mcp_ctx.report_progress(0, 100, "Starting")
    
    # ... processing steps ...
    await mcp_ctx.report_progress(50, 100, "Processing")
    
    await mcp_ctx.info("Complete!")
    await mcp_ctx.report_progress(100, 100, "Done")
    return f"Processed: {data}"

mcp = FastMCP("processor")
register_tools(mcp, [process_data], inject_mcp_ctx=True)

if __name__ == "__main__":
    mcp.run()
```

### Namespace & Conflict Handling

Organize tools with prefixes and handle name collisions:

```python
from fastmcp import FastMCP
from lc2mcp import register_tools

mcp = FastMCP("multi-domain")

# Prefix all finance tools
register_tools(mcp, finance_tools, name_prefix="finance.")

# Auto-suffix on collision: tool → tool_2 → tool_3
register_tools(mcp, ops_tools, name_prefix="ops.", on_name_conflict="suffix")

if __name__ == "__main__":
    mcp.run()
```

---

## 📖 API Reference

### `register_tools()`

Convert and register LangChain tools as FastMCP tools on a server.

```python
register_tools(
    mcp: FastMCP,
    tools: list[BaseTool | Callable],
    *,
    name_prefix: str | None = None,           # e.g. "finance." → "finance.get_stock"
    on_name_conflict: str = "error",          # "error" | "overwrite" | "suffix"
    inject_mcp_ctx: bool = False,             # inject mcp_ctx: Context
    runtime_adapter: Callable | None = None,  # Context → ToolRuntime[...]
)
```

### `to_mcp_tool()`

Convert a single LangChain tool to FastMCP tool for manual registration.

```python
to_mcp_tool(
    tool: BaseTool | Callable,
    *,
    name: str | None = None,
    description: str | None = None,
    args_schema: Type[BaseModel] | None = None,
    inject_mcp_ctx: bool = False,
    runtime_adapter: Callable | None = None,
) -> Callable
```

---

## 🔧 Compatibility

| Component | Supported Versions |
|-----------|-------------------|
| Python | 3.10, 3.11, 3.12+ |
| LangChain | >= 1.0.0 |
| FastMCP | >= 2.0.0 |

### Tool Support

| Tool Type | Status |
|-----------|--------|
| `@tool` decorated functions | ✅ Full support |
| `StructuredTool` | ✅ Full support |
| `BaseTool` subclasses | ✅ Supported (requires `args_schema`) |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Made with ❤️ for the LangChain and MCP communities</b>
</p>
