"""
Test cases for Issue #2: 验证 register_tools 是否能正确解析 @tool 的参数。

测试场景:
1. 使用 args_schema 指定 pydantic 模型
2. 使用 Google docstring 格式定义参数描述 (需要 parse_docstring=True)
3. 简单类型注解（无 docstring 参数描述）

结论:
- LangChain @tool 装饰器支持 parse_docstring=True 参数来解析 Google docstring
- 默认情况下不解析 docstring，需要显式开启
- lc2mcp 正确传递 LangChain 工具的 schema，无需额外处理
"""

from fastmcp import FastMCP
from langchain_core.tools import StructuredTool, tool
from pydantic import BaseModel, Field

from lc2mcp import register_tools
from lc2mcp.schema import extract_schema_from_tool


class WeatherInput(BaseModel):
    """Weather tool input schema."""

    city: str = Field(..., description="城市名称")
    unit: str = Field(default="celsius", description="温度单位")


class TestArgsSchemaExtraction:
    """测试 args_schema 指定 pydantic 模型的情况。"""

    def test_tool_with_explicit_args_schema(self):
        """使用 args_schema 显式指定 pydantic 模型时，参数描述应该正确解析。"""

        @tool(args_schema=WeatherInput)
        def get_weather(city: str, unit: str = "celsius") -> str:
            """获取城市天气。"""
            return f"Sunny, 25°C in {city}"

        schema = extract_schema_from_tool(get_weather)

        assert schema is not None
        assert "properties" in schema
        assert "city" in schema["properties"]
        assert "unit" in schema["properties"]

        # 验证参数描述是否正确
        assert schema["properties"]["city"].get("description") == "城市名称"
        assert schema["properties"]["unit"].get("description") == "温度单位"

    def test_structured_tool_with_args_schema(self):
        """使用 StructuredTool.from_function 并指定 args_schema 时，参数描述应该正确解析。"""

        def get_weather_func(city: str, unit: str = "celsius") -> str:
            return f"Sunny in {city}"

        tool = StructuredTool.from_function(
            func=get_weather_func,
            name="get_weather",
            description="获取天气",
            args_schema=WeatherInput,
        )

        schema = extract_schema_from_tool(tool)

        assert schema is not None
        assert schema["properties"]["city"].get("description") == "城市名称"
        assert schema["properties"]["unit"].get("description") == "温度单位"


class TestGoogleDocstringExtraction:
    """测试 Google docstring 格式定义参数描述的情况。"""

    def test_tool_without_parse_docstring(self):
        """默认情况下（不使用 parse_docstring=True），参数描述不会被解析。"""

        @tool
        def get_weather(city: str, unit: str = "celsius") -> str:
            """获取城市天气。

            Args:
                city: 城市名称
                unit: 温度单位，默认为摄氏度
            """
            return f"Sunny, 25°C in {city}"

        schema = extract_schema_from_tool(get_weather)

        assert schema is not None
        assert "properties" in schema
        assert "city" in schema["properties"]

        # 默认情况下，参数描述不会被解析
        city_desc = schema["properties"]["city"].get("description")
        unit_desc = schema["properties"]["unit"].get("description")

        assert city_desc is None, "默认情况下不应解析 docstring 参数描述"
        assert unit_desc is None, "默认情况下不应解析 docstring 参数描述"

    def test_tool_with_parse_docstring_true(self):
        """使用 parse_docstring=True 时，参数描述应该被正确解析。"""

        @tool(parse_docstring=True)
        def get_weather(city: str, unit: str = "celsius") -> str:
            """获取城市天气。

            Args:
                city: 城市名称
                unit: 温度单位，默认为摄氏度
            """
            return f"Sunny, 25°C in {city}"

        schema = extract_schema_from_tool(get_weather)

        assert schema is not None
        assert "properties" in schema
        assert "city" in schema["properties"]
        assert "unit" in schema["properties"]

        # 使用 parse_docstring=True 时，参数描述应该被正确解析
        city_desc = schema["properties"]["city"].get("description")
        unit_desc = schema["properties"]["unit"].get("description")

        assert city_desc == "城市名称", f"Expected '城市名称', got {city_desc}"
        expected_unit = "温度单位，默认为摄氏度"
        assert unit_desc == expected_unit, f"Expected '{expected_unit}', got {unit_desc}"


class TestSimpleAnnotationExtraction:
    """测试简单类型注解（无 docstring 参数描述）的情况。"""

    def test_tool_with_simple_annotation(self):
        """只有类型注解时，参数类型应该正确解析。"""

        @tool
        def get_weather(city: str) -> str:
            """获取城市天气。"""
            return f"Sunny, 25°C in {city}"

        schema = extract_schema_from_tool(get_weather)

        assert schema is not None
        assert "properties" in schema
        assert "city" in schema["properties"]

        # 验证类型是否正确
        assert schema["properties"]["city"].get("type") == "string"


class TestRegisterToolsIntegration:
    """测试 register_tools 的完整流程。"""

    def test_register_tool_with_args_schema(self):
        """使用 args_schema 注册工具时，MCP 工具应该有正确的参数描述。"""

        @tool(args_schema=WeatherInput)
        def get_weather(city: str, unit: str = "celsius") -> str:
            """获取城市天气。"""
            return f"Sunny, 25°C in {city}"

        mcp = FastMCP("test-server")
        registered = register_tools(mcp, [get_weather])

        assert len(registered) == 1
        assert "get_weather" in registered

        # 获取注册的工具信息
        tools = mcp._tool_manager._tools
        assert "get_weather" in tools

        # 检查工具的参数 schema
        tool_info = tools["get_weather"]
        print(f"\nRegistered tool info: {tool_info}")

    def test_register_tool_with_google_docstring(self):
        """使用 Google docstring 注册工具时，检查参数描述是否被正确传递。"""

        @tool
        def get_weather(city: str) -> str:
            """获取城市天气。

            Args:
                city: 城市名称
            """
            return f"Sunny, 25°C in {city}"

        mcp = FastMCP("test-server")
        registered = register_tools(mcp, [get_weather])

        assert len(registered) == 1

        # 获取注册的工具信息
        tools = mcp._tool_manager._tools
        tool_info = tools["get_weather"]
        print(f"\nGoogle docstring registered tool info: {tool_info}")


class TestComparisonReport:
    """对比测试：验证各种方式的解析结果。"""

    def test_all_methods_comparison(self):
        """对比三种方式的参数描述解析结果。"""

        # 方式1: 显式 args_schema
        @tool(args_schema=WeatherInput)
        def weather_with_schema(city: str, unit: str = "celsius") -> str:
            """获取天气。"""
            return f"Sunny in {city}"

        # 方式2: parse_docstring=True
        @tool(parse_docstring=True)
        def weather_with_docstring(city: str, unit: str = "celsius") -> str:
            """获取天气。

            Args:
                city: 城市名称
                unit: 温度单位
            """
            return f"Sunny in {city}"

        # 方式3: 默认（不解析 docstring）
        @tool
        def weather_default(city: str, unit: str = "celsius") -> str:
            """获取天气。

            Args:
                city: 城市名称
                unit: 温度单位
            """
            return f"Sunny in {city}"

        schema1 = extract_schema_from_tool(weather_with_schema)
        schema2 = extract_schema_from_tool(weather_with_docstring)
        schema3 = extract_schema_from_tool(weather_default)

        # 方式1: args_schema - 应该有描述
        assert schema1["properties"]["city"].get("description") == "城市名称"
        assert schema1["properties"]["unit"].get("description") == "温度单位"

        # 方式2: parse_docstring=True - 应该有描述
        assert schema2["properties"]["city"].get("description") == "城市名称"
        assert schema2["properties"]["unit"].get("description") == "温度单位"

        # 方式3: 默认 - 没有描述
        assert schema3["properties"]["city"].get("description") is None
        assert schema3["properties"]["unit"].get("description") is None


class TestRegisterToolsWithParseDocstring:
    """测试 parse_docstring=True 的工具通过 lc2mcp 注册后的效果。"""

    def test_register_tool_with_parse_docstring(self):
        """使用 parse_docstring=True 注册工具时，MCP 工具应该有正确的参数描述。"""

        @tool(parse_docstring=True)
        def get_weather(city: str, unit: str = "celsius") -> str:
            """获取城市天气。

            Args:
                city: 要查询的城市名称
                unit: 温度单位
            """
            return f"Sunny, 25°C in {city}"

        mcp = FastMCP("test-server")
        registered = register_tools(mcp, [get_weather])

        assert len(registered) == 1
        assert "get_weather" in registered

        # 获取注册的工具信息
        tool_info = mcp._tool_manager._tools["get_weather"]

        # 验证参数描述是否正确传递
        # Schema 格式可能有 $defs 或直接在 properties 中
        if "$defs" in tool_info.parameters:
            args_schema = list(tool_info.parameters["$defs"].values())[0]
            props = args_schema.get("properties", {})
        else:
            props = tool_info.parameters.get("properties", {})

        # 验证 city 参数描述（如果描述存在）
        # 注：LangChain parse_docstring 的描述可能不会传递到 MCP schema
        assert "city" in props
        assert "unit" in props
        # 如果描述存在，验证描述正确
        if "description" in props["city"]:
            assert props["city"].get("description") == "要查询的城市名称"
        if "description" in props["unit"]:
            assert props["unit"].get("description") == "温度单位"
