"""Configuration management using YAML and environment variables."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ServerConfig:
    """Server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    base_url: str = ""


@dataclass
class SecurityConfig:
    """Security configuration."""

    secret_key: str = "dev-secret-key-change-in-production"
    access_token_expire_minutes: int = 1440  # 24 hours


@dataclass
class OAuthConfig:
    """OAuth configuration."""

    client_id: str = "chatgpt-client"
    client_secret: str = "chatgpt-secret-change-in-production"


@dataclass
class DatabaseConfig:
    """Database configuration."""

    url: str = "sqlite+aiosqlite:///./lc2mcp.db"


@dataclass
class UploadConfig:
    """Upload configuration."""

    dir: str = "./uploads"
    max_size_mb: int = 10

    @property
    def max_size_bytes(self) -> int:
        return self.max_size_mb * 1024 * 1024

    @property
    def path(self) -> Path:
        return Path(self.dir)


@dataclass
class DemoUserConfig:
    """Demo user configuration."""

    username: str = "demo"
    password: str = "demo123"
    display_name: str = "Demo User"
    email: str = "demo@example.com"


@dataclass
class ToolsConfig:
    """Tools scanning configuration."""

    external_dirs: list[str] = field(default_factory=list)


@dataclass
class ResourcesConfig:
    """Resources scanning configuration."""

    external_dirs: list[str] = field(default_factory=list)


@dataclass
class Config:
    """Application configuration."""

    server: ServerConfig = field(default_factory=ServerConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    oauth: OAuthConfig = field(default_factory=OAuthConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    upload: UploadConfig = field(default_factory=UploadConfig)
    demo_user: DemoUserConfig = field(default_factory=DemoUserConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)

    # Convenience properties for backward compatibility
    @property
    def host(self) -> str:
        return self.server.host

    @property
    def port(self) -> int:
        return self.server.port

    @property
    def debug(self) -> bool:
        return self.server.debug

    @property
    def base_url(self) -> str:
        return self.server.base_url

    @property
    def secret_key(self) -> str:
        return self.security.secret_key

    @property
    def access_token_expire_minutes(self) -> int:
        return self.security.access_token_expire_minutes

    @property
    def oauth_client_id(self) -> str:
        return self.oauth.client_id

    @property
    def oauth_client_secret(self) -> str:
        return self.oauth.client_secret

    @property
    def database_url(self) -> str:
        return self.database.url

    @property
    def upload_dir(self) -> Path:
        return self.upload.path

    @property
    def max_upload_size(self) -> int:
        return self.upload.max_size_bytes

    @property
    def openai_api_key(self) -> str:
        """OpenAI API key from environment variable."""
        return os.getenv("OPENAI_API_KEY", "")

    def __post_init__(self):
        """Ensure upload directory exists."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)


def _get_config_yaml_path() -> Path:
    """Get path to config.yaml relative to this module."""
    return Path(__file__).parent / "config.yaml"


def load_yaml_config() -> dict[str, Any]:
    """Load configuration from config.yaml if it exists."""
    config_path = _get_config_yaml_path()
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _get_env_override(yaml_value: Any, env_var: str, default: Any = None) -> Any:
    """Get value with environment variable override."""
    env_value = os.getenv(env_var)
    if env_value is not None:
        # Type conversion based on default type
        if isinstance(default, bool):
            return env_value.lower() in ("true", "1", "yes")
        elif isinstance(default, int):
            return int(env_value)
        return env_value
    return yaml_value if yaml_value is not None else default


def _create_config() -> Config:
    """Create Config instance from YAML with environment variable overrides."""
    yaml_data = load_yaml_config()

    # Parse server config
    server_data = yaml_data.get("server", {})
    server_config = ServerConfig(
        host=_get_env_override(server_data.get("host"), "HOST", "0.0.0.0"),
        port=_get_env_override(server_data.get("port"), "PORT", 8000),
        debug=_get_env_override(server_data.get("debug"), "DEBUG", False),
        base_url=_get_env_override(server_data.get("base_url"), "BASE_URL", ""),
    )

    # Parse security config
    security_data = yaml_data.get("security", {})
    security_config = SecurityConfig(
        secret_key=_get_env_override(
            security_data.get("secret_key"),
            "SECRET_KEY",
            "dev-secret-key-change-in-production",
        ),
        access_token_expire_minutes=security_data.get("access_token_expire_minutes", 1440),
    )

    # Parse OAuth config
    oauth_data = yaml_data.get("oauth", {})
    oauth_config = OAuthConfig(
        client_id=_get_env_override(
            oauth_data.get("client_id"), "OAUTH_CLIENT_ID", "chatgpt-client"
        ),
        client_secret=_get_env_override(
            oauth_data.get("client_secret"),
            "OAUTH_CLIENT_SECRET",
            "chatgpt-secret-change-in-production",
        ),
    )

    # Parse database config
    database_data = yaml_data.get("database", {})
    database_config = DatabaseConfig(
        url=_get_env_override(
            database_data.get("url"),
            "DATABASE_URL",
            "sqlite+aiosqlite:///./lc2mcp.db",
        ),
    )

    # Parse upload config
    upload_data = yaml_data.get("upload", {})
    upload_config = UploadConfig(
        dir=_get_env_override(upload_data.get("dir"), "UPLOAD_DIR", "./uploads"),
        max_size_mb=upload_data.get("max_size_mb", 10),
    )

    # Parse demo user config
    demo_data = yaml_data.get("demo_user", {})
    demo_user_config = DemoUserConfig(
        username=demo_data.get("username", "demo"),
        password=demo_data.get("password", "demo123"),
        display_name=demo_data.get("display_name", "Demo User"),
        email=demo_data.get("email", "demo@example.com"),
    )

    # Parse tools config
    tools_data = yaml_data.get("tools", {})
    tools_config = ToolsConfig(
        external_dirs=tools_data.get("external_dirs", []) or [],
    )

    # Parse resources config
    resources_data = yaml_data.get("resources", {})
    resources_config = ResourcesConfig(
        external_dirs=resources_data.get("external_dirs", []) or [],
    )

    return Config(
        server=server_config,
        security=security_config,
        oauth=oauth_config,
        database=database_config,
        upload=upload_config,
        demo_user=demo_user_config,
        tools=tools_config,
        resources=resources_config,
    )


# Global config instance
config = _create_config()
