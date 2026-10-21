from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Ollama 本地模型
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "qwen2.5:latest"
    code_model: str = "qwen2.5-coder:14b"  # 代码生成专用；未拉取时自动回退 chat_model
    # 语义理解专用（PLAN 澄清 / spec 合并 / 报错裁判 / 升级提问）：
    # 质疑和口径推理是 14B 相对 7B 提升最大的能力，故单独配置；未拉取时自动回退 chat_model
    planner_model: str = "qwen2.5:14b"
    embed_model: str = "nomic-embed-text:latest"

    # 阿里云百炼 DashScope（可选）
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_model: str = "qwen-plus"

    # SiliconFlow 视频生成（可选，AIGC 视频功能用）
    siliconflow_api_key: str = ""
    siliconflow_video_model: str = "Wan-AI/Wan2.2-T2V-A14B"  # 文生视频；图生视频换 Wan2.2-I2V

    # 服务
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # 数据目录
    data_dir: str = "./data"
    chroma_dir: str = "./data/chroma"
    # 远程 Chroma 服务地址（如 http://<内网Chroma地址>:8091）；留空则用本地 chroma_dir
    chroma_http_url: str = ""

    @property
    def has_cloud_key(self) -> bool:
        return bool(self.dashscope_api_key.strip())

    @property
    def samples_dir(self) -> Path:
        return Path(self.data_dir) / "samples"

    @property
    def uploads_dir(self) -> Path:
        return Path(self.data_dir) / "uploads"

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_dir)


settings = Settings()
