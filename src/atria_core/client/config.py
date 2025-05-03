from pydantic_settings import BaseSettings

from atria_core.utilities.imports import _get_atria_core_base_path


class Settings(BaseSettings):
    SERVICE_NAME: str = "atria"
    ATRIAX_URL: str
    ATRIAX_API_KEY: str

    class Config:
        env_file = _get_atria_core_base_path() + "/../../.env"


settings = Settings()
