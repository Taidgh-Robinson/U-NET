import os
from pydantic_settings import BaseSettings, SettingsConfigDict

DOTENV = os.path.join(os.path.dirname(__file__), ".env")


class Settings(BaseSettings):
    logger_level: str = "INFO"
    num_epochs: int = 5

    model_config = SettingsConfigDict(env_file=DOTENV)


settings = Settings()

NUM_EPOCHS = settings.num_epochs
