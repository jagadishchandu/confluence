from pydantic_settings import BaseSettings
 
 
class Settings(BaseSettings):
    CONFLUENCE_BASE_URL: str
    CONFLUENCE_EMAIL: str
    CONFLUENCE_API_TOKEN: str
    CONFLUENCE_SPACE_KEY: str
 
    AWS_REGION: str
    BEDROCK_EMBED_MODEL_ID: str
    BEDROCK_CHAT_MODEL_ID: str
 
    QDRANT_URL: str
    QDRANT_COLLECTION: str = "confluence_chunks"
 
    UI_ORIGIN: str = "http://localhost:8080"
 
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 150
 
    class Config:
        env_file = ".env"
 
 
settings = Settings()