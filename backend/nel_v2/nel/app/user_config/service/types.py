from pydantic import BaseModel


class UserConfig(BaseModel):
    user_id: str
    taxonomy_model_id: str
    nel_model_id: str
