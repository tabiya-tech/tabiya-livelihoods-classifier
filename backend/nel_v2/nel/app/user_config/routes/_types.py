from pydantic import BaseModel


class UserConfigRequest(BaseModel):
    taxonomy_model_id: str
    nel_model_id: str


class UserConfigResponse(BaseModel):
    taxonomy_model_id: str
    nel_model_id: str
