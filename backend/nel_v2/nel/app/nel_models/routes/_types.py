from pydantic import BaseModel


class NELModelResponse(BaseModel):
    model_id: str
    dimensions: int
    description: str = ""
