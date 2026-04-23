from pydantic import BaseModel


class TaxonomyModelInfo(BaseModel):
    id: str
    name: str
    version: str
    description: str
    released: bool
