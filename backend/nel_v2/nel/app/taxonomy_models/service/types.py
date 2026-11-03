from pydantic import BaseModel


class TaxonomyModelInfo(BaseModel):
    id: str
    name: str
    version: str
    description: str
    released: bool
    # Locale short code of the taxonomy (e.g. "EU-en", "AR-es"); models that share a name
    # are only told apart by it. Empty when the taxonomy API does not report one.
    locale: str = ""
