class NELModelNotFoundError(Exception):
    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"NEL model not found: {model_id!r}")
