class PipelineRepositoryError(Exception):
    """Base for pipeline repository errors."""


class PipelineNotFoundError(PipelineRepositoryError):
    """The pipeline_id does not belong to this user or does not exist."""

    def __init__(self, pipeline_id: str, user_id: str) -> None:
        super().__init__(
            f"Pipeline '{pipeline_id}' not found for user '{user_id}'."
        )
        self.pipeline_id = pipeline_id
        self.user_id = user_id


class ReadonlyPipelineError(PipelineRepositoryError):
    """Update or delete attempted on a pipeline flagged `is_readonly=True`.

    Applies to the seeded Default Tabiya pipeline: users can clone it to
    edit, but not modify or delete it in place.
    """

    def __init__(self, pipeline_id: str) -> None:
        super().__init__(
            f"Pipeline '{pipeline_id}' is read-only and cannot be modified."
        )
        self.pipeline_id = pipeline_id
