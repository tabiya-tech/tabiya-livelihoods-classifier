"""Dependency injection factory for INERService."""

from ner.service import INERService, NERService


def get_ner_service() -> INERService:
    import ner.main as _main_module

    # A provider rather than a fixed model: one service instance serves every language,
    # and a language's model is loaded on the first request that needs it.
    return NERService(model_provider=_main_module.get_model)
