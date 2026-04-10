"""Dependency injection factory for INERService."""

from ner.service import INERService, NERService


def get_ner_service() -> INERService:
    import ner.main as _main_module
    return NERService(model=_main_module.ner_model)
