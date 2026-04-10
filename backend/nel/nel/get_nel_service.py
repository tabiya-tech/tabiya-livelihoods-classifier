"""Dependency injection factory for INELService."""

from nel.service import INELService, NELService


def get_nel_service() -> INELService:
    import nel.main as _main_module
    return NELService(linker=_main_module.nel_linker, max_top_k=_main_module.MAX_TOP_K)
