"""Dependency injection factory for INELService."""

from nel.service import INELService, NELService


def get_nel_service() -> INELService:
    import nel.main as _main_module

    # A provider rather than a fixed linker: one service instance serves every language,
    # and a language's linker is loaded on the first request that needs it.
    return NELService(
        linker_provider=_main_module.get_linker,
        max_top_k=_main_module.MAX_TOP_K,
    )
