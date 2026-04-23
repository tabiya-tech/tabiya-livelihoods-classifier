from nel.app.server_dependencies.db_dependencies import ClassifierDBProvider


def test_clear_cache_resets_both_databases():
    # GIVEN both DB references are set to a sentinel value
    ClassifierDBProvider._application_db = object()  # type: ignore[assignment]
    ClassifierDBProvider._taxonomy_db = object()  # type: ignore[assignment]

    # WHEN clear_cache is called
    ClassifierDBProvider.clear_cache()

    # THEN both are reset to None
    assert ClassifierDBProvider._application_db is None
    assert ClassifierDBProvider._taxonomy_db is None
