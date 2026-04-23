class Collections:
    # Application DB — user/admin data, model registry, cache status
    NEL_MODELS = "nel_models"
    NEL_QUALIFICATIONS = "nel_qualifications"
    EMBEDDINGS_CACHE_STATUS = "nel_embeddings_cache_status"

    # Taxonomy DB — Atlas collections with vector search indexes
    OCCUPATION_EMBEDDINGS = "nel_occupation_embeddings"
    SKILL_EMBEDDINGS = "nel_skill_embeddings"
    QUALIFICATION_EMBEDDINGS = "nel_qualification_embeddings"
