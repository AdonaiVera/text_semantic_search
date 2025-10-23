def get_cache():
    g = globals()
    if "_text_semantic_search" not in g:
        g["_text_semantic_search"] = {}

    return g["_text_semantic_search"]
