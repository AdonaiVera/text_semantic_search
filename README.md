## Text Semantic Search Plugin


This plugin is a Python plugin that allows you to search through your dataset using semantic similarity on text fields instead of exact keyword matching.


## Installation

```shell
fiftyone plugins download https://github.com/AdonaiVera/text_semantic_search
```

## Requirements

```shell
fiftyone plugins requirements @adonaivera/text_semantic_search --install
```

## Operators

### `text_semantic_search`

**Description**: Search for samples by semantic meaning using text embeddings

**Inputs**:

- `query`: The text query to search for semantically
- `search_fields`: The text fields to search in (multiple text fields supported)
- `similarity_threshold`: Minimum similarity score (0.0-1.0)
- `top_k`: Maximum number of results to return
- `batch_size`: Batch size for embedding generation (1-128, default: 32)
