## Text Semantic Search Plugin
![demo](https://github.com/user-attachments/assets/124330f9-b564-4aae-9fe8-67a5ca0606a4)


This plugin enables semantic search through your FiftyOne datasets by generating text embeddings and creating similarity indices. It supports both regular text fields and detection label fields, allowing you to search for samples based on semantic meaning rather than exact keyword matching.

## How It Works

1. **Text Extraction**: Extracts text content from selected fields including:
   - Regular string fields (`caption`, `description`, etc.)
   - Detection label fields (`ground_truth.detections.label`, `predictions.detections.label`)
   - Nested detection objects with automatic label extraction

2. **Embedding Generation**: Uses CLIP models to convert text into high-dimensional vector embeddings

3. **Similarity Indexing**: Creates FiftyOne brain similarity indices for fast semantic search

4. **Search Integration**: Enables semantic search through FiftyOne's built-in search interface

## Installation

```shell
fiftyone plugins download https://github.com/AdonaiVera/text_semantic_search
```

## Requirements

```shell
fiftyone plugins requirements @adonaivera/text_semantic_search --install
```

## Operators

### `compute_text_embeddings`

**Description**: Generate embeddings for text fields and create similarity indices for semantic search

**Inputs**:

- `search_fields`: Text fields to embed (supports detection fields like `ground_truth.detections.label`)
- `model_name`: Embedding model to use (default: `clip-ViT-B-32`)
- `batch_size`: Batch size for embedding generation (1-128, default: 32)
- `frame_skip`: Process every Nth frame (1=all frames, 2=every other frame, etc.)
- `embedding_field_name`: Custom name for embedding field (auto-generated if empty)

**Supported Field Types**:
- String fields: `caption`, `description`, `tags`
- Detection fields: `ground_truth.detections.label`, `predictions.detections.label`
- Any field containing text content

## Usage

1. **Load your dataset** in FiftyOne
2. **Run the plugin** from the samples grid actions
3. **Select text fields** to embed (detection fields will automatically extract labels)
4. **Configure parameters** (model, batch size, frame skip)
5. **Generate embeddings** and similarity indices
6. **Use FiftyOne's search** to perform semantic queries

## Example

After running the plugin on a dataset with detection fields:
- `ground_truth.detections.label`: `['car', 'person', 'bicycle']`
- `predictions.detections.label`: `['vehicle', 'pedestrian', 'bike']`

You can then search semantically for:
- "automobile" → finds samples with cars/vehicles
- "people walking" → finds samples with persons/pedestrians
- "two-wheeled transport" → finds samples with bicycles/bikes

## Features

- ✅ **Detection Field Support**: Automatically extracts labels from detection objects
- ✅ **Multiple Field Types**: Handles strings, detections, and nested fields
- ✅ **Smart Caching**: Reuses existing embeddings and similarity indices
- ✅ **Frame Sampling**: Process subsets of large datasets efficiently
- ✅ **CLIP Integration**: Uses state-of-the-art vision-language models
- ✅ **FiftyOne Integration**: Seamless integration with FiftyOne's search interface
