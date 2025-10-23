"""Text Semantic Search plugin.

|| Copyright 2017-2023, Voxel51, Inc.
|| `voxel51.com <https://voxel51.com/>`_
||
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

import fiftyone as fo
from fiftyone.core.utils import add_sys_path
import fiftyone.operators as foo
from fiftyone.operators import types
from fiftyone import ViewField as F

# Initialize NLTK WordNet data
try:
    import nltk
    nltk.download('wordnet', quiet=True)
    from nltk.corpus import wordnet
    WORDNET_AVAILABLE = True
except:
    WORDNET_AVAILABLE = False


def _is_teams_deployment():
    val = os.environ.get("FIFTYONE_INTERNAL_SERVICE", "")
    return val.lower() in ("true", "1")


TEAMS_DEPLOYMENT = _is_teams_deployment()


if not TEAMS_DEPLOYMENT:
    with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
        from cache_manager import get_cache


def get_text_fields(dataset):
    """Get all text fields in a dataset."""
    text_fields = []
    fields = dataset.get_field_schema(flat=True)
    for field, ftype in fields.items():
        full_type = str(ftype)
        if "StringField" in full_type:
            text_fields.append(field)
    return text_fields


def get_embedding_model():
    """Get or initialize the embedding model."""
    return SentenceTransformer('all-MiniLM-L6-v2')


def get_synonyms(word):
    """Get synonyms using pre-loaded NLTK WordNet."""
    if WORDNET_AVAILABLE:
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' ').lower())
        return list(synonyms)
    else:
        # Fallback to simple synonyms if NLTK not available
        fallback_synonyms = {
            'car': ['car', 'vehicle', 'auto', 'automobile'],
            'night': ['night', 'dark', 'evening'],
            'day': ['day', 'daytime', 'bright', 'sunny'],
            'building': ['building', 'house', 'structure'],
            'person': ['person', 'people', 'pedestrian']
        }
        return fallback_synonyms.get(word.lower(), [word])


def simple_keyword_match(text, query):
    """Enhanced keyword matching with WordNet synonyms."""
    text_lower = text.lower()
    query_lower = query.lower()
    
    # Check direct match first
    if query_lower in text_lower:
        return 1.0
    
    # Get synonyms for each word in query
    query_words = query_lower.split()
    all_synonyms = set(query_words)
    
    for word in query_words:
        synonyms = get_synonyms(word)
        all_synonyms.update(synonyms)
    
    # Check if any synonym appears in text
    for synonym in all_synonyms:
        if synonym in text_lower:
            return 0.8
    
    return 0.0


def get_or_create_embeddings(dataset, fields, model, batch_size=32):
    """Get existing embeddings or create new ones for samples."""
    embedding_field = f"embeddings_{'_'.join(fields)}"
    
    if embedding_field not in dataset.get_field_schema(flat=True):
        print(f"Creating embedding field: {embedding_field}")
        dataset.add_sample_field(embedding_field, fo.VectorField)
    
    samples_to_process = []
    existing_embeddings = {}
    
    for sample in dataset:
        if embedding_field in sample and sample[embedding_field] is not None:
            existing_embeddings[sample.id] = sample[embedding_field]
        else:
            text_content = []
            for field in fields:
                if field in sample and sample[field] is not None:
                    text_content.append(str(sample[field]))
            
            if text_content:
                samples_to_process.append((sample.id, " ".join(text_content)))
    
    if samples_to_process:
        print(f"Generating embeddings for {len(samples_to_process)} samples...")
        sample_ids, sample_texts = zip(*samples_to_process)
        new_embeddings = []
        
        for i in range(0, len(sample_texts), batch_size):
            batch_texts = sample_texts[i:i + batch_size]
            batch_embeddings = model.encode(
                batch_texts, 
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            new_embeddings.extend(batch_embeddings)
        
        new_embeddings = np.array(new_embeddings)
        
        for sample_id, embedding in zip(sample_ids, new_embeddings):
            sample = dataset[sample_id]
            sample[embedding_field] = embedding
            sample.save()
            existing_embeddings[sample_id] = embedding
    
    return existing_embeddings, embedding_field


def semantic_search(dataset, fields, query_text, threshold=0.3, top_k=100, batch_size=32):
    """Perform hybrid semantic search using embeddings + keyword matching."""
    model = get_embedding_model()
    
    sample_embeddings, embedding_field = get_or_create_embeddings(dataset, fields, model, batch_size)
    
    if not sample_embeddings:
        return dataset.limit(0)
    
    query_embedding = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )[0]
    
    sample_ids = list(sample_embeddings.keys())
    embeddings_matrix = np.array([sample_embeddings[sid] for sid in sample_ids])
    embedding_similarities = np.dot(embeddings_matrix, query_embedding)
    
    # Add keyword matching scores
    combined_scores = []
    for i, sample_id in enumerate(sample_ids):
        sample = dataset[sample_id]
        text_content = []
        for field in fields:
            if field in sample and sample[field] is not None:
                text_content.append(str(sample[field]))
        
        full_text = " ".join(text_content)
        keyword_score = simple_keyword_match(full_text, query_text)
        
        # Combine embedding similarity (0.7 weight) + keyword score (0.3 weight)
        combined_score = 0.7 * embedding_similarities[i] + 0.3 * keyword_score
        combined_scores.append(combined_score)
    
    combined_scores = np.array(combined_scores)
    
    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    top_similarities = combined_scores[top_indices]
    valid_indices = top_indices[top_similarities >= threshold]
    valid_ids = [sample_ids[i] for i in valid_indices]
    valid_similarities = top_similarities[top_similarities >= threshold]
    
    view = dataset.select(valid_ids)
    
    similarity_field = f"similarity_{'_'.join(fields)}"
    if similarity_field not in dataset.get_field_schema(flat=True):
        dataset.add_sample_field(similarity_field, fo.FloatField)
    
    for sample_id, similarity in zip(valid_ids, valid_similarities):
        sample = dataset[sample_id]
        sample[similarity_field] = float(similarity)
        sample.save()
    
    return view


class TextSemanticSearch(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="text_semantic_search",
            label="Text Semantic Search: Find samples by text meaning",
            dynamic=True,
        )
        _config.icon = "/assets/icon_white.svg"
        return _config

    def resolve_placement(self, ctx):
        return types.Placement(
            types.Places.SAMPLES_GRID_ACTIONS,
            types.Button(
                label="Text Semantic Search",
                icon="/assets/icon_white.svg",
                dark_icon="/assets/icon.svg",
                light_icon="/assets/icon_white.svg",
                prompt=True,
            ),
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="Text Semantic Search", description="Find samples by text meaning"
        )

        text_fields = get_text_fields(ctx.dataset)

        if not TEAMS_DEPLOYMENT:
            cache = get_cache()
            if "fields" in cache:
                default_fields = cache["fields"]
            else:
                default_fields = text_fields[:2] if len(text_fields) >= 2 else text_fields
        else:
            default_fields = text_fields[:2] if len(text_fields) >= 2 else text_fields

        field_dropdown = types.Dropdown(label="Fields to search within", multiple=True)
        for tf in text_fields:
            field_dropdown.add_choice(tf, label=tf)

        inputs.list(
            "search_fields",
            types.String(),
            default=default_fields,
            view=field_dropdown,
        )

        inputs.float(
            "similarity_threshold",
            label="Similarity Threshold",
            default=0.3,
            min=0.0,
            max=1.0,
            description="Minimum similarity score (0.0-1.0). Higher values = more strict matching. Results will include similarity scores for filtering."
        )

        inputs.int(
            "top_k",
            label="Max Results",
            default=100,
            min=1,
            max=1000,
        )

        inputs.int(
            "batch_size",
            label="Batch Size",
            default=32,
            min=1,
            max=128,
        )

        new_default_fields = ctx.params.get("search_fields", default_fields)

        if not TEAMS_DEPLOYMENT:
            get_cache()["fields"] = new_default_fields

        inputs.str("query", label="Search Query", required=True)
        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        query = ctx.params["query"]
        fields = ctx.params["search_fields"]
        threshold = ctx.params["similarity_threshold"]
        top_k = ctx.params["top_k"]
        batch_size = ctx.params["batch_size"]
        view = semantic_search(ctx.dataset, fields, query, threshold, top_k, batch_size)
        
        similarity_field = f"similarity_{'_'.join(fields)}"
        if similarity_field in ctx.dataset.get_field_schema(flat=True):
            view = view.sort_by(similarity_field, reverse=True)
        
        ctx.ops.set_view(view=view)
        ctx.ops.reload_dataset()
        
        num_results = len(view)
        if num_results > 0:
            scores = [sample[similarity_field] for sample in view if similarity_field in sample and sample[similarity_field] is not None]
            if scores:
                max_score = max(scores)
                min_score = min(scores)
                avg_score = sum(scores) / len(scores)
                print(f"Found {num_results} results with similarity scores ranging from {min_score:.3f} to {max_score:.3f} (avg: {avg_score:.3f})")
            else:
                print(f"Found {num_results} results")
        else:
            print(f"No results found above threshold {threshold}. Try lowering the threshold or using different search terms.")


def register(plugin):
    plugin.register(TextSemanticSearch)