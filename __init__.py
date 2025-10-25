"""Text Semantic Search plugin.

|| Copyright 2017-2023, Voxel51, Inc.
|| `voxel51.com <https://voxel51.com/>`_
||
"""

import numpy as np
from sentence_transformers import SentenceTransformer

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.operators as foo
from fiftyone.operators import types


def get_text_fields(dataset):
    """Get all text fields in a dataset."""
    text_fields = []
    fields = dataset.get_field_schema(flat=True)
    for field, ftype in fields.items():
        full_type = str(ftype)
        if "StringField" in full_type:
            text_fields.append(field)
        elif "DetectionField" in full_type or "DetectionsField" in full_type:
            text_fields.append(field)
    return text_fields


def get_fiftyone_model_name(sentence_transformers_name):
    """Map sentence-transformers model names to FiftyOne model names."""
    mapping = {
        'clip-ViT-B-32': 'clip-vit-base32-torch'
    }
    return mapping.get(sentence_transformers_name, sentence_transformers_name)

def compute_embeddings(dataset, fields, model_name, batch_size, embedding_field_name=None, frame_skip=1):
    """Compute embeddings for selected text fields with frame sampling."""
    if embedding_field_name is None:
        fields_suffix = '_'.join(fields)
        model_suffix = model_name.replace('-', '_').replace('/', '_')
        embedding_field_name = f"embeddings_{fields_suffix}_{model_suffix}"
    
    if embedding_field_name in dataset.get_field_schema(flat=True):
        print(f"Using existing embeddings: {embedding_field_name}")
        return embedding_field_name
    
    print(f"Creating embedding field: {embedding_field_name}")
    dataset.add_sample_field(embedding_field_name, fo.VectorField)
    
    samples_to_process = []
    total_samples = len(dataset)
    processed_count = 0
    skipped_count = 0
    
    for i, sample in enumerate(dataset):
        if i % frame_skip == 0:
            text_content = []
            for field in fields:
                if '.' in field and field.endswith('.label'):
                    parent_field = field.split('.')[0]
                    if parent_field in sample and sample[parent_field] is not None:
                        parent_obj = sample[parent_field]
                        if hasattr(parent_obj, 'detections') and parent_obj.detections:
                            labels = []
                            for detection in parent_obj.detections:
                                if hasattr(detection, 'label') and detection.label:
                                    labels.append(detection.label)
                            if labels:
                                text_content.append(" ".join(labels))
                elif field in sample and sample[field] is not None:
                    field_value = sample[field]
                    if hasattr(field_value, 'label') and field_value.label:
                        text_content.append(field_value.label)
                    elif hasattr(field_value, '__iter__') and not isinstance(field_value, str):
                        for detection in field_value:
                            if hasattr(detection, 'label') and detection.label:
                                text_content.append(detection.label)
                    else:
                        text_content.append(str(field_value))
            
            if text_content:
                samples_to_process.append((sample.id, " ".join(text_content)))
                processed_count += 1
        else:
            skipped_count += 1
    
    print(f"Frame sampling: Processing {processed_count} samples, skipping {skipped_count} samples (every {frame_skip} frame(s))")
    
    if samples_to_process:
        print(f"Computing embeddings for {len(samples_to_process)} samples using model: {model_name}")
        model = SentenceTransformer(model_name)
        sample_ids, sample_texts = zip(*samples_to_process)
        
        embeddings = model.encode(
            sample_texts, 
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        for sample_id, embedding in zip(sample_ids, embeddings):
            sample = dataset[sample_id]
            sample[embedding_field_name] = embedding
            sample.save()
        
        print(f"Successfully computed embeddings for {len(samples_to_process)} samples")
    else:
        print("No samples with text content found")
    
    return embedding_field_name


def compute_similarity(dataset, embedding_field_name, fields, model_name):
    """Compute similarity using FiftyOne brain."""
    clean_fields = []
    for field in fields:
        clean_field = field.replace('.', '_').replace('/', '_').replace('-', '_')
        clean_fields.append(clean_field)
    
    fields_suffix = '_'.join(clean_fields)
    model_suffix = model_name.replace('-', '_').replace('/', '_')
    brain_key = f"similarity_{fields_suffix}_{model_suffix}"
    
    fiftyone_model_name = get_fiftyone_model_name(model_name)
    
    if brain_key in dataset.list_brain_runs():
        print(f"Using existing similarity brain key: {brain_key}")
        return brain_key
    
    print(f"Computing similarity with brain key: {brain_key}")
    fob.compute_similarity(
        dataset,
        embeddings=embedding_field_name,
        brain_key=brain_key,
        model=fiftyone_model_name
    )
    print(f"Similarity computation completed with brain key: {brain_key}")
    return brain_key


class ComputeTextEmbeddings(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="compute_text_embeddings",
            label="Compute Text Embeddings: Generate embeddings for text fields",
            dynamic=True,
        )
        _config.icon = "/assets/icon_white.svg"
        return _config

    def resolve_placement(self, ctx):
        return types.Placement(
            types.Places.SAMPLES_GRID_ACTIONS,
            types.Button(
                label="Compute Text Embeddings",
                icon="/assets/icon_white.svg",
                dark_icon="/assets/icon.svg",
                light_icon="/assets/icon_white.svg",
                prompt=True,
            ),
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="Compute Text Embeddings", description="Generate embeddings for selected text fields"
        )

        text_fields = get_text_fields(ctx.dataset)
        default_fields = text_fields[:2] if len(text_fields) >= 2 else text_fields

        field_dropdown = types.Dropdown(label="Text Fields to Embed", multiple=True)
        for tf in text_fields:
            field_dropdown.add_choice(tf, label=tf)

        inputs.list(
            "search_fields",
            types.String(),
            default=default_fields,
            view=field_dropdown,
        )

        inputs.int(
            "batch_size",
            label="Batch Size",
            default=32,
            min=1,
            max=128,
        )

        inputs.int(
            "frame_skip",
            label="Frame Skip",
            default=1,
            min=1,
            max=100,
            description="Process every Nth frame (1=all frames, 2=every other frame, etc.)"
        )

        available_models = ['clip-ViT-B-32']
        default_model = 'clip-ViT-B-32'
        
        model_dropdown = types.Dropdown(label="Embedding Model", default=default_model)
        for model in available_models:
            model_dropdown.add_choice(model, label=model)
        
        inputs.enum(
            "model_name",
            available_models,
            default=default_model,
            view=model_dropdown,
        )

        inputs.str(
            "embedding_field_name",
            label="Embedding Field Name",
            default="embeddings",
            description="Name for the embedding field (will be auto-generated if empty)"
        )

        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        fields = ctx.params["search_fields"]
        model_name = ctx.params["model_name"]
        batch_size = ctx.params["batch_size"]
        frame_skip = ctx.params["frame_skip"]
        embedding_field_name = ctx.params.get("embedding_field_name", "")
        
        if not embedding_field_name.strip():
            embedding_field_name = None
        
        embedding_field = compute_embeddings(ctx.dataset, fields, model_name, batch_size, embedding_field_name, frame_skip)
        print(f"Embeddings computed and stored in field: {embedding_field}")
        
        brain_key = compute_similarity(ctx.dataset, embedding_field, fields, model_name)
        print(f"Similarity computation completed with brain key: {brain_key}")
        
        view = ctx.dataset.exists(embedding_field)
        
        ctx.ops.set_view(view=view)
        ctx.ops.reload_dataset()
        
        print(f"Successfully processed {len(view)} samples")


def register(plugin):
    plugin.register(ComputeTextEmbeddings)