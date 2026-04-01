import os
import json
import torch


def load_model(model_path):
    """Load the trained model from the specified path."""
    model = torch.load(model_path)
    model.eval()  # Set the model to evaluation mode
    return model


def load_metadata(metadata_path):
    """Load model metadata from a JSON file."""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata


def get_label_mapping(metadata):
    """Get the label mapping from the metadata."""
    return metadata.get("label_mapping", {})


def get_input_dim(metadata):
    """Get the input dimension from the metadata."""
    return metadata.get("input_dim", 0)