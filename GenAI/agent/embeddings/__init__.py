# This package handles embeddings for the GenAI agent.
## Import necessary modules or classes here.

__version__ = "1.0.0"
__package_name__ = "embeddings"


__all__ = [
    "EmbeddingGenerator",
    "normalize_embedding",
    "calculate_similarity",
]


def initialize_package():
    print(f"Initializing Package: {__package_name__} v{__version__}")
    

## Automatic initialization
initialize_package()