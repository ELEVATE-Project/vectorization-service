from sentence_transformers import SentenceTransformer
from app.config import settings

# Initialize embedding model
embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

def generate_embeddings(texts: list):
    """Generate embeddings for a list of texts"""
    return embedding_model.encode(texts)

def generate_single_embedding(text: str):
    """Generate embedding for a single text"""
    return embedding_model.encode(text)
