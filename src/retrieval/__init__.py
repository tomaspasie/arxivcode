"""
Retrieval module for ArXivCode.
Provides dense retrieval and re-ranking capabilities.
"""

from .dense_retrieval import DenseRetrieval
from .cross_encoder_reranker import CrossEncoderReranker, RerankingPipeline

# FAISS is optional - only needed if using FAISSIndexManager
try:
    from .faiss_index import FAISSIndexManager
    __all__ = ['FAISSIndexManager', 'DenseRetrieval', 'CrossEncoderReranker', 'RerankingPipeline']
except ImportError:
    # FAISS not installed - that's fine, we use NumPy arrays instead
    __all__ = ['DenseRetrieval', 'CrossEncoderReranker', 'RerankingPipeline']