"""
Embedding Model Module
Sentence-Transformers wrapper for semantic text embeddings
"""

import numpy as np
from typing import List, Union, Optional
from functools import lru_cache


class EmbeddingModel:
    """
    Wrapper for Sentence-Transformers embedding models
    Provides semantic embeddings for text matching
    """
    
    # Default model - good balance of speed and accuracy
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    
    # Alternative models
    AVAILABLE_MODELS = {
        "all-MiniLM-L6-v2": {
            "size": "80MB",
            "dimensions": 384,
            "speed": "fast",
            "description": "Best balance of speed and accuracy"
        },
        "all-mpnet-base-v2": {
            "size": "420MB",
            "dimensions": 768,
            "speed": "medium",
            "description": "Higher accuracy, slower inference"
        },
        "paraphrase-MiniLM-L6-v2": {
            "size": "80MB",
            "dimensions": 384,
            "speed": "fast",
            "description": "Good for paraphrase detection"
        }
    }
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize the embedding model
        
        Args:
            model_name: Name of the sentence-transformer model
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self._model = None
    
    @property
    def model(self):
        """Lazy load the model"""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self) -> None:
        """Load the sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Try auto-detection first (Best Performance), Fallback to CPU (Best Reliability)
            try:
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as e:
                print(f"⚠️ GPU Load Failed. Falling back to CPU. Error: {e}")
                self._model = SentenceTransformer(self.model_name, device='cpu')
                
            print(f"Loaded embedding model: {self.model_name}")
            
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install it with: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        normalize: bool = True,
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode text(s) into embeddings
        
        Args:
            texts: Single text or list of texts to encode
            normalize: Whether to L2-normalize embeddings (for cosine similarity)
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings, shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=show_progress
        )
        
        return embeddings
    
    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a single text - more efficient for single inputs
        
        Args:
            text: Text to encode
            normalize: Whether to normalize
            
        Returns:
            1D numpy array of embedding
        """
        return self.encode([text], normalize=normalize)[0]
    
    def cosine_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0 to 1 if normalized)
        """
        # If embeddings are already normalized, dot product equals cosine similarity
        similarity = np.dot(embedding1, embedding2)
        
        # Ensure it's within valid range due to floating point
        return float(np.clip(similarity, -1.0, 1.0))
    
    def similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate pairwise similarity matrix
        
        Args:
            embeddings: Matrix of embeddings (n_samples, embedding_dim)
            
        Returns:
            Similarity matrix (n_samples, n_samples)
        """
        # For normalized embeddings, similarity = dot product
        return np.dot(embeddings, embeddings.T)
    
    def find_most_similar(
        self, 
        query_embedding: np.ndarray, 
        candidate_embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar embeddings to a query
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: Matrix of candidate embeddings
            top_k: Number of top matches to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        # Calculate similarities
        similarities = np.dot(candidate_embeddings, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
    
    def get_model_info(self) -> dict:
        """Get information about the current model"""
        if self.model_name in self.AVAILABLE_MODELS:
            info = self.AVAILABLE_MODELS[self.model_name].copy()
        else:
            info = {"description": "Custom model"}
        
        info["model_name"] = self.model_name
        
        if self._model is not None:
            info["loaded"] = True
            info["device"] = str(self._model.device)
        else:
            info["loaded"] = False
        
        return info
    
    @staticmethod
    def list_available_models() -> dict:
        """List all available pre-configured models"""
        return EmbeddingModel.AVAILABLE_MODELS.copy()
