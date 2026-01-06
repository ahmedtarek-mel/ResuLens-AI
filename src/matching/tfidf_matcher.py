"""
TF-IDF Matcher Module
Keyword-based matching using TF-IDF vectorization
"""

import numpy as np
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFMatcher:
    """
    TF-IDF based text matching
    Good for keyword overlap and explicit term matching
    """
    
    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_df: float = 0.95
    ):
        """
        Initialize TF-IDF matcher
        
        Args:
            max_features: Maximum number of features (vocabulary size)
            ngram_range: Range of n-grams to consider (1,2) = unigrams and bigrams
            min_df: Minimum document frequency for a term
            max_df: Maximum document frequency for a term (remove very common terms)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            lowercase=True,
            stop_words='english',
            sublinear_tf=True  # Apply sublinear tf scaling (1 + log(tf))
        )
        self._is_fitted = False
    
    def fit(self, texts: List[str]) -> 'TFIDFMatcher':
        """
        Fit the vectorizer on a corpus of texts
        
        Args:
            texts: List of documents to fit on
            
        Returns:
            self for chaining
        """
        self.vectorizer.fit(texts)
        self._is_fitted = True
        return self
    
    def calculate_similarity(
        self, 
        text1: str, 
        text2: str,
        fit_first: bool = True
    ) -> float:
        """
        Calculate TF-IDF cosine similarity between two texts
        
        Args:
            text1: First text (e.g., resume)
            text2: Second text (e.g., job description)
            fit_first: Whether to fit the vectorizer on these texts first
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        try:
            if fit_first or not self._is_fitted:
                # Fit on both texts together
                corpus = [text1, text2]
                tfidf_matrix = self.vectorizer.fit_transform(corpus)
                self._is_fitted = True
            else:
                # Transform using existing vocabulary
                tfidf_matrix = self.vectorizer.transform([text1, text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            
            return float(similarity[0][0])
        except ValueError:
            # Handle edge case where vectorizer prunes all terms
            # (happens with very short texts or when max_df/min_df filters everything)
            return 0.0

    
    def get_important_terms(
        self, 
        text: str, 
        top_n: int = 15
    ) -> List[Tuple[str, float]]:
        """
        Get the most important terms in a text based on TF-IDF scores
        
        Args:
            text: Input text
            top_n: Number of top terms to return
            
        Returns:
            List of (term, score) tuples
        """
        try:
            if not self._is_fitted:
                # Fit on this text
                self.vectorizer.fit([text])
                self._is_fitted = True
            
            # Transform the text
            tfidf_vector = self.vectorizer.transform([text])
            
            # Get feature names
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get non-zero scores
            nonzero_indices = tfidf_vector[0].nonzero()[1]
            scores = [(feature_names[i], tfidf_vector[0, i]) for i in nonzero_indices]
            
            # Sort by score descending
            scores.sort(key=lambda x: x[1], reverse=True)
            
            return scores[:top_n]
        except ValueError:
            # Handle edge case where vectorizer prunes all terms
            return []

    
    def get_matching_terms(
        self, 
        text1: str, 
        text2: str
    ) -> List[Tuple[str, float, float]]:
        """
        Find terms that appear in both texts with their TF-IDF scores
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            List of (term, score_in_text1, score_in_text2) tuples
        """
        # Fit on both texts
        corpus = [text1, text2]
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self._is_fitted = True
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Find terms present in both
        matches = []
        for i, term in enumerate(feature_names):
            score1 = tfidf_matrix[0, i]
            score2 = tfidf_matrix[1, i]
            
            if score1 > 0 and score2 > 0:
                matches.append((term, float(score1), float(score2)))
        
        # Sort by combined score
        matches.sort(key=lambda x: x[1] + x[2], reverse=True)
        
        return matches
    
    def vectorize(self, texts: List[str], fit: bool = True) -> np.ndarray:
        """
        Vectorize texts to TF-IDF representations
        
        Args:
            texts: List of texts to vectorize
            fit: Whether to fit the vectorizer first
            
        Returns:
            TF-IDF matrix (sparse or dense)
        """
        if fit:
            matrix = self.vectorizer.fit_transform(texts)
            self._is_fitted = True
        else:
            matrix = self.vectorizer.transform(texts)
        
        return matrix.toarray()
    
    def get_vocabulary_size(self) -> int:
        """Get the size of the fitted vocabulary"""
        if not self._is_fitted:
            return 0
        return len(self.vectorizer.vocabulary_)
    
    def reset(self) -> None:
        """Reset the vectorizer to unfitted state"""
        self._is_fitted = False
        self.vectorizer = TfidfVectorizer(
            max_features=self.vectorizer.max_features,
            ngram_range=self.vectorizer.ngram_range,
            min_df=self.vectorizer.min_df,
            max_df=self.vectorizer.max_df,
            lowercase=True,
            stop_words='english',
            sublinear_tf=True
        )
