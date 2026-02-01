"""
Semantic Matcher Module
BERT-based semantic similarity matching
"""

import numpy as np
from typing import List, Dict, Optional, Tuple

from ..nlp.embeddings import EmbeddingModel
from ..nlp.preprocessor import TextPreprocessor


class SemanticMatcher:
    """
    Semantic similarity matching using sentence embeddings
    Captures meaning and context beyond keyword matching
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        device: str = None
    ):
        """
        Initialize semantic matcher
        
        Args:
            model_name: Sentence-transformer model to use
            device: Device for model inference ('cuda', 'cpu', or None for auto)
        """
        self.embedding_model = EmbeddingModel(model_name=model_name, device=device)
        self.preprocessor = TextPreprocessor(remove_stopwords=False, lemmatize=False)
    
    def calculate_similarity(
        self, 
        text1: str, 
        text2: str,
        preprocess: bool = True
    ) -> float:
        """
        Calculate semantic similarity between two texts
        
        Args:
            text1: First text (e.g., resume)
            text2: Second text (e.g., job description)
            preprocess: Whether to preprocess texts first
            
        Returns:
            Semantic similarity score (0 to 1)
        """
        if preprocess:
            text1 = self.preprocessor.preprocess(text1, for_embedding=True)
            text2 = self.preprocessor.preprocess(text2, for_embedding=True)
        
        # Encode both texts
        embedding1 = self.embedding_model.encode_single(text1)
        embedding2 = self.embedding_model.encode_single(text2)
        
        # Calculate cosine similarity
        similarity = self.embedding_model.cosine_similarity(embedding1, embedding2)
        
        # Normalize to 0-1 range (cosine can be negative for very different texts)
        normalized = (similarity + 1) / 2
        
        return float(normalized)
    
    def calculate_section_similarity(
        self, 
        resume_sections: Dict[str, str], 
        job_sections: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calculate similarity between corresponding sections
        
        Args:
            resume_sections: Dict of section_name -> section_text for resume
            job_sections: Dict of section_name -> section_text for job
            
        Returns:
            Dict of section_name -> similarity_score
        """
        results = {}
        
        # Map common sections
        section_mappings = {
            'experience': ['experience', 'responsibilities', 'work_history'],
            'skills': ['skills', 'requirements', 'qualifications'],
            'education': ['education', 'education_requirements'],
            'summary': ['summary', 'about', 'description']
        }
        
        for key, variations in section_mappings.items():
            resume_text = ""
            job_text = ""
            
            # Find resume section
            for var in variations:
                if var in resume_sections and resume_sections[var]:
                    resume_text = str(resume_sections[var])
                    break
            
            # Find job section
            for var in variations:
                if var in job_sections and job_sections[var]:
                    job_text = str(job_sections[var])
                    break
            
            # Calculate similarity if both sections exist
            if resume_text and job_text:
                results[key] = self.calculate_similarity(resume_text, job_text)
            else:
                results[key] = 0.0
        
        return results
    
    def calculate_sentence_level_similarity(
        self, 
        text1: str, 
        text2: str,
        top_n: int = 5
    ) -> Dict:
        """
        Calculate sentence-level similarities and find best matches
        
        Args:
            text1: First text
            text2: Second text
            top_n: Number of top sentence pairs to return
            
        Returns:
            Dict with average similarity and top matching sentence pairs
        """
        # Split into sentences
        sentences1 = self.preprocessor.tokenize_sentences(text1)
        sentences2 = self.preprocessor.tokenize_sentences(text2)
        
        if not sentences1 or not sentences2:
            return {
                'average_similarity': 0.0,
                'top_pairs': [],
                'sentence_similarities': []
            }
        
        # Encode all sentences
        embeddings1 = self.embedding_model.encode(sentences1)
        embeddings2 = self.embedding_model.encode(sentences2)
        
        # Calculate similarity matrix
        similarity_matrix = np.dot(embeddings1, embeddings2.T)
        
        # Normalize to 0-1
        similarity_matrix = (similarity_matrix + 1) / 2
        
        # Find top matching pairs
        pairs = []
        for i in range(len(sentences1)):
            for j in range(len(sentences2)):
                pairs.append({
                    'sentence1': sentences1[i][:100],  # Truncate for display
                    'sentence2': sentences2[j][:100],
                    'similarity': float(similarity_matrix[i, j])
                })
        
        # Sort by similarity
        pairs.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Calculate average similarity (best match for each sentence in text1)
        best_matches = np.max(similarity_matrix, axis=1)
        avg_similarity = float(np.mean(best_matches))
        
        return {
            'average_similarity': avg_similarity,
            'top_pairs': pairs[:top_n],
            'sentence_count_1': len(sentences1),
            'sentence_count_2': len(sentences2)
        }
    
    def encode_document(self, text: str) -> np.ndarray:
        """
        Encode a document into a single embedding vector
        
        Args:
            text: Document text
            
        Returns:
            Embedding vector
        """
        processed = self.preprocessor.preprocess(text, for_embedding=True)
        return self.embedding_model.encode_single(processed)
    
    def batch_similarity(
        self, 
        query_text: str, 
        candidate_texts: List[str]
    ) -> List[Tuple[int, float]]:
        """
        Calculate similarity of query against multiple candidates
        
        Args:
            query_text: Query text (e.g., job description)
            candidate_texts: List of candidate texts (e.g., multiple resumes)
            
        Returns:
            List of (index, similarity) tuples, sorted by similarity
        """
        # Preprocess all texts
        query = self.preprocessor.preprocess(query_text, for_embedding=True)
        candidates = [
            self.preprocessor.preprocess(t, for_embedding=True) 
            for t in candidate_texts
        ]
        
        # Encode query
        query_embedding = self.embedding_model.encode_single(query)
        
        # Encode all candidates
        candidate_embeddings = self.embedding_model.encode(candidates)
        
        # Calculate similarities
        similarities = np.dot(candidate_embeddings, query_embedding)
        
        # Normalize
        similarities = (similarities + 1) / 2
        
        # Create and sort results
        results = [(i, float(sim)) for i, sim in enumerate(similarities)]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def get_model_info(self) -> dict:
        """Get information about the embedding model"""
        return self.embedding_model.get_model_info()
