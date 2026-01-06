"""
Text Preprocessing Module
Cleans and normalizes text for NLP processing
"""

import re
import string
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer


class TextPreprocessor:
    """
    Text preprocessing pipeline for resumes and job descriptions
    Handles cleaning, tokenization, and normalization
    """
    
    def __init__(self, remove_stopwords: bool = False, lemmatize: bool = False):
        """
        Initialize the preprocessor
        
        Args:
            remove_stopwords: Whether to remove stopwords (default False for semantic matching)
            lemmatize: Whether to apply lemmatization
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize tools
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        
        # Add domain-specific words to keep (don't remove as stopwords)
        self.keep_words = {
            'python', 'java', 'c', 'r', 'go', 'no', 'not', 'more', 'most',
            'can', 'will', 'should', 'must', 'need', 'required'
        }
    
    def _download_nltk_data(self) -> None:
        """Download required NLTK data packages"""
        required_packages = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'punkt_tab']
        
        for package in required_packages:
            try:
                nltk.data.find(f'tokenizers/{package}' if 'punkt' in package else f'corpora/{package}')
            except LookupError:
                try:
                    nltk.download(package, quiet=True)
                except:
                    pass  # Skip if download fails
    
    def preprocess(self, text: str, for_embedding: bool = True) -> str:
        """
        Full preprocessing pipeline
        
        Args:
            text: Raw input text
            for_embedding: If True, keeps more context for semantic models
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Step 1: Basic cleaning
        text = self._clean_text(text)
        
        # Step 2: Normalize whitespace
        text = self._normalize_whitespace(text)
        
        # For embedding models, we want to keep more context
        if for_embedding:
            return text
        
        # Step 3: Tokenization and further processing (for TF-IDF, etc.)
        tokens = self.tokenize(text)
        
        # Step 4: Remove stopwords if enabled
        if self.remove_stopwords:
            tokens = self._remove_stopwords(tokens)
        
        # Step 5: Lemmatization if enabled
        if self.lemmatize:
            tokens = self._lemmatize(tokens)
        
        return ' '.join(tokens)
    
    def _clean_text(self, text: str) -> str:
        """Clean raw text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        
        # Remove email addresses (but keep a marker)
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', ' ', text)
        
        # Remove phone numbers
        text = re.sub(r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{4,6}', ' ', text)
        
        # Keep alphanumeric, spaces, and some punctuation
        text = re.sub(r'[^a-z0-9\s\.\,\-\+\#]', ' ', text)
        
        # Normalize programming language notations
        text = self._normalize_tech_terms(text)
        
        return text
    
    def _normalize_tech_terms(self, text: str) -> str:
        """Normalize technical terms for consistency"""
        replacements = {
            r'c\s*\+\+': 'cplusplus',
            r'c\s*#': 'csharp',
            r'\.net': 'dotnet',
            r'node\.?js': 'nodejs',
            r'react\.?js': 'reactjs',
            r'vue\.?js': 'vuejs',
            r'next\.?js': 'nextjs',
            r'type\s*script': 'typescript',
            r'java\s*script': 'javascript',
            r'machine\s*learning': 'machinelearning',
            r'deep\s*learning': 'deeplearning',
            r'data\s*science': 'datascience',
            r'ci\s*/\s*cd': 'cicd',
            r'dev\s*ops': 'devops'
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace and line breaks"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback to simple split if NLTK fails
            tokens = text.split()
        
        return tokens
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize text into sentences"""
        try:
            sentences = sent_tokenize(text)
        except:
            # Fallback to simple split
            sentences = re.split(r'[.!?]+', text)
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords while keeping important domain words"""
        return [
            token for token in tokens 
            if token.lower() not in self.stop_words or token.lower() in self.keep_words
        ]
    
    def _lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens"""
        if self.lemmatizer is None:
            return tokens
        
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def extract_keywords(self, text: str, top_n: int = 20) -> List[str]:
        """Extract top keywords from text using TF-IDF-like scoring"""
        # Preprocess with stopword removal
        tokens = self.tokenize(self._clean_text(text))
        
        # Remove stopwords and short tokens
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        
        # Count frequencies
        freq = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1
        
        # Sort by frequency
        sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        return [kw for kw, _ in sorted_keywords[:top_n]]
    
    def get_ngrams(self, text: str, n: int = 2) -> List[str]:
        """Extract n-grams from text"""
        tokens = self.tokenize(self._clean_text(text))
        
        if len(tokens) < n:
            return []
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams.append(ngram)
        
        return ngrams
