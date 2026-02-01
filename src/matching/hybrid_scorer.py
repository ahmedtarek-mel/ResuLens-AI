"""
Hybrid Scorer Module
Combines multiple matching strategies for comprehensive scoring
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np

from ..utils.config import Config
from ..nlp.skill_extractor import SkillExtractor, ExtractedSkills
from ..nlp.preprocessor import TextPreprocessor
from .tfidf_matcher import TFIDFMatcher
from .semantic_matcher import SemanticMatcher


@dataclass
class MatchResult:
    """Container for complete match analysis results"""
    overall_score: float
    grade: str
    
    # Component scores
    semantic_score: float
    skill_score: float
    keyword_score: float
    
    # Skill analysis
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    extra_skills: List[str] = field(default_factory=list)
    
    # Detailed metrics
    skill_overlap_percentage: float = 0.0
    resume_skill_count: int = 0
    job_skill_count: int = 0
    
    # Top matching keywords
    top_keywords: List[tuple] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "grade": self.grade,
            "semantic_score": self.semantic_score,
            "skill_score": self.skill_score,
            "keyword_score": self.keyword_score,
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
            "extra_skills": self.extra_skills,
            "skill_overlap_percentage": self.skill_overlap_percentage,
            "resume_skill_count": self.resume_skill_count,
            "job_skill_count": self.job_skill_count,
            "top_keywords": self.top_keywords,
            "recommendations": self.recommendations
        }


class HybridScorer:
    """
    Hybrid scoring system combining:
    1. Semantic similarity (BERT embeddings)
    2. Skill matching (NER-based)
    3. Keyword matching (TF-IDF)
    
    Final score is a weighted combination with configurable weights
    """
    
    GRADE_THRESHOLDS = {
        'A+': 90,
        'A': 85,
        'B+': 80,
        'B': 75,
        'C+': 70,
        'C': 65,
        'D': 55,
        'F': 0
    }
    
    def __init__(
        self,
        semantic_weight: float = None,
        skill_weight: float = None,
        keyword_weight: float = None,
        model_name: str = None
    ):
        """
        Initialize hybrid scorer
        
        Args:
            semantic_weight: Weight for semantic similarity (0-1)
            skill_weight: Weight for skill matching (0-1)
            keyword_weight: Weight for keyword matching (0-1)
            model_name: Embedding model to use
        """
        # Load config
        config = Config()
        
        # Set weights from config or parameters
        weights = config.matching_weights
        self.semantic_weight = semantic_weight or weights.get('semantic_weight', 0.40)
        self.skill_weight = skill_weight or weights.get('skill_weight', 0.35)
        self.keyword_weight = keyword_weight or weights.get('keyword_weight', 0.25)
        
        # Normalize weights to sum to 1
        total = self.semantic_weight + self.skill_weight + self.keyword_weight
        if total != 1.0:
            self.semantic_weight /= total
            self.skill_weight /= total
            self.keyword_weight /= total
        
        # Check for lightweight mode (for low-memory deployments like Render free tier)
        import os
        self.lightweight_mode = os.environ.get('LIGHTWEIGHT_MODE', 'false').lower() == 'true'
        
        # Initialize matchers
        model = model_name or config.embedding_model
        
        if self.lightweight_mode:
            # Skip BERT to save memory - use TF-IDF for semantic similarity
            print("⚡ LIGHTWEIGHT MODE: Skipping BERT model to save memory")
            self.semantic_matcher = None
        else:
            self.semantic_matcher = SemanticMatcher(model_name=model)
            
        self.tfidf_matcher = TFIDFMatcher()
        self.skill_extractor = SkillExtractor()
        self.preprocessor = TextPreprocessor()

    
    def calculate_match(
        self, 
        resume_text: str, 
        job_text: str,
        resume_skills: Optional[ExtractedSkills] = None,
        job_skills: Optional[ExtractedSkills] = None
    ) -> MatchResult:
        """
        Calculate comprehensive match score
        
        Args:
            resume_text: Full resume text
            job_text: Full job description text
            resume_skills: Pre-extracted resume skills (optional)
            job_skills: Pre-extracted job skills (optional)
            
        Returns:
            MatchResult with complete analysis
        """
        # Preprocess texts
        resume_clean = self.preprocessor.preprocess(resume_text, for_embedding=True)
        job_clean = self.preprocessor.preprocess(job_text, for_embedding=True)
        
        # 1. Calculate semantic similarity
        if self.lightweight_mode or self.semantic_matcher is None:
            # Use TF-IDF as fallback for semantic similarity in lightweight mode
            semantic_score = self.tfidf_matcher.calculate_similarity(
                resume_clean, job_clean
            )
        else:
            semantic_score = self.semantic_matcher.calculate_similarity(
                resume_clean, job_clean, preprocess=False
            )

        
        # 2. Calculate skill match
        if resume_skills is None:
            resume_skills = self.skill_extractor.extract(resume_text)
        if job_skills is None:
            job_skills = self.skill_extractor.extract(job_text)
        
        skill_analysis = self.skill_extractor.calculate_skill_overlap(
            resume_skills, job_skills
        )
        skill_score = skill_analysis['overlap_percentage'] / 100
        
        # 3. Calculate TF-IDF keyword similarity
        keyword_score = self.tfidf_matcher.calculate_similarity(
            resume_clean, job_clean
        )
        
        # 4. Calculate weighted overall score
        overall_score = (
            self.semantic_weight * semantic_score +
            self.skill_weight * skill_score +
            self.keyword_weight * keyword_score
        ) * 100  # Convert to 0-100 scale
        
        # Get grade
        grade = self._calculate_grade(overall_score)
        
        # Get top matching keywords
        matching_terms = self.tfidf_matcher.get_matching_terms(resume_clean, job_clean)
        top_keywords = [(term, round(s1, 3)) for term, s1, s2 in matching_terms[:10]]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_score,
            skill_analysis['missing_skills'],
            semantic_score,
            keyword_score
        )
        
        return MatchResult(
            overall_score=round(overall_score, 1),
            grade=grade,
            semantic_score=round(semantic_score * 100, 1),
            skill_score=round(skill_score * 100, 1),
            keyword_score=round(keyword_score * 100, 1),
            matched_skills=skill_analysis['matched_skills'],
            missing_skills=skill_analysis['missing_skills'],
            extra_skills=skill_analysis['extra_skills'],
            skill_overlap_percentage=skill_analysis['overlap_percentage'],
            resume_skill_count=skill_analysis['resume_skills_count'],
            job_skill_count=skill_analysis['required_count'],
            top_keywords=top_keywords,
            recommendations=recommendations
        )
    
    def _calculate_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        for grade, threshold in self.GRADE_THRESHOLDS.items():
            if score >= threshold:
                return grade
        return 'F'
    
    def _generate_recommendations(
        self,
        overall_score: float,
        missing_skills: List[str],
        semantic_score: float,
        keyword_score: float
    ) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Skill-based recommendations
        if missing_skills:
            top_missing = missing_skills[:5]
            if len(top_missing) == 1:
                recommendations.append(
                    f"💡 Consider adding '{top_missing[0]}' to strengthen your application."
                )
            else:
                skills_str = ", ".join(top_missing[:3])
                if len(top_missing) > 3:
                    skills_str += f", and {len(top_missing)-3} more"
                recommendations.append(
                    f"💡 Key skills to develop: {skills_str}"
                )
        
        # Semantic score recommendations
        if semantic_score < 0.5:
            recommendations.append(
                "📝 Consider tailoring your resume language to better match "
                "the job description's terminology and focus areas."
            )
        
        # Keyword recommendations
        if keyword_score < 0.3:
            recommendations.append(
                "🔑 Include more relevant keywords from the job description "
                "in your resume to improve ATS compatibility."
            )
        
        # Overall score recommendations
        if overall_score >= 80:
            recommendations.append(
                "✅ Strong match! Your profile aligns well with this position."
            )
        elif overall_score >= 60:
            recommendations.append(
                "📈 Good foundation. Address the skill gaps above to strengthen your application."
            )
        else:
            recommendations.append(
                "⚠️ Consider gaining more relevant experience or skills "
                "before applying to similar positions."
            )
        
        return recommendations
    
    def get_weights(self) -> Dict[str, float]:
        """Get current scoring weights"""
        return {
            'semantic_weight': self.semantic_weight,
            'skill_weight': self.skill_weight,
            'keyword_weight': self.keyword_weight
        }
    
    def update_weights(
        self,
        semantic_weight: float = None,
        skill_weight: float = None,
        keyword_weight: float = None
    ) -> None:
        """Update scoring weights"""
        if semantic_weight is not None:
            self.semantic_weight = semantic_weight
        if skill_weight is not None:
            self.skill_weight = skill_weight
        if keyword_weight is not None:
            self.keyword_weight = keyword_weight
        
        # Normalize
        total = self.semantic_weight + self.skill_weight + self.keyword_weight
        self.semantic_weight /= total
        self.skill_weight /= total
        self.keyword_weight /= total
