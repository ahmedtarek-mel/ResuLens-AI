"""
Tests for document parsers
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.resume_parser import ResumeParser
from src.parsers.job_parser import JobDescriptionParser


class TestResumeParser:
    """Tests for resume parser"""
    
    @pytest.fixture
    def sample_resume_text(self):
        return """
John Doe
john.doe@email.com | 555-123-4567
linkedin.com/in/johndoe | github.com/johndoe

SUMMARY
Experienced software engineer with expertise in Python and machine learning.

EXPERIENCE
Senior Software Engineer | TechCorp | 2020 - Present
- Led development of ML systems
- Managed team of 5 engineers

Software Engineer | StartupXYZ | 2018 - 2020
- Built REST APIs using Python and Flask
- Implemented CI/CD pipelines

EDUCATION
M.S. Computer Science | Stanford University | 2018

SKILLS
Python, Machine Learning, TensorFlow, AWS, Docker, Kubernetes
"""
    
    def test_extract_contact_info(self, sample_resume_text):
        """Should extract contact information"""
        parser = ResumeParser()
        resume = parser._parse_text(sample_resume_text)
        
        assert 'email' in resume.contact_info
        assert 'john.doe@email.com' in resume.contact_info['email']
    
    def test_extract_skills(self, sample_resume_text):
        """Should extract skills from skills section"""
        parser = ResumeParser()
        resume = parser._parse_text(sample_resume_text)
        
        assert len(resume.skills) > 0
    
    def test_split_sections(self, sample_resume_text):
        """Should split into sections"""
        parser = ResumeParser()
        sections = parser._split_into_sections(sample_resume_text)
        
        assert 'summary' in sections or 'header' in sections
        assert 'experience' in sections
        assert 'education' in sections


class TestJobParser:
    """Tests for job description parser"""
    
    @pytest.fixture
    def sample_job_text(self):
        return """
Senior Python Developer
TechCorp Inc. | San Francisco, CA | Remote

About Us:
We are a fast-growing technology company looking for talented engineers.

Requirements:
- 5+ years of experience in Python
- Experience with Machine Learning and TensorFlow
- Strong communication skills
- Bachelor's degree in Computer Science

Preferred:
- Experience with AWS
- Docker and Kubernetes knowledge

Responsibilities:
- Design and implement scalable systems
- Mentor junior developers
- Collaborate with cross-functional teams

Benefits:
- Competitive salary $150k - $200k
- Remote work flexibility
- Health insurance
"""
    
    def test_extract_title(self, sample_job_text):
        """Should extract job title"""
        parser = JobDescriptionParser()
        job = parser.parse(sample_job_text)
        
        assert 'python' in job.title.lower() or 'developer' in job.title.lower()
    
    def test_extract_skills(self, sample_job_text):
        """Should extract required skills"""
        parser = JobDescriptionParser()
        job = parser.parse(sample_job_text)
        
        all_skills = job.get_all_skills()
        skill_names = [s.lower() for s in all_skills]
        
        assert 'python' in skill_names or any('python' in s for s in skill_names)
    
    def test_extract_experience(self, sample_job_text):
        """Should extract experience requirement"""
        parser = JobDescriptionParser()
        job = parser.parse(sample_job_text)
        
        # Should find "5+ years" or similar
        assert job.required_experience or any('5' in str(x) for x in [job.required_experience])
    
    def test_extract_responsibilities(self, sample_job_text):
        """Should extract responsibilities"""
        parser = JobDescriptionParser()
        job = parser.parse(sample_job_text)
        
        assert len(job.responsibilities) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
