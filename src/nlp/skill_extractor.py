"""
Skill Extractor Module
NER-based skill extraction from resumes and job descriptions
"""

import re
import json
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ExtractedSkills:
    """Container for extracted skills by category"""
    technical: List[str]
    soft: List[str]
    tools: List[str]
    certifications: List[str]
    all_skills: List[str]
    
    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "technical": self.technical,
            "soft": self.soft,
            "tools": self.tools,
            "certifications": self.certifications,
            "all_skills": self.all_skills
        }


class SkillExtractor:
    """
    Skill extraction using pattern matching and a comprehensive skills database
    Categorizes skills into technical, soft skills, tools, and certifications
    """
    
    # Comprehensive skills database
    TECHNICAL_SKILLS = {
        # Programming Languages
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'c', 'ruby', 'go', 'golang',
        'rust', 'swift', 'kotlin', 'scala', 'php', 'perl', 'r', 'matlab', 'julia', 'haskell',
        'objective-c', 'dart', 'lua', 'groovy', 'shell', 'bash', 'powershell', 'sql', 'plsql',
        
        # Web Technologies
        'html', 'html5', 'css', 'css3', 'sass', 'scss', 'less', 'bootstrap', 'tailwind',
        'javascript', 'jquery', 'ajax', 'json', 'xml', 'rest', 'restful', 'graphql', 'websocket',
        
        # Frontend Frameworks
        'react', 'reactjs', 'react.js', 'angular', 'angularjs', 'vue', 'vuejs', 'vue.js',
        'next.js', 'nextjs', 'nuxt', 'nuxt.js', 'svelte', 'gatsby', 'ember', 'backbone',
        
        # Backend Frameworks
        'django', 'flask', 'fastapi', 'express', 'express.js', 'nodejs', 'node.js',
        'spring', 'spring boot', 'springboot', '.net', 'dotnet', 'asp.net', 'rails',
        'ruby on rails', 'laravel', 'symfony', 'gin', 'echo', 'fiber',
        
        # Databases
        'mysql', 'postgresql', 'postgres', 'sqlite', 'oracle', 'sql server', 'mssql',
        'mongodb', 'redis', 'cassandra', 'dynamodb', 'couchdb', 'neo4j', 'elasticsearch',
        'mariadb', 'firebase', 'supabase', 'influxdb', 'timescaledb',
        
        # Cloud & DevOps
        'aws', 'amazon web services', 'azure', 'gcp', 'google cloud', 'heroku', 'digitalocean',
        'docker', 'kubernetes', 'k8s', 'jenkins', 'gitlab ci', 'github actions', 'circleci',
        'terraform', 'ansible', 'puppet', 'chef', 'vagrant', 'nginx', 'apache',
        'ci/cd', 'cicd', 'devops', 'devsecops', 'microservices', 'serverless',
        
        # Data Science & ML
        'machine learning', 'deep learning', 'artificial intelligence', 'ai', 'ml',
        'neural networks', 'nlp', 'natural language processing', 'computer vision',
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn', 'pandas', 'numpy',
        'scipy', 'matplotlib', 'seaborn', 'plotly', 'opencv', 'spacy', 'nltk', 'huggingface',
        'transformers', 'bert', 'gpt', 'llm', 'langchain', 'rag',
        'data science', 'data analysis', 'data engineering', 'data mining',
        'big data', 'spark', 'pyspark', 'hadoop', 'hive', 'kafka', 'airflow',
        'etl', 'data pipeline', 'data warehouse', 'data lake',
        
        # Mobile Development
        'android', 'ios', 'react native', 'flutter', 'xamarin', 'ionic', 'cordova',
        'swift', 'kotlin', 'swiftui', 'jetpack compose',
        
        # Version Control
        'git', 'github', 'gitlab', 'bitbucket', 'svn', 'mercurial',
        
        # Testing
        'unit testing', 'integration testing', 'e2e testing', 'test automation',
        'selenium', 'cypress', 'jest', 'mocha', 'pytest', 'junit', 'testng',
        'playwright', 'puppeteer',
        
        # Architecture & Design
        'microservices', 'monolith', 'soa', 'api design', 'system design',
        'design patterns', 'solid', 'oop', 'functional programming',
        
        # Security
        'cybersecurity', 'security', 'oauth', 'jwt', 'ssl', 'tls', 'encryption',
        'penetration testing', 'owasp', 'siem', 'soc',
        
        # Other
        'agile', 'scrum', 'kanban', 'jira', 'confluence', 'trello',
        'linux', 'unix', 'windows server', 'networking', 'tcp/ip',
        'blockchain', 'web3', 'solidity', 'smart contracts'
    }
    
    SOFT_SKILLS = {
        'leadership', 'communication', 'teamwork', 'collaboration', 'problem solving',
        'problem-solving', 'critical thinking', 'analytical', 'creativity', 'innovation',
        'time management', 'project management', 'organization', 'multitasking',
        'adaptability', 'flexibility', 'attention to detail', 'self-motivated',
        'interpersonal', 'presentation', 'public speaking', 'negotiation',
        'conflict resolution', 'mentoring', 'coaching', 'strategic thinking',
        'decision making', 'decision-making', 'emotional intelligence', 'empathy',
        'customer service', 'client management', 'stakeholder management',
        'cross-functional', 'cross functional', 'team building', 'team-building'
    }
    
    TOOLS = {
        # IDEs & Editors
        'vscode', 'visual studio', 'intellij', 'pycharm', 'eclipse', 'sublime',
        'atom', 'vim', 'neovim', 'emacs', 'xcode', 'android studio',
        
        # Design Tools
        'figma', 'sketch', 'adobe xd', 'photoshop', 'illustrator', 'invision',
        'zeplin', 'canva', 'framer',
        
        # Project Management
        'jira', 'confluence', 'trello', 'asana', 'monday', 'notion', 'linear',
        'clickup', 'basecamp', 'azure devops',
        
        # Communication
        'slack', 'microsoft teams', 'zoom', 'discord', 'mattermost',
        
        # API & Testing Tools
        'postman', 'insomnia', 'swagger', 'soapui', 'charles proxy',
        
        # Monitoring & Logging
        'grafana', 'prometheus', 'datadog', 'splunk', 'new relic', 'dynatrace',
        'elk stack', 'kibana', 'logstash', 'cloudwatch',
        
        # Documentation
        'markdown', 'sphinx', 'doxygen', 'swagger', 'openapi', 'readme',
        
        # Other Tools
        'excel', 'powerpoint', 'tableau', 'power bi', 'looker',
        'salesforce', 'hubspot', 'zendesk', 'intercom'
    }
    
    CERTIFICATIONS = {
        # AWS
        'aws certified', 'aws solutions architect', 'aws developer', 'aws sysops',
        'aws devops', 'aws machine learning', 'aws data analytics',
        
        # Azure
        'azure certified', 'az-900', 'az-104', 'az-204', 'az-400', 'az-305',
        
        # Google Cloud
        'gcp certified', 'google cloud certified', 'professional cloud architect',
        
        # Other Cloud/Tech
        'kubernetes certified', 'cka', 'ckad', 'docker certified',
        'terraform certified', 'hashicorp certified',
        
        # Security
        'cissp', 'cism', 'ceh', 'comptia security+', 'comptia a+', 'comptia network+',
        'oscp', 'ccna', 'ccnp',
        
        # Project Management
        'pmp', 'prince2', 'scrum master', 'csm', 'psm', 'safe', 'itil',
        
        # Data
        'data science certification', 'machine learning certification',
        'tableau certified', 'power bi certified', 'snowflake certified'
    }
    
    def __init__(self, custom_skills_path: Optional[str] = None):
        """
        Initialize the skill extractor
        
        Args:
            custom_skills_path: Path to custom skills JSON file
        """
        self.technical_skills = self.TECHNICAL_SKILLS.copy()
        self.soft_skills = self.SOFT_SKILLS.copy()
        self.tools = self.TOOLS.copy()
        self.certifications = self.CERTIFICATIONS.copy()
        
        # Load custom skills if provided
        if custom_skills_path:
            self._load_custom_skills(custom_skills_path)
    
    def _load_custom_skills(self, path: str) -> None:
        """Load custom skills from JSON file"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                custom = json.load(f)
                
            if 'technical' in custom:
                self.technical_skills.update(set(s.lower() for s in custom['technical']))
            if 'soft' in custom:
                self.soft_skills.update(set(s.lower() for s in custom['soft']))
            if 'tools' in custom:
                self.tools.update(set(s.lower() for s in custom['tools']))
            if 'certifications' in custom:
                self.certifications.update(set(s.lower() for s in custom['certifications']))
        except Exception as e:
            print(f"Warning: Could not load custom skills: {e}")
    
    def extract(self, text: str) -> ExtractedSkills:
        """
        Extract and categorize skills from text
        
        Args:
            text: Input text (resume or job description)
            
        Returns:
            ExtractedSkills object with categorized skills
        """
        text_lower = text.lower()
        
        technical = self._find_skills(text_lower, self.technical_skills)
        soft = self._find_skills(text_lower, self.soft_skills)
        tools = self._find_skills(text_lower, self.tools)
        certs = self._find_skills(text_lower, self.certifications)
        
        # Combine all unique skills
        all_skills = list(set(technical + soft + tools + certs))
        
        return ExtractedSkills(
            technical=technical,
            soft=soft,
            tools=tools,
            certifications=certs,
            all_skills=all_skills
        )
    
    def _find_skills(self, text: str, skill_set: Set[str]) -> List[str]:
        """Find skills from a skill set in text"""
        found = []
        
        for skill in skill_set:
            # Create a pattern that matches the skill as a whole word
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                found.append(skill.title() if len(skill) > 3 else skill.upper())
        
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for skill in found:
            skill_lower = skill.lower()
            if skill_lower not in seen:
                seen.add(skill_lower)
                unique.append(skill)
        
        return unique
    
    def calculate_skill_overlap(
        self, 
        resume_skills: ExtractedSkills, 
        job_skills: ExtractedSkills
    ) -> Dict[str, any]:
        """
        Calculate skill overlap between resume and job
        
        Returns:
            Dictionary with overlap metrics and details
        """
        resume_set = set(s.lower() for s in resume_skills.all_skills)
        job_set = set(s.lower() for s in job_skills.all_skills)
        
        matched = resume_set.intersection(job_set)
        missing = job_set - resume_set
        extra = resume_set - job_set
        
        # Calculate overlap percentage (relative to job requirements)
        overlap_pct = (len(matched) / len(job_set) * 100) if job_set else 0
        
        return {
            'matched_skills': [s.title() for s in matched],
            'missing_skills': [s.title() for s in missing],
            'extra_skills': [s.title() for s in extra],
            'overlap_percentage': round(overlap_pct, 2),
            'matched_count': len(matched),
            'required_count': len(job_set),
            'resume_skills_count': len(resume_set)
        }
    
    def get_skill_categories(self, skills: List[str]) -> Dict[str, List[str]]:
        """Categorize a list of skills"""
        categories = {
            'technical': [],
            'soft': [],
            'tools': [],
            'certifications': [],
            'other': []
        }
        
        for skill in skills:
            skill_lower = skill.lower()
            
            if skill_lower in self.technical_skills:
                categories['technical'].append(skill)
            elif skill_lower in self.soft_skills:
                categories['soft'].append(skill)
            elif skill_lower in self.tools:
                categories['tools'].append(skill)
            elif skill_lower in self.certifications:
                categories['certifications'].append(skill)
            else:
                categories['other'].append(skill)
        
        return categories
