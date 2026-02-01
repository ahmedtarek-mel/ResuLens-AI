"""
Job Description Parser Module
Extracts structured information from job postings
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ParsedJobDescription:
    """Structured job description data"""
    raw_text: str
    title: str = ""
    company: str = ""
    location: str = ""
    job_type: str = ""  # Full-time, Part-time, Contract, etc.
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    required_experience: str = ""
    education_requirements: List[str] = field(default_factory=list)
    responsibilities: List[str] = field(default_factory=list)
    benefits: List[str] = field(default_factory=list)
    salary_range: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "job_type": self.job_type,
            "required_skills": self.required_skills,
            "preferred_skills": self.preferred_skills,
            "required_experience": self.required_experience,
            "education_requirements": self.education_requirements,
            "responsibilities": self.responsibilities,
            "benefits": self.benefits,
            "salary_range": self.salary_range
        }
    
    def get_all_skills(self) -> List[str]:
        """Return combined required and preferred skills"""
        return list(set(self.required_skills + self.preferred_skills))


class JobDescriptionParser:
    """
    Parser for job descriptions/postings
    Extracts requirements, skills, and other structured data
    """
    
    # Section patterns for job descriptions (case insensitive matching applied separately)
    SECTION_PATTERNS = {
        'requirements': r'(requirements?|qualifications?|what\s*we\'?re?\s*looking\s*for|must\s*have|required)',
        'preferred': r'(preferred|nice\s*to\s*have|bonus|plus|desired)',
        'responsibilities': r'(responsibilities?|duties|what\s*you\'?ll?\s*do|role|job\s*description)',
        'benefits': r'(benefits?|perks?|what\s*we\s*offer|compensation)',
        'about': r'(about\s*(?:us|the\s*company|the\s*role)|company\s*overview|who\s*we\s*are)'
    }
    
    # Experience level patterns
    EXPERIENCE_PATTERNS = [
        r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)',
        r'(?:experience|exp)\s*(?:of)?\s*(\d+)\+?\s*(?:years?|yrs?)',
        r'(entry[\s-]?level|junior|mid[\s-]?level|senior|lead|principal|staff)'
    ]
    
    # Job type patterns
    JOB_TYPE_PATTERNS = [
        r'(?i)(full[\s-]?time)',
        r'(?i)(part[\s-]?time)',
        r'(?i)(contract|contractor)',
        r'(?i)(freelance)',
        r'(?i)(remote)',
        r'(?i)(hybrid)',
        r'(?i)(on[\s-]?site|onsite)'
    ]
    
    # Salary patterns
    SALARY_PATTERNS = [
        r'\$[\d,]+(?:\s*[-–]\s*\$?[\d,]+)?(?:\s*(?:per|/)\s*(?:year|yr|hour|hr|annum))?',
        r'[\d,]+k?\s*[-–]\s*[\d,]+k?\s*(?:USD|EUR|GBP)?'
    ]
    
    def parse(self, job_text: str) -> ParsedJobDescription:
        """
        Parse a job description and extract structured information
        
        Args:
            job_text: Raw job description text
            
        Returns:
            ParsedJobDescription object with extracted data
        """
        job = ParsedJobDescription(raw_text=job_text)
        
        # Extract job metadata
        job.title = self._extract_job_title(job_text)
        job.company = self._extract_company(job_text)
        job.location = self._extract_location(job_text)
        job.job_type = self._extract_job_type(job_text)
        job.required_experience = self._extract_experience(job_text)
        job.salary_range = self._extract_salary(job_text)
        
        # Split into sections
        sections = self._split_into_sections(job_text)
        
        # Extract from sections
        job.responsibilities = self._extract_list_items(sections.get('responsibilities', ''))
        job.benefits = self._extract_list_items(sections.get('benefits', ''))
        
        # Extract skills
        requirements_text = sections.get('requirements', '')
        preferred_text = sections.get('preferred', '')
        
        job.required_skills = self._extract_skills_from_section(requirements_text, job_text)
        job.preferred_skills = self._extract_skills_from_section(preferred_text)
        
        # Extract education
        job.education_requirements = self._extract_education(requirements_text + ' ' + job_text)
        
        return job
    
    def _extract_job_title(self, text: str) -> str:
        """Extract job title (usually first line or after 'Position:')"""
        lines = text.strip().split('\n')
        
        # Check for explicit title patterns
        title_patterns = [
            r'(?i)(?:job\s*title|position|role)\s*[:\-]\s*(.+)',
            r'(?i)^((?:senior|junior|lead|principal|staff)?\s*\w+(?:\s+\w+){0,4}(?:engineer|developer|manager|analyst|designer|scientist|specialist|coordinator|director|consultant))',
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        # Fallback: first non-empty line that looks like a title
        for line in lines[:5]:
            line = line.strip()
            if line and len(line) < 100 and not re.search(r'[@\d{10,}]', line):
                # Looks like it could be a title
                if re.search(r'(?i)(engineer|developer|manager|analyst|designer|scientist)', line):
                    return line
        
        return lines[0].strip() if lines else ""
    
    def _extract_company(self, text: str) -> str:
        """Extract company name"""
        patterns = [
            r'(?i)(?:company|employer|organization)\s*[:\-]\s*(.+?)(?:\n|$)',
            r'(?i)(?:at|@)\s+([A-Z][A-Za-z0-9\s&]+?)(?:\s+is|\s+are|\n|$)',
            r'(?i)^([A-Z][A-Za-z0-9\s&]+?)\s+is\s+(?:looking|hiring|seeking)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_location(self, text: str) -> str:
        """Extract job location"""
        patterns = [
            r'(?i)(?:location|based\s+in|office)\s*[:\-]\s*(.+?)(?:\n|$)',
            r'(?i)(remote|hybrid|on[\s-]?site)',
            r'([A-Z][a-z]+(?:,\s*[A-Z]{2})?)(?:\s*[-–]\s*(remote|hybrid))?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                location = match.group(1).strip() if match.group(1) else match.group(0).strip()
                return location
        
        return ""
    
    def _extract_job_type(self, text: str) -> str:
        """Extract job type (full-time, part-time, etc.)"""
        job_types = []
        
        for pattern in self.JOB_TYPE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                job_types.append(match.group(1).strip())
        
        return ', '.join(job_types) if job_types else ""
    
    def _extract_experience(self, text: str) -> str:
        """Extract required experience level"""
        for pattern in self.EXPERIENCE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1) if match.group(1).isdigit() else match.group(1)
        
        return ""
    
    def _extract_salary(self, text: str) -> str:
        """Extract salary range if present"""
        for pattern in self.SALARY_PATTERNS:
            match = re.search(pattern, text)
            if match:
                return match.group(0).strip()
        
        return ""
    
    def _split_into_sections(self, text: str) -> Dict[str, str]:
        """Split job description into sections"""
        sections = {}
        lines = text.split('\n')
        
        current_section = 'intro'
        current_content = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check if this line is a section header
            found_section = None
            for section, pattern in self.SECTION_PATTERNS.items():
                if re.match(pattern, line_stripped, re.IGNORECASE) or re.search(pattern, line_stripped, re.IGNORECASE):
                    found_section = section
                    break
            
            if found_section:
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start new section
                current_section = found_section
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def _extract_list_items(self, text: str) -> List[str]:
        """Extract bullet points or list items from text"""
        items = []
        
        if not text.strip():
            return items
        
        # Split by common list markers
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove bullet points and numbering
            item = re.sub(r'^[\s•\-\*\d\.○●◦]+\s*', '', line)
            
            if item and len(item) > 5:
                items.append(item)
        
        return items
    
    def _extract_skills_from_section(self, section_text: str, full_text: str = "") -> List[str]:
        """Extract skills from a section"""
        skills = set()
        
        text_to_search = section_text if section_text else full_text
        
        if not text_to_search:
            return list(skills)
        
        # Common technical skill patterns
        tech_patterns = [
            # Programming languages
            r'\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|Ruby|Go|Rust|Swift|Kotlin|PHP|Scala|R|MATLAB)\b',
            # Frameworks and libraries
            r'\b(React|Angular|Vue|Node\.?js|Django|Flask|FastAPI|Spring|\.NET|TensorFlow|PyTorch|Keras|Scikit-learn|Pandas|NumPy)\b',
            # Databases
            r'\b(SQL|MySQL|PostgreSQL|MongoDB|Redis|Elasticsearch|DynamoDB|Cassandra|Oracle|SQLite)\b',
            # Cloud and DevOps
            r'\b(AWS|Azure|GCP|Docker|Kubernetes|Jenkins|GitLab|GitHub|Terraform|Ansible|CI/CD)\b',
            # Data/ML
            r'\b(Machine Learning|Deep Learning|NLP|Computer Vision|Data Science|Data Analysis|Big Data|Spark|Hadoop|ETL)\b',
            # General tech
            r'\b(REST|API|GraphQL|Microservices|Agile|Scrum|Linux|Unix|Git|HTML|CSS|SASS)\b'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text_to_search, re.IGNORECASE)
            for match in matches:
                skills.add(match.strip())
        
        # Also extract from bullet points
        items = self._extract_list_items(section_text)
        for item in items:
            # Short items are likely skill names
            if len(item) < 50:
                # Clean and add
                skill = re.sub(r'^\d+\+?\s*years?\s*(?:of\s*)?(?:experience\s*(?:in|with)?\s*)?', '', item, flags=re.IGNORECASE)
                skill = skill.strip()
                if skill and len(skill) > 1:
                    skills.add(skill)
        
        return list(skills)
    
    def _extract_education(self, text: str) -> List[str]:
        """Extract education requirements"""
        education = []
        
        degree_patterns = [
            r"(?i)(bachelor'?s?|master'?s?|phd|ph\.d|doctorate|associate'?s?)\s*(?:degree)?\s*(?:in\s+[\w\s]+)?",
            r'(?i)(B\.?S\.?|M\.?S\.?|B\.?A\.?|M\.?A\.?|MBA|PhD)\s*(?:in\s+[\w\s]+)?',
            r'(?i)(computer science|engineering|mathematics|statistics|physics|data science|information technology)'
        ]
        
        for pattern in degree_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if match and match not in education:
                    education.append(match.strip())
        
        return education
