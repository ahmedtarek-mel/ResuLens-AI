
import os
import sys
import io
import json
from flask import Flask, render_template, request, jsonify, send_file, session, url_for, redirect
from werkzeug.utils import secure_filename

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.parsers.resume_parser import ResumeParser
from src.parsers.job_parser import JobDescriptionParser
from src.nlp.skill_extractor import SkillExtractor
from src.matching.hybrid_scorer import HybridScorer
from src.utils.report_generator import ReportGenerator

app = Flask(__name__)
# Secure secret key for session management
app.secret_key = os.environ.get('SECRET_KEY', 'resulens-dev-secret-key-change-in-prod')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Configuration for weights (Could be moved to config file)
WEIGHTS = {
    'semantic': 0.40,
    'skill': 0.35,
    'keyword': 0.25
}

SAMPLE_RESUME = """
JOHN SMITH
Senior Machine Learning Engineer
Email: john.smith@email.com | Phone: (555) 123-4567
LinkedIn: linkedin.com/in/johnsmith | GitHub: github.com/johnsmith

PROFESSIONAL SUMMARY
Passionate Machine Learning Engineer with 5+ years of experience developing NLP systems and 
deep learning models. Expertise in Python, TensorFlow, PyTorch, and cloud deployments. 
Led teams to deliver production-grade AI solutions serving millions of users.

TECHNICAL SKILLS
Languages: Python, SQL, JavaScript, C++
ML/DL: TensorFlow, PyTorch, Keras, Scikit-learn, HuggingFace
NLP: BERT, GPT, Transformers, spaCy, NLTK, Word2Vec
Cloud: AWS (SageMaker, Lambda, EC2), GCP (AI Platform), Docker, Kubernetes
Data: Pandas, NumPy, Spark, PostgreSQL, MongoDB
Tools: Git, MLflow, Airflow, Jupyter, VS Code

EXPERIENCE
Senior Machine Learning Engineer | TechCorp AI | 2021 - Present
- Led development of NLP pipeline processing 10M+ documents daily
- Implemented transformer models achieving 95% accuracy on text classification
- Reduced model inference time by 60% through optimization
- Mentored team of 4 junior engineers

Machine Learning Engineer | DataSolutions Inc | 2019 - 2021
- Built recommendation system increasing user engagement by 35%
- Developed computer vision models for quality control
- Created data pipelines using Apache Spark and Airflow

Data Scientist | Analytics Co | 2018 - 2019
- Conducted statistical analysis and A/B testing
- Built predictive models using Random Forest and XGBoost
- Created interactive dashboards with Tableau

EDUCATION
M.S. Computer Science, Machine Learning | Stanford University | 2018
B.S. Computer Science | UC Berkeley | 2016

CERTIFICATIONS
- AWS Machine Learning Specialty
- TensorFlow Developer Certificate
- Deep Learning Specialization - Coursera
"""

SAMPLE_JOB_DESCRIPTION = """
Senior NLP Engineer
TechVision AI | San Francisco, CA | Remote Friendly

About Us:
TechVision AI is a leading artificial intelligence company building next-generation 
natural language processing solutions. We're looking for talented engineers to join 
our growing team.

What You'll Do:
- Design and implement state-of-the-art NLP models for text understanding
- Build scalable ML pipelines processing millions of documents
- Collaborate with cross-functional teams to deliver AI solutions
- Mentor junior team members and contribute to technical decisions
- Research and implement latest advances in transformer architectures

Requirements:
- 4+ years of experience in Machine Learning or NLP roles
- Strong proficiency in Python and deep learning frameworks (TensorFlow, PyTorch)
- Experience with transformer models (BERT, GPT, T5)
- Solid understanding of NLP concepts: tokenization, embeddings, attention
- Experience with cloud platforms (AWS, GCP, or Azure)
- Strong communication and collaboration skills

Preferred Qualifications:
- M.S. or Ph.D. in Computer Science, ML, or related field
- Experience with distributed training and model optimization
- Contributions to open-source ML projects
- Publications in NLP/ML conferences

Tech Stack:
Python, PyTorch, HuggingFace, FastAPI, Docker, Kubernetes, AWS SageMaker

Benefits:
- Competitive salary: $180,000 - $250,000
- Equity package
- Remote-first culture with flexible hours
- Health, dental, and vision insurance
- $5,000 annual learning budget
- Unlimited PTO

TechVision AI is an equal opportunity employer.
"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data_source = request.form.get('source')
        
        resume_text = ""
        job_description = ""
        resume_filename = "Resume.pdf"
        
        if data_source == 'demo':
            resume_text = SAMPLE_RESUME
            job_description = SAMPLE_JOB_DESCRIPTION
            resume_filename = "Sample_Resume.txt"
        else:
            # Handle File Upload
            if 'resume_file' not in request.files:
                return jsonify({'error': 'No resume file uploaded'}), 400
            
            file = request.files['resume_file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
                
            resume_filename = secure_filename(file.filename)
            job_description = request.form.get('job_description', '')
            
            if len(job_description) < 50:
                 return jsonify({'error': 'Job description is too short (min 50 chars)'}), 400

            # Parse Resume
            parser = ResumeParser()
            try:
                # We need to read the file into bytes first
                file_content = file.read()
                file_stream = io.BytesIO(file_content)
                resume_data = parser.parse(file_stream, resume_filename)
                resume_text = resume_data.raw_text
                # cache structured data
                session['resume_data'] = resume_data.to_dict()
            except Exception as e:
                return jsonify({'error': f'Failed to parse resume: {str(e)}'}), 500

        # Perform Analysis
        job_parser = JobDescriptionParser()
        skill_extractor = SkillExtractor()
        
        job_data = job_parser.parse(job_description)
        
        resume_skills = skill_extractor.extract(resume_text)
        job_skills = skill_extractor.extract(job_description)
        
        scorer = HybridScorer(
            semantic_weight=WEIGHTS['semantic'],
            skill_weight=WEIGHTS['skill'],
            keyword_weight=WEIGHTS['keyword']
        )
        
        result = scorer.calculate_match(
            resume_text,
            job_description,
            resume_skills,
            job_skills
        )
        
        # Determine Color Grade
        grades = {
             'A+': 'Excellent', 'A': 'Excellent',
             'B+': 'Good', 'B': 'Good',
             'C+': 'Average', 'C': 'Average',
             'D': 'Needs Work', 'F': 'Poor'
        }
        
        # Serialize result for session/response
        match_result = {
            "overall_score": result.overall_score,
            "semantic_score": result.semantic_score,
            "skill_score": result.skill_score,
            "keyword_score": result.keyword_score,
            "grade": result.grade,
            "grade_text": grades.get(result.grade, 'Unknown'),
            "matched_skills": result.matched_skills,
            "missing_skills": result.missing_skills,
            "extra_skills": result.extra_skills,
            "recommendations": result.recommendations,
            "top_keywords": result.top_keywords
        }
        
        # Store in session for report generation
        session['last_result'] = match_result
        session['job_title'] = job_data.title if job_data.title else 'Job Position'
        
        # If not demo, we already set resume_data. If demo, we didn't. 
        # For simplicity in report, we might need basic data.
        if data_source == 'demo':
             session['resume_data'] = {'contact_info': {}, 'skills': []} # Simplified

        return jsonify(match_result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/report')
def download_report():
    result_data = session.get('last_result')
    if not result_data:
        return "No analysis found", 404
        
    # Reconstruct MatchResult object (partial is enough for report generator usually)
    from src.matching.hybrid_scorer import MatchResult
    
    # We need to recreate the object because the ReportGenerator expects it
    # We'll create a dummy object and populate it
    result = MatchResult(
        overall_score=result_data['overall_score'],
        semantic_score=result_data['semantic_score'],
        skill_score=result_data['skill_score'],
        keyword_score=result_data['keyword_score'],
        grade=result_data['grade'],
        matched_skills=result_data['matched_skills'],
        missing_skills=result_data['missing_skills'],
        extra_skills=result_data['extra_skills'],
        recommendations=result_data['recommendations'],
        top_keywords=result_data['top_keywords'],
        resume_skill_count=len(result_data['matched_skills']) + len(result_data['extra_skills']),
        job_skill_count=len(result_data['matched_skills']) + len(result_data['missing_skills'])
    )
    
    generator = ReportGenerator()
    try:
        pdf_bytes = generator.generate(
            result,
            session.get('job_title', 'Job Position')
        )
        
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name='ResuLens_Analysis_Report.pdf'
        )
    except Exception as e:
        return f"Error generating report: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
