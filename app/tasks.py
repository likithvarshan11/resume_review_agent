from crewai import Task
from .agents import resume_reader_feedback, resume_writing_advisor, job_researcher

resume_feedback_task = Task(
    description=(
        """Give feedback on the resume to make it stand out for recruiters. 
        Review every section, including summary, work experience, skills, and education. 
        Suggest additional sections if missing. Score the resume out of 10. 
        This is the resume: {resume}"""
    ),
    expected_output="The overall score of the resume followed by bullet-point feedback.",
    agent=resume_reader_feedback
)

resume_advisor_task = Task(
    description=(
        """Rewrite the resume based on advisor feedback. 
        Update summary, work experience, skills, and education. 
        Do not make up facts. This is the resume: {resume}"""
    ),
    expected_output="Improved resume in markdown format.",
    context=[resume_feedback_task],
    agent=resume_writing_advisor
)

research_task = Task(
    description=(
        """Find 5 relevant recent job postings based on the improved resume and location: {location}. 
        Provide links and detailed job descriptions."""
    ),
    expected_output="Markdown list of 5 job openings with links and descriptions.",
    agent=job_researcher
)
