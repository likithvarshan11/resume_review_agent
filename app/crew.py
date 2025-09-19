import os
from dotenv import load_dotenv
from crewai import Crew, LLM

from .agents import resume_reader_feedback, resume_writing_advisor, job_researcher
from .tasks import resume_feedback_task, resume_advisor_task, research_task
from .utils import extract_text_from_resume

# Load env vars
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # make sure key is uppercase in .env
MODEL_NAME = os.getenv("MODEL", "groq/llama-3.1-70b-versatile")  # default model

# Configure LLM with Groq
llm = LLM(
    model=MODEL_NAME,
    api_key=GROQ_API_KEY,
)

def clean_output(output):
    if not output or not hasattr(output, "raw"):
        return ""
    text = output.raw
    return (
        text.replace("```markdown", "")
            .replace("```", "")
            .replace("Here is the result:", "")
            .strip()
    )

crew = Crew(
    agents=[resume_reader_feedback, resume_writing_advisor, job_researcher],
    tasks=[resume_feedback_task, resume_advisor_task, research_task],
    llm=llm,   # 👈 attach LLM here
    verbose=True
)

def resume_agent(file_path: str, location: str):
    resume_text = extract_text_from_resume(file_path)
    
    crew.kickoff(inputs={"resume": resume_text, "location": location})
    
    if not resume_feedback_task.output or not resume_advisor_task.output or not research_task.output:
        raise RuntimeError("Crew execution failed. One or more tasks did not return output.")
    
    feedback = clean_output(resume_feedback_task.output)
    improved_resume = clean_output(resume_advisor_task.output)
    job_roles = [
        line.strip("- ").strip()
        for line in clean_output(research_task.output).split("\n")
        if line.strip()
    ]

    return {
        "feedback": feedback,
        "improved_resume": improved_resume,
        "job_roles": job_roles
    }
