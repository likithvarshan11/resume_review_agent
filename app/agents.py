from crewai import Agent
from crewai.tools import BaseTool
from dotenv import load_dotenv
import os
import requests
from typing import Type
from pydantic import BaseModel, Field

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")  # Try uppercase first, then lowercase
SERPER_KEY = os.getenv("SERPER_KEY") or os.getenv("serper_key")  # Try uppercase first, then lowercase
MODEL_NAME = os.getenv("MODEL") or f"groq/{os.getenv('model')}"  # Use MODEL if set, otherwise add groq prefix

# Custom tool schema for search
class SearchToolInput(BaseModel):
    query: str = Field(..., description="Search query for job opportunities")

# Custom search tool to replace SerperDevTool
class JobSearchTool(BaseTool):
    name: str = "job_search_tool"
    description: str = "Search for job opportunities using web search"
    args_schema: Type[BaseModel] = SearchToolInput

    def _run(self, query: str) -> str:
        """Perform job search using Serper API"""
        if not SERPER_KEY:
            raise ValueError("Missing SERPER_KEY. Please set it in your .env file.")

        try:
            url = "https://google.serper.dev/search"
            payload = {
                'q': f"{query} jobs",
                'gl': 'us',
                'hl': 'en'
            }
            headers = {
                'X-API-KEY': SERPER_KEY,
                'Content-Type': 'application/json'
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = []
                
                # Process organic results
                if 'organic' in data:
                    for item in data['organic'][:5]:  # Limit to 5 results
                        results.append(
                            f"Title: {item.get('title', 'N/A')}\n"
                            f"URL: {item.get('link', 'N/A')}\n"
                            f"Snippet: {item.get('snippet', 'N/A')}\n"
                        )
                
                return "\n".join(results) if results else "No job results found."
            else:
                return f"Search API error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Search error: {str(e)}"

# Initialize the custom job search tool
job_search_tool = JobSearchTool()

resume_reader_feedback = Agent(
    role="Professional Resume Advisor",
    goal="Give feedback on the resume to make it stand out in the job market.",
    verbose=True,
    backstory="With a strategic mind and an eye for detail, you excel at providing feedback on resumes to highlight the most relevant skills and experiences."
)

resume_writing_advisor = Agent(
    role="Professional Resume Writer",
    goal="Refine and write a professional resume that stands out in the job market, based on the advisor feedback.",
    verbose=True,
    backstory="With years of experience in crafting compelling resumes, you help individuals showcase their unique strengths and accomplishments."
)

job_researcher = Agent(
    role="Professional Job Search Advisor and Senior Recruitment Consultant",
    goal="Find the 5 most relevant, recently posted jobs based on the improved resume and location.",
    tools=[job_search_tool],
    verbose=True,
    backstory="With extensive knowledge of industries and job markets, you excel at finding the most suitable job opportunities for your clients."
)
