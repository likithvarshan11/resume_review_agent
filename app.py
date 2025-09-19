# Entry point (Gradio UI + app runner)

import warnings
warnings.filterwarnings("ignore")
import os

import gradio as gr
from app import resume_agent

def run_resume_agent(file_path, location):
    if not file_path:
        return "Please upload a resume file.", "", ""
    
    if not location:
        return "Please enter a preferred location.", "", ""
    
    try:
        result = resume_agent(file_path, location)
        return result["feedback"], result["improved_resume"], "\n".join(result["job_roles"])
    except Exception as e:
        return f"Error processing resume: {str(e)}", "", ""

# Create custom CSS to fix styling issues
custom_css = """
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
}

.upload-container {
    border: 2px dashed #ccc;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
}

.output-textbox {
    min-height: 200px;
}
"""

# Create the Gradio interface
with gr.Blocks(
    theme=gr.themes.Soft(), 
    css=custom_css,
    title="Resume Review Agent"
) as demo:
    gr.Markdown("# 📄 Resume Review & Job Matching Agent")
    gr.Markdown("*Upload your resume and get professional feedback, an improved version, and matching job opportunities.*")

    with gr.Row():
        with gr.Column(scale=1):
            resume_upload = gr.File(
                label="📎 Upload Your Resume",
                file_types=[".pdf", ".docx"],
                file_count="single",
                height=150,
                elem_classes=["upload-container"]
            )
            location_input = gr.Textbox(
                label="📍 Preferred Job Location",
                placeholder="e.g., New York, NY or San Francisco, CA",
                lines=1
            )
            submit_button = gr.Button(
                value="🚀 Analyze Resume", 
                variant="primary",
                size="lg"
            )

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("📝 Resume Feedback"):
                    feedback_output = gr.Textbox(
                        label="Professional Feedback",
                        lines=10,
                        elem_classes=["output-textbox"],
                        placeholder="Feedback will appear here..."
                    )
                
                with gr.TabItem("📋 Improved Resume"):
                    improved_resume_output = gr.Markdown(
                        label="Enhanced Resume",
                        value="Improved resume will appear here..."
                    )
                
                with gr.TabItem("💼 Job Opportunities"):
                    job_roles_output = gr.Textbox(
                        label="Matching Job Roles",
                        lines=15,
                        elem_classes=["output-textbox"],
                        placeholder="Job opportunities will appear here..."
                    )

    # Event handlers
    submit_button.click(
        fn=lambda: gr.update(value="🔄 Processing...", interactive=False),
        inputs=[], 
        outputs=submit_button
    ).then(
        fn=run_resume_agent,
        inputs=[resume_upload, location_input],
        outputs=[feedback_output, improved_resume_output, job_roles_output]
    ).then(
        fn=lambda: gr.update(value="🚀 Analyze Resume", interactive=True),
        inputs=[], 
        outputs=submit_button
    )

    # Add example section
    gr.Markdown("## 💡 Tips for Best Results")
    gr.Markdown("""
    - Upload a clear, well-formatted resume in PDF or DOCX format
    - Specify your preferred job location accurately
    - Ensure your resume includes relevant work experience, skills, and education
    - The analysis typically takes 1-2 minutes to complete
    """)

if __name__ == "__main__":
    # Configure demo launch with proper settings to avoid console errors
    demo.queue(max_size=10)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,              # Disable sharing to avoid HuggingFace postMessage issues
        show_error=True,
        quiet=False,
        favicon_path=None         # Disable favicon to avoid 404 errors
    )