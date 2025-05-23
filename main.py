import os
import uuid
import pypandoc # For Markdown to PDF conversion
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv
import tempfile
import shutil

# --- Configuration ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit(1)

# Ensure Pandoc is available (pypandoc-binary should help, but good to check)
try:
    pypandoc.get_pandoc_version()
    print(f"Pandoc version: {pypandoc.get_pandoc_version()} found.")
except OSError:
    print("Error: Pandoc not found. Please install Pandoc and add it to your PATH.")
    print("Visit https://pandoc.org/installing.html")
    print("If using pypandoc-binary, ensure it installed correctly.")
    # exit(1) # Exit if Pandoc is critical and not found; for now, let it try.

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(
    title="Scientific Article Generator",
    description="Generates a draft scientific article in PDF format based on a topic, using an LLM for Markdown generation and Pandoc for PDF conversion. Includes APA-style references (user to verify for hallucinations)."
)

# --- Pydantic Models ---
class TopicRequest(BaseModel):
    topic: str = Body(..., embed=True, description="The topic for the scientific article.")
    # min_references: int = Body(5, ge=3, le=15, description="Minimum number of references to attempt to generate.")

class ArticleResponse(BaseModel):
    message: str
    markdown_filename: str
    pdf_filename: str
    references_generated: List[str] # To help with manual checking

# --- Helper Function for LLM Interaction ---
def generate_article_markdown_from_llm(topic: str, min_references: int = 7) -> (str | None, List[str]):
    """
    Generates the Markdown content for a scientific article using an LLM.
    Returns the Markdown string and a list of extracted references.
    """
    system_prompt = f"""
    You are an AI assistant specialized in drafting scientific articles.
    Your task is to generate a comprehensive draft of a scientific article in MARKDOWN format based on the provided topic.

    The article MUST follow this structure:
    1.  **Abstract:** A concise summary (approx. 150-250 words).
    2.  **1. Introduction:**
        - Background and context.
        - Problem statement or research question.
        - Significance of the research.
        - Objectives of the article.
        - (Use subheadings like 1.1, 1.2 if appropriate)
    3.  **2. Literature Review:** (If distinct from Introduction)
        - Summary of existing relevant research.
        - Identify gaps your hypothetical research addresses.
        - (Use subheadings like 2.1, 2.2 if appropriate)
    4.  **3. Methodology:**
        - Describe the hypothetical research design.
        - Data collection methods.
        - Data analysis techniques.
        - (Use subheadings like 3.1, 3.2 if appropriate)
    5.  **4. Results:** (Present hypothetical findings)
        - Use clear descriptions.
        - You MAY include simple tables using Markdown table syntax if relevant to illustrate hypothetical data.
        - (Use subheadings like 4.1, 4.2 if appropriate)
    6.  **5. Discussion:**
        - Interpretation of the hypothetical results.
        - Comparison with literature.
        - Limitations of the hypothetical study.
        - Implications for future research or practice.
        - (Use subheadings like 5.1, 5.2 if appropriate)
    7.  **6. Conclusion:**
        - Briefly summarize the main points and hypothetical findings.
        - Reiterate the significance.
    8.  **References:**
        - A list of AT LEAST {min_references} plausible-sounding academic references.
        - **CRITICAL: Format ALL references strictly in APA 7th edition style.**
        - **CRITICAL: Ensure EVERY reference listed in the 'References' section is CITED in the body of the text (Introduction to Conclusion) using APA 7th edition in-text citation style, e.g., (Author, Year) or Author (Year).**

    General Instructions:
    - Output ONLY the Markdown content. No preamble or explanations outside the Markdown.
    - Use standard Markdown for headings (#, ##, ###), lists, bold, italics, and tables.
    - The content should be scientifically plausible and well-organized for the given topic.
    - Aim for a total article length that is substantial but manageable (e.g., 1500-3000 words, excluding references).
    - The language should be formal and academic.
    """
    user_prompt = f"Generate a scientific article on the topic: \"{topic}\""

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-0125", # Or gpt-4-turbo for better quality and longer context
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5 # Allow for some creativity but maintain structure
        )
        markdown_content = completion.choices[0].message.content

        if not markdown_content:
            return None, []

        # Rudimentary extraction of references from the "References" section for later display
        # This is a simple heuristic and might need refinement.
        references_section = []
        in_references_section = False
        if markdown_content:
            for line in markdown_content.splitlines():
                if line.strip().lower().startswith("## references") or \
                   line.strip().lower().startswith("# references") or \
                   line.strip().lower().startswith("**references**"):
                    in_references_section = True
                    continue
                if in_references_section:
                    if line.strip().startswith("#") or line.strip().startswith("##"): # Start of a new main section
                        break
                    if line.strip() and not line.strip().startswith("---"): # Avoid horizontal rules
                        references_section.append(line.strip())
        
        return markdown_content, references_section

    except Exception as e:
        print(f"Error communicating with OpenAI: {e}")
        return None, []

# --- Helper Function for Markdown to PDF Conversion ---
def convert_markdown_to_pdf_pandoc(markdown_string: str, output_pdf_path: str) -> bool:
    """
    Converts a Markdown string to a PDF file using Pandoc.
    Returns True on success, False on failure.
    """
    try:
        # Using pypandoc.convert_text
        # You might need to specify a PDF engine if the default doesn't work well, e.g., 'xelatex', 'pdflatex', 'wkhtmltopdf', 'weasyprint'
        # For APA, a LaTeX engine (pdflatex, xelatex, lualatex) often gives the best typographical results if you have a TeX distribution installed.
        # Pandoc will try to choose a sensible default.
        # To use a specific citation style file (CSL) for APA, you'd add:
        # extra_args=['--csl=apa.csl', '--bibliography=references.bib']
        # This requires managing bibliography files, which is more advanced than the current LLM-only approach.
        # For now, we rely on the LLM to embed APA formatting directly.
        pypandoc.convert_text(
            markdown_string,
            'pdf',
            format='md',
            outputfile=output_pdf_path,
            extra_args=['--pdf-engine=pdflatex'] # Example: try specifying an engine; remove if default works
                                               # common engines: pdflatex, xelatex, lualatex, wkhtmltopdf, weasyprint
                                               # LaTeX engines often require a TeX distribution (e.g., MiKTeX, TeX Live).
        )
        print(f"PDF successfully generated: {output_pdf_path}")
        return True
    except Exception as e: # Catch RuntimeError from pypandoc or other exceptions
        print(f"Error converting Markdown to PDF using Pandoc: {e}")
        print("Make sure Pandoc is installed and in your PATH, or a suitable PDF engine is available.")
        print("Consider installing a TeX distribution like MiKTeX (Windows) or TeX Live (Linux/macOS) for LaTeX-based PDF engines.")
        return False

# --- API Endpoint ---
@app.post("/generate-article/", response_class=FileResponse) # Returns the PDF directly
async def generate_article_endpoint(request: TopicRequest):
    """
    Generates a scientific article.
    Input: JSON with "topic" field.
    Output: PDF file of the generated article.
    The response will also include custom headers with metadata.
    """
    topic = request.topic
    if not topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")

    print(f"Received topic: {topic}")
    markdown_content, extracted_references = generate_article_markdown_from_llm(topic)

    if not markdown_content:
        raise HTTPException(status_code=500, detail="Failed to generate article content from LLM.")

    # Create a temporary directory to store MD and PDF files
    temp_dir = tempfile.mkdtemp()
    base_filename = f"article_{uuid.uuid4()}"
    temp_md_path = os.path.join(temp_dir, f"{base_filename}.md")
    temp_pdf_path = os.path.join(temp_dir, f"{base_filename}.pdf")

    with open(temp_md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"Markdown content saved to: {temp_md_path}")

    if not convert_markdown_to_pdf_pandoc(markdown_content, temp_pdf_path):
        shutil.rmtree(temp_dir) # Clean up temp directory
        raise HTTPException(status_code=500, detail="Failed to convert Markdown to PDF. Check Pandoc installation and PDF engine.")

    # Prepare response headers with metadata
    # Note: Extracting references accurately from generated Markdown is non-trivial.
    # The `extracted_references` list is a best-effort attempt.
    # A more robust solution would involve the LLM outputting references in a structured way (e.g., a JSON list).
    headers = {
        "X-Generated-Markdown-Path": temp_md_path, # For debugging, not for client
        "X-Generated-PDF-Path": temp_pdf_path,     # For debugging, not for client
        "X-Topic": topic,
        "X-References-Count": str(len(extracted_references)),
        # "X-References-List": json.dumps(extracted_references) # Could be too long for a header
        "Content-Disposition": f"attachment; filename=\"{topic.replace(' ', '_')}_article.pdf\""
    }
    print(f"Extracted references count: {len(extracted_references)}")
    # For the assignment, you'd manually review the references in the PDF.

    # FileResponse will automatically handle cleanup of the temp_pdf_path if background tasks are used,
    # but here we'll clean up the whole temp_dir after response is sent if not using background tasks.
    # For simplicity, we let the OS clean up temp dirs, or you can implement a BackgroundTask.
    # To ensure cleanup:
    # from starlette.background import BackgroundTask
    # return FileResponse(temp_pdf_path, media_type='application/pdf', filename=f"{topic.replace(' ', '_')}_article.pdf", headers=headers, background=BackgroundTask(shutil.rmtree, temp_dir))
    # For now, simpler FileResponse:
    return FileResponse(temp_pdf_path, media_type='application/pdf', filename=f"{topic.replace(' ', '_')}_article.pdf", headers=headers)


# --- To run this application (save as main.py) ---
# 1. Ensure Pandoc is installed (https://pandoc.org/installing.html) if pypandoc-binary fails.
#    And a LaTeX distribution if using pdflatex/xelatex (e.g., MiKTeX, TeX Live).
# 2. Install dependencies: pip install -r requirements.txt
# 3. Create a .env file with your OPENAI_API_KEY.
# 4. Run with Uvicorn: uvicorn main:app --reload
# 5. Access in browser (Swagger UI for POST): http://127.0.0.1:8000/docs
#    Or use curl:
#    curl -X POST "http://127.0.0.1:8000/generate-article/" \
#         -H "Content-Type: application/json" \
#         -d "{\"topic\":\"The Impact of AI on Renewable Energy Management\"}" \
#         --output generated_article.pdf