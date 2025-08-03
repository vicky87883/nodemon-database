import os
import fitz  # PyMuPDF
import docx
from groq import Groq
import re

def extract_pdf_content(file_path):
    try:
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting PDF content: {e}")
        return ""

def extract_docx_content(file_path):
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error extracting DOCX content: {e}")
        return ""

def extract_tex_content(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error extracting TeX content: {e}")
        return ""

def extract_content(file_path):
    """
    Extract text content from the given file based on its extension.
    Supported file types: PDF, DOCX, and TeX.
    """
    ext = os.path.splitext(file_path)[1].lower()
    print(f"Detected file extension: {ext}")  # Log the extension for debugging

    if ext == ".pdf":
        return extract_pdf_content(file_path)
    elif ext == ".docx":
        return extract_docx_content(file_path)
    elif ext == ".tex":
        return extract_tex_content(file_path)
    else:
        # Log unsupported file type and raise an error
        print(f"Unsupported file type: {ext}")
        raise ValueError("Unsupported file type")

def analyze_content_with_groq(file_path):
    try:
        # Extract the complete text manually from the file (this is the content column)
        content_text = extract_content(file_path)

        # Groq API logic to extract title, author, algorithms, references
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise Exception("GROQ_API_KEY not set in environment variables")

        client = Groq(api_key=api_key)
        prompt = (
            "Extract the following fields from the research paper text:\n"
            "title, author, algorithms, references.\n"
            "Return as JSON with keys: title, author, algorithms, references.\n\n"
            f"Paper:\n{content_text[:8000]}"  # Send the content text for analysis
        )

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.2,
        )

        print("LLM raw response:", response.choices[0].message.content)

        import json
        content = response.choices[0].message.content

        # Extract JSON from Markdown code block if present
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            match = re.search(r"(\{.*\})", content, re.DOTALL)
            json_str = match.group(1) if match else ""

        try:
            data = json.loads(json_str)
        except Exception as e:
            print("JSON parsing error:", e)
            data = {
                "title": "",
                "author": "",
                "content": content_text,  # Fill content with manually extracted text
                "algorithms": "",
                "references": "",
            }

        # Fill the content column with the raw text extracted from the file
        data["content"] = content_text

        return data

    except ValueError as ve:
        print(f"Error: {ve}")
        return {"error": "Unsupported file type or failed to extract content"}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {"error": "An unexpected error occurred"}
