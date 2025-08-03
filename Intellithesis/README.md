# Intellithesis - Research Paper Analysis System

A Django-based application that uses a **two-step approach** to analyze and categorize research papers: first extracting content using traditional methods, then using Groq LLM to intelligently categorize and organize the extracted content.

## 🚀 Recent Improvements (Latest Update)

### ✅ Fixed Issues
- **Database Model Fixed**: Updated ResearchPaper model with correct field names and added missing fields
- **Two-Step Algorithm**: Completely redesigned approach for better content extraction and categorization
- **Content Extraction Improved**: Traditional text processing + AI analysis for robust results
- **Error Handling**: Comprehensive error handling throughout the pipeline

### 🎯 Algorithm Enhancements
- **Step 1: Traditional Extraction**: Uses text processing to extract document sections
- **Step 2: AI Analysis**: Uses Groq LLM to categorize and improve extracted content
- **Better Content Organization**: More accurate section identification and categorization
- **Intelligent Fallbacks**: Multiple strategies ensure content is always extracted
- **Enhanced Error Recovery**: Graceful degradation when API calls fail

## Features

- Upload research papers (PDF, DOCX, LaTeX)
- **Two-step content analysis**: Traditional extraction + AI categorization
- Automatic text extraction from various file formats
- AI-powered content analysis using Groq LLM
- Structured data extraction (title, authors, abstract, sections)
- Content categorization (Research Paper, Thesis, Journal)
- PostgreSQL database storage
- REST API support
- **NEW**: Two-step algorithm approach
- **NEW**: Traditional text processing + AI analysis
- **NEW**: Enhanced content organization

## Setup Instructions

### 1. Prerequisites

- Python 3.8+
- PostgreSQL database
- Groq API key

### 2. Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up PostgreSQL database:
   - Create database: `Intellithesis_Scholar_DB`
   - Update database settings in `Intellithesis/settings.py` if needed

4. Run migrations:
   ```bash
   python manage.py migrate
   ```

5. Test the application:
   ```bash
   python manage.py runserver
   ```

### 3. Configuration

- Update Groq API key in `scholar/views.py` if needed
- Configure database settings in `Intellithesis/settings.py`

### 4. Running the Application

```bash
python manage.py runserver
```

Access the application at:
- Upload page: http://localhost:8000/scholar/upload/
- List page: http://localhost:8000/scholar/list/

## API Endpoints

- `POST /scholar/upload/` - Upload and analyze research paper
- `GET /scholar/list/` - List all analyzed papers

## Algorithm Details

### Two-Step Approach

The improved algorithm uses a **two-step process** for better accuracy and efficiency:

#### Step 1: Traditional Content Extraction
- **Document Section Extraction**: Uses text processing to identify and extract sections
- **Pattern Recognition**: Identifies titles, authors, abstracts, and other sections
- **Keyword Detection**: Finds section boundaries using common academic keywords
- **Text Normalization**: Cleans and structures the extracted text

#### Step 2: AI-Powered Analysis
- **Content Categorization**: Uses Groq LLM to classify document type
- **Section Verification**: AI validates and improves extracted sections
- **Content Enhancement**: LLM provides better organization and categorization
- **Intelligent Classification**: Determines if document is Research Paper, Thesis, or Journal

### Content Extraction Strategies

#### Traditional Methods (Step 1)
- **Title Extraction**: Pattern matching and capitalization analysis
- **Author Detection**: Keyword search for author patterns
- **Section Parsing**: Keyword-based section boundary detection
- **Text Processing**: Line-by-line analysis for structure

#### AI Analysis (Step 2)
- **Content Verification**: LLM validates extracted sections
- **Type Classification**: Intelligent categorization based on content
- **Content Enhancement**: Improves and organizes extracted information
- **Fallback Handling**: Uses extracted content when AI fails

## How It Works

1. **File Upload**: User uploads research paper (PDF, DOCX, LaTeX)
2. **Text Extraction**: Traditional methods extract text from file
3. **Section Identification**: Algorithm identifies document sections using keywords
4. **Content Extraction**: Extracts title, authors, abstract, introduction, methodology, results, conclusion, references
5. **AI Analysis**: Groq LLM analyzes extracted content for categorization
6. **Content Enhancement**: LLM improves and validates extracted sections
7. **Database Storage**: All extracted and categorized content is saved
8. **Display**: Results are shown to user with organized sections

## Troubleshooting

### Common Issues

1. **Groq API Connection Failed**
   - Verify API key is correct
   - Check internet connection
   - Ensure Groq account has sufficient credits

2. **Database Connection Issues**
   - Verify PostgreSQL is running
   - Check database credentials in settings.py
   - Ensure database exists

3. **File Upload Issues**
   - Check file format (PDF, DOCX, LaTeX only)
   - Verify file size limits
   - Check media directory permissions

4. **Text Extraction Issues**
   - Ensure PyMuPDF is installed for PDF processing
   - Check python-docx for DOCX files
   - Verify file encoding for LaTeX files

### Debug Mode

Enable debug mode in `settings.py` to see detailed error messages:

```python
DEBUG = True
```

## Model Structure

The `ResearchPaper` model includes:
- `file`: Uploaded document
- `title`: Extracted title
- `authors`: Author names
- `abstract`: Paper abstract
- `introduction`: Introduction section
- `methodology`: Methodology section
- `results`: Results section
- `conclusion`: Conclusion section
- `references`: References section
- `content_type`: Classification (RESEARCH_PAPER, THESIS, JOURNAL)
- `extracted_date`: Timestamp of analysis

## Fine-tuning the Algorithm

To improve categorization accuracy:

1. **Adjust traditional extraction** in `extract_document_sections()` function
2. **Modify AI prompts** in `analyze_extracted_content_with_groq()` function
3. **Tune section detection** keywords for better boundary identification
4. **Adjust temperature** settings (currently 0.1 for consistency)
5. **Increase max_tokens** for longer responses (currently 3000)

## Performance Optimizations

- **Two-Step Efficiency**: Traditional extraction reduces AI token usage
- **Content Limiting**: 500 character previews for AI analysis
- **Caching**: Consider implementing response caching for repeated analyses
- **Batch Processing**: For multiple files, consider batch API calls
- **Async Processing**: For production, implement async processing for large files

## Security Notes

- Keep API keys secure
- Use environment variables for sensitive data
- Enable HTTPS in production
- Regular security updates

## Support

For issues and questions, check the troubleshooting section above or review the Django logs.

## Changelog

### v3.0 (Latest)
- ✅ Implemented two-step algorithm approach
- ✅ Added traditional content extraction (Step 1)
- ✅ Enhanced AI-powered analysis (Step 2)
- ✅ Improved content organization and categorization
- ✅ Better error handling and fallback mechanisms
- ✅ More efficient token usage and processing

### v2.0
- ✅ Fixed database model issues
- ✅ Enhanced Groq integration with better prompts
- ✅ Added multi-strategy content parsing
- ✅ Improved error handling and fallback mechanisms
- ✅ Added intelligent content classification
- ✅ Enhanced text extraction algorithms 