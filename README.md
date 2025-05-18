# Document Analysis Agent

An intelligent document analysis tool that allows users to upload multiple PDF documents and ask questions about their content using AI.

## Features

- Multiple PDF document upload support
- AI-powered document analysis
- Context-aware question answering
- Source references with page numbers
- Interactive web interface

## Requirements

- Python 3.8+
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sampathpulukurthi/document_agent.git
cd document_agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Run the application:
```bash
python app.py
```

2. Open your browser and go to http://localhost:7868

3. Upload PDF documents using the interface

4. Click "Process Document" to analyze the documents

5. Select documents from the dropdown and ask questions

6. View answers with source references and page numbers

## Technologies Used

- LangChain
- OpenAI GPT-4
- FAISS Vector Store
- Gradio UI
- PyPDF2
