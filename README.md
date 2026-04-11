# Conversational RAG Chatbot

A simple conversational chatbot built with Streamlit, LangChain, and Groq.

## Features

- Clean Streamlit chat UI
- Conversation memory during session
- Fast LLM responses using Groq (`llama-3.1-8b-instant`)
- Environment-based API key loading via `.env`

## Tech Stack

- Python
- Streamlit
- LangChain
- langchain-groq
- python-dotenv

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- README.md
|-- LICENSE
`-- .gitignore
```

## Clone This Repository

```bash
git clone https://github.com/Zeeshan5932/Conversational-RAG-Chatbot.git
cd Conversational-RAG-Chatbot
```

## Setup and Installation

1. Create virtual environment:

```bash
python -m venv .venv
```

2. Activate virtual environment:

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Important:
- Never commit `.env` to GitHub.
- Regenerate keys immediately if exposed.

## Run the App

```bash
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## Usage

1. Type your query in input box.
2. Click `Submit`.
3. Read AI response in app.

## Git Workflow (Quick)

```bash
git add .
git commit -m "Update project"
git push origin main
```

## License

This project is licensed under MIT. See `LICENSE` for details.

## Maintainer

- GitHub: https://github.com/Zeeshan5932

