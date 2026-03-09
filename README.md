# VC Tutor — Venture Capital Funding Chatbot

A Streamlit chatbot for the **Venture Capital Funding** course (Kozminski University, Konrad Sowa). Students can switch between **Didactic (explicit instruction)** and **Socratic (question-led)** teaching modes.

## Tech stack

- **Frontend:** Python, Streamlit  
- **Backend:** OpenAI API

## Features

- Chat interface with streaming AI responses  
- Sidebar toggle: **Didactic** vs **Socratic** teaching mode  
- Session memory (conversation history for the session)  
- Dynamic system prompts that change when you switch modes  
- **Clear Chat** button to reset the conversation (recommended when switching modes)

## API key safety

- **Never commit your OpenAI API key.** The repo uses a `.env` file that is listed in `.gitignore` and is not pushed to GitHub.  
- Only `.env.example` (with a placeholder) is committed. Copy it to `.env` and add your own key locally.

## Setup and run

1. **Clone the repo** (or download the code).

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Or use a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set your OpenAI API key:**
   - Copy `.env.example` to `.env`
   - Edit `.env` and replace `sk-your-key-here` with your key from [OpenAI API keys](https://platform.openai.com/api-keys)

4. **Run the app:**
   ```bash
   streamlit run streamlit_app.py
   ```
   Open the URL shown (usually http://localhost:8501).

## Deployment on Streamlit Cloud

1. **Connect your GitHub repo** to [Streamlit Cloud](https://share.streamlit.io).
2. **Set the OpenAI API key as a secret:**
   - In your Streamlit Cloud app settings, go to "Secrets".
   - Add: `OPENAI_API_KEY = "sk-your-actual-key-here"`
3. **Deploy** the app. It will automatically use the secret for the API key.

## License

Use as needed for the course. Ensure your use of the OpenAI API complies with OpenAI’s terms of use.
