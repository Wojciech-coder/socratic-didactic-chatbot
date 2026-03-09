"""
VC Tutor Chatbot — Venture Capital Funding course assistant (Kozminski University).

How to run:
  1. Install dependencies: pip install -r requirements.txt
  2. Set your OpenAI API key:
     - Create a .env file in this directory with: OPENAI_API_KEY=sk-your-key-here
     - Or use Streamlit secrets for deployment
  3. Run the app: streamlit run streamlit_app.py

Uses gpt-4o by default; you can change MODEL to "gpt-3.5-turbo" in the code if preferred.
"""
import streamlit as st
import os
import json
import time
import datetime
from pathlib import Path
import tiktoken
from openai import APIConnectionError, APIError, OpenAI

# Load environment variables (set OPENAI_API_KEY in .env or Streamlit secrets)
from dotenv import load_dotenv
load_dotenv()

# Configure OpenAI API — use Streamlit secrets for deployment, fallback to .env for local
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not api_key:
    st.error("OpenAI API key not found. Please set OPENAI_API_KEY in Streamlit secrets (for deployment) or in a .env file (for local development).")
    st.stop()
client = OpenAI(api_key=api_key)

# Constants / Configuration
MODEL = "gpt-4o" 
MAX_TOKENS = 64000  # Maximum tokens to send to the API
temperature_setting = 0.5

# Didactic (Explicit Instruction) system prompt for VC course
DIDACTIC_PROMPT = """Role and Context You are an expert AI teaching assistant for the "Venture Capital Funding" course at Kozminski University, taught by Konrad Sowa. Your primary goal is to help students directly and efficiently learn the venture capital market, the fundraising process, startup valuation, and investor-founder negotiations.
Pedagogical Directive: The Didactic Approach (Explicit Instruction) You must strictly adhere to an explicit, didactic teaching framework based on Cognitive Load Theory. Because students are novices learning highly structured concepts, your goal is to prevent working memory overload by providing clear, step-by-step guidance. Follow these behavioral rules:
Explain the Rule/Idea Succinctly: When a student asks a question or introduces a topic, provide a direct, clear, and unambiguous explanation immediately. Do not answer questions with more questions.
Demonstrate with a Worked Example: Follow your explanation with a concrete, step-by-step example drawing directly from the course materials (e.g., cap table math, VC fund mechanics).
Give Guided Practice: After demonstrating, provide a similar but new problem or scenario and ask the student to solve it or apply the rule ("Now you try the next one").
Correct and Fade Support: Provide immediate, clear feedback on their attempt. Correct any mistakes directly, explain why it was wrong, and gradually reduce your scaffolding as they demonstrate competence.
Core Course Knowledge Base Draw your explanations, worked examples, and practice problems exclusively from the following course themes:
The Fundraising Process: Explicitly teach the 9 to 13-month (37-51 week) fundraising cycle, mapping out the steps from ideation and preparation (weeks 1-7) to due diligence, deal negotiation, and money transfer (weeks 28-51).
VC Structure & Economics: Teach the fund structure directly: Limited Partners (LPs) provide the capital, General Partners (GPs) manage the fund. Explain the standard economics: a 2% Management Fee and a 20% Carried Interest (Carry) on profits. Explain that VCs rely on "home runs" (85% of returns coming from 10% of investments; an 8x multiplier).
Valuation & Dilution (Cap Tables): Directly explain the formula: Pre-money Valuation + Investment Amount = Post-money Valuation. Show how equity percentages are calculated (Investment / Post-money Valuation = Investor %). Use the course's cap table progression (e.g., starting at 50/50 founder split, moving to 48/48 after Uncle's 5% FFF round, etc.) as your worked examples.
Market Sizing (TAM/SAM/SOM): Explicitly define Total Addressable Market, Serviceable Available Market, and Serviceable Obtainable Market. Use the course examples (e.g., OLED market or lipstick market) to demonstrate how to calculate them.
Evaluation Frameworks: Directly teach the OUTSIDE-IMPACTS framework (Opportunity, Uncertainties, Team, Strategy, Investment, Deal, Exit, etc.).
Deal Negotiation & Term Sheets: Teach students exactly what to look for. Explain that term sheets precede investment agreements, and explicitly outline the dangers of "bad investors" (dementors) and the mechanics of a Lead Investor dictating terms (as seen in the CVC case study).
Example Interaction Guidelines:
Student: "How does startup valuation work in a funding round?"
Tutor: "In venture capital, valuation is based on a simple formula: Pre-money Valuation + Investment Amount = Post-money Valuation. The investor's equity percentage is then calculated by dividing their Investment Amount by the Post-money Valuation. Example: If a VC invests €2,000,000 at a €6,000,000 Pre-money valuation, the Post-money valuation becomes €8,000,000. The VC's ownership is €2M / €8M, which equals 25%. Now you try: A startup agrees to a Pre-money valuation of €10,000,000. A VC invests €2,500,000. What is the Post-money valuation, and what percentage of the company will the VC own? I'll check your steps."
Student: "What's the difference between a GP and an LP?"
Tutor: "General Partners (GPs) are the fund managers who make the investment decisions. Limited Partners (LPs) are the investors (like pension funds or family offices) who provide the actual money. Example: In a €30M fund, the LPs might provide 99% of the capital. The GPs take a 2% 'Management Fee' annually to run the fund and keep 20% of the net profits, known as 'Carried Interest' or 'Carry', when the investments are sold. Now you try: If a €50M fund generates €100M in net profit, how much money goes to the GPs through their 20% Carried Interest?"
Tone and Persona Be structured, clear, and highly instructive. Act as a subject-matter expert transferring foundational competence to a novice. Use step-by-step logic, bullet points, and bold text for key formulas and rules. Provide immediate validation or correction."""

# Socratic (Question-led) system prompt for VC course
SOCRATIC_PROMPT = """Role and Context You are an expert AI teaching assistant for the "Venture Capital Funding" course at Kozminski University, taught by Konrad Sowa. Your primary goal is to help students understand the venture capital market, the fundraising process, startup valuation, and investor-founder negotiations.
Pedagogical Directive: The Socratic Approach You must strictly adhere to a Constructivist/dialogic learning framework. Knowledge is actively built by the learner, not passively received. Do not act as an encyclopedia. Act as a Socratic tutor.
Your core mechanisms are retrieval practice, self-explanation, conceptual change, and metacognition. Follow these behavioral rules:
Never Give the Direct Answer First: When a student asks a question, respond with a targeted, thought-provoking question that leads them toward the answer (e.g., "Before I explain that, what do you think would happen to the founder's equity if the valuation drops during a Series A round?")
Prompt Self-Explanation: Frequently ask the student to synthesize concepts in their own words (e.g., "Can you explain the difference between a General Partner (GP) and a Limited Partner (LP) in your own words?")
Challenge Inconsistencies: Listen carefully to the student's logic. If they contradict themselves, point it out gently (e.g., "That seems to conflict with what you said earlier about the founder maintaining control. If they give up 15% in the Seed round and 29% in Series A, how can both be true?")
Scaffold and Hint (Minimal Information): Withhold information until the learner is genuinely stuck. When they are stuck, provide a small hint or a real-world analogy rather than a full explanation, keeping them in their "just-right challenge" zone.
Core Course Knowledge Base Draw your scenarios, hints, and core concepts exclusively from the following course themes:
The Fundraising Process: Guide students through the 9 to 13-month (37-51 week) fundraising cycle, from ideation and pitching to due diligence and money transfer.
VC Structure & Mechanics: Test their understanding of GP vs. LP dynamics, Management Fees (e.g., 2% of fund size), Carried Interest (e.g., 20% of net profit), and why VCs rely on "home runs" (85% of returns coming from 10% of investments).
Valuation & Dilution: Use the cap table progression to ask questions about dilution. E.g., How does a founder go from 50% equity at the pre-investment stage to 9% by Series F? Challenge them on pre-money vs. post-money valuation calculations.
Evaluation Frameworks: Elicit concepts from the OUTSIDE-IMPACTS framework (Opportunity, Uncertainties, Team, Strategy, Investment, Deal, Exit) and the concept of TAM, SAM, and SOM.
The Pitch Deck: Ask students what makes a company "investable" versus "non-investable" (e.g., clear cap table, IP sorted out, no debt to public authorities) and what investors look for (Product-market fit, Growth, Team, Business model).
Deal Negotiation (CVC Case Study): Use the Corporate Venture Capital (CVC) case study as a role-play scenario. Challenge students on how they would handle a lead investor ("Fund ABC") who suddenly changes the term sheet to demand total control of the startup, pushing the corporate fund out.
Example Interaction Guidelines:
Student: "What is a term sheet?"
Tutor: "A term sheet is a crucial document in the fundraising cycle, usually proposed around week 15-20 of the process. But before we dive into its exact contents, what role do you think this document plays before lawyers draft the final investment agreement?"
Student: "How much equity should I give up in a seed round?"
Tutor: "That's a great question, and it's all about 'splitting the pie.' Let's look at a standard scenario. If a VC invests €2,000,000, what variables do we need to know to calculate the percentage they get?"
Tone and Persona Be encouraging, intellectually stimulating, and academically rigorous. Never patronize the student. Use phrases like "You're on the right track, but consider..." or "Let's unpack that further." """
# App title
st.set_page_config(page_title="VC Tutor — Venture Capital Funding")

# Hide all Streamlit UI elements
hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Target the deploy button with its specific class */
        .stAppDeployButton {display: none !important;}
        div[data-testid="stAppDeployButton"] {display: none !important;}
        button[data-testid="stBaseButton-header"] {display: none !important;}
        
        /* Hide the entire toolbar if needed */
        div[data-testid="stToolbar"] {visibility: hidden;}
        .stAppToolbar {visibility: hidden;}
        </style>
        """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# Function to count tokens
def count_tokens(text):
    """Count the number of tokens in a text string"""
    encoding = tiktoken.encoding_for_model(MODEL)
    return len(encoding.encode(text))

def count_message_tokens(message):
    """Count tokens in a message"""
    encoding = tiktoken.encoding_for_model(MODEL)
    num_tokens = 4  # Approximate tokens for role
    if "content" in message and message["content"]:
        num_tokens += len(encoding.encode(message["content"]))
    return num_tokens

def count_messages_tokens(messages):
    """Count the total number of tokens in the messages"""
    encoding = tiktoken.encoding_for_model(MODEL)
    num_tokens = 0
    for message in messages:
        num_tokens += count_message_tokens(message)
    # Add a few tokens for the message format
    num_tokens += 3  # End of sequence tokens
    return num_tokens

# Function to save conversation to JSON file (optional logging; uses teaching_mode in metadata)
def save_conversation(messages, conversation_id=None, teaching_mode=None):
    log_dir = Path("conversation_logs")
    log_dir.mkdir(exist_ok=True)
    if conversation_id is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        conversation_id = f"conversation_{timestamp}.json"
    total_tokens = count_messages_tokens([{"role": m["role"], "content": m["content"]} for m in messages])
    conversation_data = {
        "metadata": {
            "teaching_mode": teaching_mode,
            "total_messages": len(messages),
            "total_tokens": total_tokens,
            "model": MODEL,
            "last_updated": datetime.datetime.now().isoformat()
        },
        "messages": messages
    }
    
    # Create full file path
    filename = log_dir / conversation_id
    
    # Write to file
    with open(filename, 'w') as f:
        json.dump(conversation_data, f, indent=2)
    
    return conversation_id

# Function for generating LLM response (system prompt is dynamic based on teaching_mode)
def generate_response(messages, teaching_mode):
    api_messages = []
    system_prompt = DIDACTIC_PROMPT if teaching_mode == "didactic" else SOCRATIC_PROMPT
    api_messages.append({"role": "system", "content": system_prompt})
    
    # Add conversation history (excluding timestamps and token counts)
    for msg in messages:
        if msg["role"] in ["user", "assistant"]:
            api_messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Check token count and truncate if necessary
    while count_messages_tokens(api_messages) > MAX_TOKENS:
        # Remove the oldest user/assistant message (after the system message)
        if len(api_messages) > 2:  # Keep at least the system message and the latest user message
            api_messages.pop(1)  # Remove the second message (first after system)
        else:
            # If we can't reduce further, truncate the latest message
            content = api_messages[-1]["content"]
            api_messages[-1]["content"] = content[:len(content)//2]
    
    # Create Chat Completion with stream=True for the typing effect
    stream = client.chat.completions.create(
        model=MODEL,
        messages=api_messages,
        temperature=temperature_setting,
        stream=True  # Enable streaming
    )
    
    return stream

# Session state: teaching mode (didactic / socratic) and chat messages
if "teaching_mode" not in st.session_state:
    st.session_state.teaching_mode = "didactic"
if "conversation_id" not in st.session_state:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.conversation_id = f"conversation_{timestamp}.json"
if "messages" not in st.session_state:
    initial_message = {
        "role": "assistant",
        "content": "What would you like to explore today? Ask me about venture capital, fundraising, valuation, or term sheets.",
        "timestamp_start": datetime.datetime.now().isoformat(),
        "timestamp_end": datetime.datetime.now().isoformat(),
        "tokens": 0,
    }
    st.session_state.messages = [initial_message]
    save_conversation(
        st.session_state.messages,
        st.session_state.conversation_id,
        st.session_state.teaching_mode,
    )

# Sidebar: pedagogical mode and Clear Chat
with st.sidebar:
    st.subheader("Teaching mode")
    mode_label = st.selectbox(
        "Pedagogical approach",
        options=["Didactic (Explicit Instruction)", "Socratic (Question-led)"],
        index=0 if st.session_state.teaching_mode == "didactic" else 1,
        key="teaching_mode_select",
    )
    st.session_state.teaching_mode = "didactic" if "Didactic" in mode_label else "socratic"
    st.caption("Switch modes anytime. Use **Clear Chat** when changing mode to start a fresh conversation.")
    if st.button("Clear Chat", type="primary"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "What would you like to explore today? Ask me about venture capital, fundraising, valuation, or term sheets.",
                "timestamp_start": datetime.datetime.now().isoformat(),
                "timestamp_end": datetime.datetime.now().isoformat(),
                "tokens": 0,
            }
        ]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.conversation_id = f"conversation_{timestamp}.json"
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if prompt := st.chat_input():
    timestamp_start = datetime.datetime.now().isoformat()
    token_count = count_tokens(prompt)
    user_message = {
        "role": "user",
        "content": prompt,
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_start,
        "tokens": token_count,
    }
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.write(prompt)
    save_conversation(
        st.session_state.messages,
        st.session_state.conversation_id,
        st.session_state.teaching_mode,
    )

# Generate streaming response when last message is from user
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        timestamp_start = datetime.datetime.now().isoformat()
        token_count = 0
        try:
            for chunk in generate_response(st.session_state.messages, st.session_state.teaching_mode):
                if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content is not None:
                    content_chunk = chunk.choices[0].delta.content
                    full_response += content_chunk
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.01)
            message_placeholder.markdown(full_response)
            timestamp_end = datetime.datetime.now().isoformat()
            token_count = count_tokens(full_response)
        except APIConnectionError:
            err_msg = (
                "**Connection error.** The app could not reach OpenAI's servers. "
                "Check your internet connection, firewall, or VPN. If you're on a corporate network, "
                "it may block access to api.openai.com. Try another network or contact your IT department."
            )
            message_placeholder.markdown(err_msg)
            full_response = err_msg
        except APIError as e:
            err_msg = f"**API error:** {str(e)}. Check your API key in `.env` and that your OpenAI account has access."
            message_placeholder.markdown(err_msg)
            full_response = err_msg
        timestamp_end = datetime.datetime.now().isoformat()
    assistant_message = {
        "role": "assistant",
        "content": full_response,
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
        "tokens": token_count,
    }
    st.session_state.messages.append(assistant_message)
    save_conversation(
        st.session_state.messages,
        st.session_state.conversation_id,
        st.session_state.teaching_mode,
    )
