import streamlit as st
import os
import smtplib
from email.mime.text import MIMEText
import pandas as pd
from pypdf import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Pamiya-GPT v5", page_icon="🧠")
st.title("🧠 Pamiya-GPT: The Smart Agent")

# ---- CONFIG & SECRETS ----
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

EMAIL_ENABLED = False
if "EMAIL_ADDRESS" in st.secrets and "EMAIL_PASSWORD" in st.secrets:
    EMAIL_ENABLED = True
    EMAIL_ADDRESS = st.secrets["EMAIL_ADDRESS"]
    EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]

# ---- SIDEBAR: FILES ----
st.sidebar.header("Knowledge Base")
pdf_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
if pdf_file:
    with open("temp_doc.pdf", "wb") as f:
        f.write(pdf_file.getbuffer())
    st.sidebar.success("PDF Saved!")

csv_file = st.sidebar.file_uploader("Upload CSV", type="csv")
if csv_file:
    with open("temp_data.csv", "wb") as f:
        f.write(csv_file.getbuffer())
    st.sidebar.success("CSV Saved!")

# ---- TOOLS ----
@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Sends an email to the specified recipient."""
    if not EMAIL_ENABLED:
        return f"[SIMULATION MODE] Email to {recipient} NOT sent (No credentials).\nSubject: {subject}\nBody: {body}"
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = recipient
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        return f"Email successfully sent to {recipient}!"
    except Exception as e:
        return f"Failed to send email: {e}"

@tool
def read_current_pdf() -> str:
    """Reads the uploaded PDF."""
    if not os.path.exists("temp_doc.pdf"): return "No PDF found."
    try:
        reader = PdfReader("temp_doc.pdf")
        text = ""
        for page in reader.pages: text += page.extract_text()
        return text
    except Exception as e: return f"Error: {e}"

@tool
def analyze_csv_data() -> str:
    """Reads the uploaded CSV."""
    if not os.path.exists("temp_data.csv"): return "No CSV found."
    try:
        df = pd.read_csv("temp_data.csv")
        if len(df) > 100: return f"Data (First 100 rows):\n{df.head(100).to_markdown()}"
        return f"Full Data:\n{df.to_markdown()}"
    except Exception as e: return f"Error: {e}"

@tool
def web_search(query: str) -> str:
    """Web search."""
    return DuckDuckGoSearchRun().invoke(query)

tools = [send_email, read_current_pdf, analyze_csv_data, web_search]

# ---- AGENT ----
@st.cache_resource
def get_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Pamiya-GPT. You remember previous conversations. Use tools only when needed."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor = get_agent()

# ---- UI ----
if "messages" not in st.session_state: st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.write(msg["content"])

if user_input := st.chat_input("Ask Pamiya-GPT..."):
    with st.chat_message("user"): st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # --- CONVERT HISTORY FOR AGENT ---
    chat_history = []
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # PASS THE HISTORY HERE
                response = agent_executor.invoke({
                    "input": user_input,
                    "chat_history": chat_history
                })
                st.write(response["output"])
                st.session_state.messages.append({"role": "assistant", "content": response["output"]})
            except Exception as e: st.error(f"Error: {e}")
