import streamlit as st
import os
import pandas as pd  
from pypdf import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools import DuckDuckGoSearchRun

st.set_page_config(page_title="Pamiya-GPT v3", page_icon="📊")
st.title("📊 Pamiya-GPT: The Analyst")

if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# ---- FILE UPLOADER ----
st.sidebar.header("Knowledge Base")

# 1. PDF Uploader
pdf_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
if pdf_file:
    with open("temp_doc.pdf", "wb") as f:
        f.write(pdf_file.getbuffer())
    st.sidebar.success("PDF Saved!")

# 2. CSV Uploader 
csv_file = st.sidebar.file_uploader("Upload CSV", type="csv")
if csv_file:
    with open("temp_data.csv", "wb") as f:
        f.write(csv_file.getbuffer())
    st.sidebar.success("CSV Saved!")

# ---- TOOLS ----

@tool
def read_current_pdf() -> str:
    """Reads the uploaded PDF file."""
    if not os.path.exists("temp_doc.pdf"):
        return "No PDF found."
    try:
        reader = PdfReader("temp_doc.pdf")
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error: {e}"

@tool
def analyze_csv_data() -> str:
    """Reads the uploaded CSV file and returns the data as a table."""
    if not os.path.exists("temp_data.csv"):
        return "No CSV file found."
    try:
        # Load data with Pandas
        df = pd.read_csv("temp_data.csv")
        
        # If the file is HUGE, we only read the first 100 rows to save space
        if len(df) > 100:
            return f"Data (First 100 rows only):\n{df.head(100).to_markdown()}"
        
        return f"Full Data:\n{df.to_markdown()}"
    except Exception as e:
        return f"Error reading CSV: {e}"

@tool
def web_search(query: str) -> str:
    """Search the web."""
    search = DuckDuckGoSearchRun()
    return search.invoke(query)

tools = [read_current_pdf, analyze_csv_data, web_search]

# ---- AGENT ----
@st.cache_resource
def get_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Pamiya-GPT. Use 'read_current_pdf' for documents and 'analyze_csv_data' for data/tables."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor = get_agent()

# ---- CHAT UI ----
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_input := st.chat_input("Ask about your Data..."):
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                response = agent_executor.invoke({"input": user_input})
                st.write(response["output"])
                st.session_state.messages.append({"role": "assistant", "content": response["output"]})
            except Exception as e:
                st.error(f"Error: {e}")
