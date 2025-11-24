import streamlit as st
import os
import datetime
from pypdf import PdfReader 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools import DuckDuckGoSearchRun

# ---- Page config ----
st.set_page_config(page_title="Pamiya-GPT v2", page_icon="📂")
st.title("📂 Pamiya-GPT: The Researcher")

# ---- API key ----
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# ---- FILE UPLOADER ----
st.sidebar.header("Knowledge Base")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

# Save the uploaded file to disk so the tool can read it
if uploaded_file:
    with open("temp_doc.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success("PDF Uploaded & Saved!")

# ---- PDF READER ----
@tool
def read_current_pdf() -> str:
    """Reads the content of the currently uploaded PDF file."""
    # Check if file exists
    if not os.path.exists("temp_doc.pdf"):
        return "Tell the user: No PDF has been uploaded yet."
    
    try:
        reader = PdfReader("temp_doc.pdf")
        text = ""
        # Read every page
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

@tool
def web_search(query: str) -> str:
    """Search the web via DuckDuckGo."""
    search = DuckDuckGoSearchRun()
    return search.invoke(query)

@tool
def get_current_time() -> str:
    """Return current date & time."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

# Combine all tools
tools = [read_current_pdf, web_search, get_current_time]

# ---- AGENT SETUP ----
@st.cache_resource
def get_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Pamiya-GPT. If the user asks about the 'document' or 'PDF', use the 'read_current_pdf' tool."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor = get_agent()

# ---- CHAT INTERFACE ----
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_input := st.chat_input("Ask about your PDF..."):
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing document..."):
            try:
                response = agent_executor.invoke({"input": user_input})
                final_text = response["output"]
                st.write(final_text)
                st.session_state.messages.append({"role": "assistant", "content": final_text})
            except Exception as e:
                st.error(f"Error: {e}")
