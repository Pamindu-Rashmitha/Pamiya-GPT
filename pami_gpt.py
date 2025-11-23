import os
import datetime
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import DuckDuckGoSearchRun

# ---- Page config ----
st.set_page_config(page_title="Pamiya-GPT", page_icon="🤖")
st.title("🤖 Pamiya-GPT")

# ---- API key from Streamlit secrets  ----
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("GOOGLE_API_KEY missing in Streamlit secrets. Add it via the app settings.")
    st.stop()
else:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# ---- Helper to convert model output into plain text ----
def to_text(output) -> str:
    """
    Convert Gemini/LangChain output into readable text.
    Handles lists of {"type":"text","text":"..."} or plain strings.
    """
    if isinstance(output, list):
        parts = []
        for item in output:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(output)

# ---- Tools ----
@tool
def web_search(query: str) -> str:
    """Search the web via DuckDuckGo and return a short summary."""
    search = DuckDuckGoSearchRun()
    # DuckDuckGoSearchRun supports .invoke(...) in the pinned versions
    result = search.invoke(query)
    # result may already be text; return as string
    return str(result)

@tool
def get_current_time() -> str:
    """Return current date & time (useful for responses)."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

@tool
def multiply(a: int, b: int) -> int:
    return a * b

@tool
def get_length_of_word(word: str) -> int:
    return len(word)

tools = [web_search, get_current_time, multiply, get_length_of_word]

# ---- Create / cache the agent ----
@st.cache_resource
def get_agent():
    # Initialize the Gemini LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    
    prompt = ChatPromptTemplate.from_messages([
        ("system","""You are Pamiya-GPT.You MUST ALWAYS introduce yourself as "Pamiya-GPT" when asked who you are.Do NOT say you are "a large language model", "Gemini", or "trained by Google".If the user asks who you are, respond exactly: "I am Pamiya-GPT, your helpful assistant."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
    )
    return agent

agent_executor = get_agent()

# ---- Session state for chat messages ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---- Input area ----
if user_input := st.chat_input("Ask Pamiya-GPT (Try 'What is the Google stock price?' or 'What time is it?')"):
    # show user message
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Call agent and render assistant message
    with st.chat_message("assistant"):
        with st.spinner("Hold up, let Pamiya-GPT cook..."):
            try:
                response = agent_executor.invoke({"input": user_input})
                output = response.get("output") if isinstance(response, dict) else response

                # convert to plain text
                response_text = to_text(output)

                # show and save
                st.write(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

            except Exception as e:
                st.error(f"An error occurred: {e}")
