import streamlit as st
import os
import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools import DuckDuckGoSearchRun

# ---- Page config ----
st.set_page_config(page_title="Pamiya-GPT", page_icon="🤖")
st.title("🤖 Pamiya-GPT")

# ---- API key from Streamlit secrets ----
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("GOOGLE_API_KEY missing in Streamlit secrets. Add it via the app settings.")
    st.stop()
else:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# ---- Tools ----
@tool
def web_search(query: str) -> str:
    """Search the web via DuckDuckGo and return a short summary."""
    search = DuckDuckGoSearchRun()
    return search.invoke(query)

@tool
def get_current_time() -> str:
    """Return current date & time."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers."""
    return a * b

@tool
def get_length_of_word(word: str) -> int:
    """Returns the count of characters in a word."""
    return len(word)

tools = [web_search, get_current_time, multiply, get_length_of_word]

# ---- Create / cache the agent ----
@st.cache_resource
def get_agent():
    # 1. Use the REAL model name (1.5, not 2.5)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    # 2. Define the Prompt (Keeping your strict instructions)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are Pamiya-GPT.
        You MUST ALWAYS introduce yourself as "Pamiya-GPT" when asked who you are.
        Do NOT say you are "a large language model", "Gemini", or "trained by Google".
        If the user asks who you are, respond exactly: "I am Pamiya-GPT, your helpful assistant."
        
        You are a resourceful assistant. Always use your 'web_search' tool if the user asks about recent events or facts you don't know."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # 3. Use the Modern Agent Constructor (Reliable)
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor = get_agent()

# ---- Session state for chat messages ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---- Input area ----
if user_input := st.chat_input("Ask Pamiya-GPT..."):
    # show user message
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Call agent and render assistant message
    with st.chat_message("assistant"):
        with st.spinner("Pamiya-GPT is cooking..."):
            try:
                # Invoke the agent
                response_dict = agent_executor.invoke({"input": user_input})
                response_text = response_dict["output"]
                
                # --- Output Clean Up ---
                # Sometimes Gemini returns a complex list object. We grab the text.
                if isinstance(response_text, list) and len(response_text) > 0 and isinstance(response_text[0], dict) and "text" in response_text[0]:
                    final_text = response_text[0]["text"]
                else:
                    final_text = str(response_text)

                st.write(final_text)
                st.session_state.messages.append({"role": "assistant", "content": final_text})

            except Exception as e:
                st.error(f"An error occurred: {e}")
