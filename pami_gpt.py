import streamlit as st
import os
import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools import DuckDuckGoSearchRun

# setup & configuration
st.set_page_config(page_title="Pamiya-GPT", page_icon="🤖")
st.title("🤖 Pamiya-GPT")

# api-key
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# define tools

# Search
@tool
def web_search(query: str) -> str:
    """Searches the internet for current events, news, or facts."""
    search = DuckDuckGoSearchRun()
    return search.invoke(query)

# Time
@tool
def get_current_time() -> str:
    """Returns the current date and time."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

# Math
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers together."""
    return a * b

# Utilities
@tool
def get_length_of_word(word: str) -> int:
    """Returns the number of characters in a word."""
    return len(word)

# Combine all tools into a list
tools = [web_search, get_current_time, multiply, get_length_of_word]

# initialize agent
@st.cache_resource
def get_agent():

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    # The System Prompt gives the bot its personality
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Pamiya-GPT. You are a resourceful assistant. Always use your 'web_search' tool if the user asks about recent events or facts you don't know."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor = get_agent()

# session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat ui
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# handle input
if user_input := st.chat_input("Ask Pamiya-GPT (Try 'What is the Google stock price?' or 'What time is it?')"):
    
    # Display User Message
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Hold up, let Pamiya-GPT cook..."):
            try:
                response_dict = agent_executor.invoke({"input": user_input})
                response_text = response_dict["output"]

                # Sometimes Gemini returns a list like [{"text": "..."}]
                if isinstance(response_text, list) and len(response_text) > 0 and "text" in response_text[0]:
                    final_text = response_text[0]["text"]
                else:
                    # Sometimes it returns a normal string
                    final_text = str(response_text)
                
                st.write(final_text)

                # Save Response
                st.session_state.messages.append({"role": "assistant", "content": final_text})
            except Exception as e:
                st.error(f"An error occurred: {e}")