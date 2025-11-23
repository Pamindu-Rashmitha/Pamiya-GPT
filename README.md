# 🤖 Pamiya-GPT: The Smart AI Agent

Pamiya-GPT is a sophisticated AI assistant capable of real-time web search, mathematical reasoning, and context-aware conversation. Built using **Google's Gemini 1.5 Flash**, **LangChain**, and **Streamlit**.

Unlike standard chatbots that rely solely on training data, Pamiya-GPT is an **Agent** equipped with tools. It intelligently decides when to browse the internet for recent news or perform calculations, making it far more capable than a basic LLM wrapper.

## ✨ Features

* **🧠 Advanced Reasoning:** Powered by Google's fast and efficient `gemini-2.5-flash` model.
* **🌐 Internet Access:** Uses **DuckDuckGo** to search the live web for current events, stock prices, and news (overcoming the "knowledge cutoff").
* **🧮 Tool Use:** Can autonomously use Python tools to solve math problems, count characters, or tell the current time.
* **💾 Conversation Memory:** Remembers context and chat history during the session.
* **🎨 Clean UI:** A beautiful, responsive chat interface built with Streamlit.

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Frontend:** [Streamlit](https://streamlit.io/)
* **AI Model:** Google Gemini API (`google-generativeai`)
* **Orchestration:** [LangChain](https://www.langchain.com/) (Agents & Tool Calling)
* **Search:** DuckDuckGo (`duckduckgo-search`)

## 🚀 Installation & Setup

Follow these steps to run Pamiya-GPT locally on your machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/pamiya-gpt.git](https://github.com/yourusername/pamiya-gpt.git)
cd pamiya-gpt
