# 🧠 Pamiya-GPT: The Smart AI Agent

Pamiya-GPT is a sophisticated AI assistant capable of real-time web search, data analysis, document reading, and email automation. Built using **Google's Gemini 2.5 Flash**, **LangChain**, and **Streamlit**.

Unlike standard chatbots that rely solely on training data, Pamiya-GPT is an **Agent** equipped with specialized tools. It intelligently decides when to browse the internet, analyze a spreadsheet, or read a PDF document to answer your questions.

## ✨ Features

* **🧠 Advanced Reasoning:** Powered by Google's fast and efficient `gemini-2.5-flash` model.
* **🌐 Internet Access:** Uses **DuckDuckGo** to search the live web for current events and news.
* **📄 Document Reader:** Can read and summarize uploaded **PDF** files.
* **📊 Data Analyst:** Can analyze uploaded **CSV** files and create data tables using Pandas.
* **📧 Email Automation:** Can draft and send real emails via Gmail (SMTP).
* **💾 Memory:** Remembers context and chat history, allowing for follow-up questions.
* **🎨 Clean UI:** A professional interface built with Streamlit, featuring a sidebar for file management.

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Frontend:** [Streamlit](https://streamlit.io/)
* **AI Model:** Google Gemini API (`google-generativeai`)
* **Orchestration:** [LangChain](https://www.langchain.com/) (Agents & Tool Calling)
* **Tools:** `pandas`, `pypdf`, `duckduckgo-search`, `smtplib`

## 🚀 Installation & Setup

Follow these steps to run Pamiya-GPT locally.

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/pamiya-gpt.git](https://github.com/yourusername/pamiya-gpt.git)
cd pamiya-gpt
