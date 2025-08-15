# Fetched tools and agents
# (D:\Udemy\Complete_GenAI_Langchain_Huggingface\Python\venv) 
# D:\Udemy\Complete_GenAI_Langchain_Huggingface\Python\32-Search Engine with Langchain Tools and Agents>streamlit run 4-app.py

import streamlit as st  # type: ignore
from langchain_groq import ChatGroq  # type: ignore
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper  # type: ignore
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun  # type: ignore
from langchain import hub  # type: ignore
from langchain.agents import create_openai_tools_agent, AgentExecutor  # type: ignore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # type:ignore
import os

# Streamlit configuration
st.set_page_config(page_title="LangChain Search Chat", page_icon="🔎")
st.title("🔎 LangChain - Chat with Search")

# Sidebar for API Key input
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Initialize LLM only if API key is provided
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="openai/gpt-oss-120b")
else:
    st.warning("⚠️ Please enter your GROQ API Key to continue.")
    st.stop()  # Prevent further execution until key is provided

# Initialize tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
search_tool = DuckDuckGoSearchRun(name="Search")
tools = [search_tool, arxiv_tool, wiki_tool]

# Load prompt from ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant for question-answering tasks.Use llm model, tools to answer the question precisely."
    "If you don't know the answer, say that you don't know."),MessagesPlaceholder(variable_name="agent_scratchpad"),("user", "{input}")])

# Create agent
agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user input
user_input = st.chat_input(placeholder="Ask anything...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    #response = agent_executor.invoke({"input": user_input})
    #st.session_state.messages.append({'role':'assistant',"content":response})
    #st.write(response)
    # Run agent and get response
    try:
        response = agent_executor.invoke({"input": user_input})
        final_response = response.get("output", "Sorry, I couldn't find an answer.")
    except Exception as e:
        final_response = f"❌ Error: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": final_response})
    st.chat_message("assistant").write(final_response)