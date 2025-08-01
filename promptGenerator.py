from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from google.colab import userdata

os.environ["GOOGLE_API_KEY"] = userdata.get("GeminiKey")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-pro",
    temperature=0.7,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

loader = WebBaseLoader("https://apple.com")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
vectorstore = FAISS.from_documents(docs, embedding=embedding)
retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are PromptBot, an expert at writing optimized and effective prompts 
        for AI tools like ChatGPT, DALL·E, Codex, Midjourney, etc. 
        Based on the user’s goal or question, generate a powerful, clear, and actionable prompt.
        If the user provides vague input, ask clarifying questions first.
        Keep your responses concise and tailored."""
    ),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template(
        "User's Goal or Input: {query}\n\nContext: {context}"
    )
])

prompt_chain = ({
    "context": retriever | format_docs,
    "query": lambda x: x["query"]
} | prompt | llm)

chat_map = {}

def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chat_map:
        chat_map[session_id] = InMemoryChatMessageHistory()
    return chat_map[session_id]

pipeline_with_history = RunnableWithMessageHistory(
    prompt_chain,
    get_session_history=get_chat_history,
    input_messages_key="query",
    history_messages_key="history"
)

response = pipeline_with_history.invoke(
    {"query": "I want a prompt to generate a logo for a tech startup using DALL·E"},
    config={"session_id": "user123"}
)
print(response.content)
