from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from flask import Flask, request, jsonify
import os
import pinecone
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load documents from directory
def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

# Split documents into smaller chunks
def split_docs(documents, chunk_size=5000, chunk_overlap=2000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

# Initialize embeddings, vectorstore, and LangChain LLM
def setup_pinecone_and_embeddings(docs):
    embeddings = OpenAIEmbeddings()
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT")
    )
    index_name = "fastcode"
    index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    return index

# Flask app setup
app = Flask(__name__)

# Load documents and initialize components
directory = 'data'
documents = load_docs(directory)
docs = split_docs(documents)
index = setup_pinecone_and_embeddings(docs)

# Initialize LangChain LLM, memory, and agents
model_name = "gpt-4"
llm = OpenAI(model_name=model_name)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
tools = [
    Tool(
        name="Document Search",
        func=lambda q: index.similarity_search(q, k=5),
        description="Use this tool to search for relevant documents related to the query."
    )
]
agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory)

@app.route('/')
def main():
    # Fetch the question from the request
    query = request.args.get('question')
    if not query:
        return jsonify({"error": "Missing 'question' parameter"}), 400

    try:
        # Use the agent to answer the query
        response = agent.run(query)
        return jsonify({"response": response})
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        return jsonify({"error": "An error occurred while processing your query."}), 500

if __name__ == '__main__':
    app.run(port=8080)

