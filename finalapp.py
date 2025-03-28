import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Load the NVIDIA API Key
os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')

# Initialize the LLM model (NVIDIA NIM inference)
llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")  # NVIDIA NIM inferencing

def vector_embedding():
    """Function to create vector embeddings from PDF documents."""
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        
        # Load all PDFs from the directory
        st.session_state.loader = DirectoryLoader("./us_census", glob="*.pdf", loader_cls=PyPDFLoader)
        st.session_state.docs = st.session_state.loader.load()
        
        # Split text into chunks for better processing
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        
        # Create FAISS vector store from document embeddings
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Streamlit UI setup
st.title("NVIDIA NIM Demo")

# Define prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    
    <context>
    {context}
    </context>
    
    Question: {input}
    """
)

# User input for querying documents
prompt1 = st.text_input("Enter your Question From Documents")

# Button to start document embedding
if st.button("Document Embedding"):
    vector_embedding()
    st.write("FAISS Vector Store DB Is Ready Using NvidiaEmbedding")

# If user provides a question, process it
if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write("Response time:", time.process_time() - start)
    st.write(response['answer'])
    
    # Display relevant document chunks
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("------------------")