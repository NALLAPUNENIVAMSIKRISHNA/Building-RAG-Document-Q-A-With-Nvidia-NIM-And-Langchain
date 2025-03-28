# NVIDIA NIM Document Query System

Demo Video Link :- https://drive.google.com/file/d/1__gTOEOkLIaEMAshYkE0eAnvgoq0QAg6/view?usp=sharing

## Overview
This project is a **Document Query System** using **NVIDIA NIM AI Inference**, **LangChain**, and **FAISS Vector Store**. The system allows users to upload a directory of PDFs, generate vector embeddings, and perform **semantic search** to retrieve the most relevant information based on user queries.

## Features
- **Loads and processes multiple PDFs from a directory**
- **Generates vector embeddings using NVIDIA NIM**
- **Stores embeddings in FAISS for fast retrieval**
- **Processes user queries and retrieves relevant document chunks**
- **Displays retrieved answers and similar document sections**

## Technologies Used
- **Python**
- **Streamlit** (UI framework)
- **NVIDIA NIM AI Endpoints** (for embedding and inference)
- **LangChain** (for document processing and retrieval)
- **FAISS** (for vector storage and similarity search)
- **PyPDFLoader** (for loading PDF documents)

## Installation

### 1. Create a Virtual Environment (Optional but Recommended)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a **.env** file in the project root and add the following:
```
NVIDIA_API_KEY=your_nvidia_api_key_here
```

## Usage
### 1. Run the Application
```bash
streamlit run finalapp.py
```

### 2. Steps to Use
1. **Upload PDFs**: Place your PDFs inside the `./us_census/` directory.
2. **Click 'Document Embedding'**: This will generate vector embeddings.
3. **Ask a Question**: Enter your query in the text input box.
4. **Get Answers**: The system retrieves the most relevant document sections and displays them.

## Project Structure
```
Nvidia-NIM-Document-Query/
│── us_census/               # Directory containing PDF documents
│── finalapp.py              # Main application script
│── requirements.txt         # Dependencies
│── .env                     # NVIDIA API Key (not included in the repository)
│── README.md                # Project documentation
```

## Notes
- Ensure you have a valid **NVIDIA API Key** to use the AI inference features.
- Make sure your PDFs are inside the **us_census/** directory before running the application.
