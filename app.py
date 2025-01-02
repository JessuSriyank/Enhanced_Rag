import os
import time
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
groq_api_key = "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"

# Set up Streamlit
st.set_page_config(
    page_title="Enhanced RAG Application",
    page_icon="📚",
    layout="wide",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f5f5f5;
    }
    .main-header {
        color: #4CAF50;
        font-weight: bold;
        font-size: 30px;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        color: #666;
        text-align: center;
        margin-bottom: 20px;
    }
    .response-box {
        background-color: #e7f4e4;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .question {
        font-size: 22px;
        font-weight: bold;
        color: #333;
        margin-top: 20px;
    }
    .answer {
        font-size: 16px;
        color: #555;
        margin-top: 10px;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #aaa;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main Headers
st.markdown('<div class="main-header">📚 Enhanced RAG Application</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload PDFs, ask questions, and get instant responses!</div>', unsafe_allow_html=True)

# Ensure the temp directory exists
if not os.path.exists("temp"):
    os.makedirs("temp")

# File Upload Section
st.markdown("### Upload Documents")
uploaded_files = st.file_uploader(
    label="Select your PDF files", type="pdf", accept_multiple_files=True, key="file_uploader"
)

# Initialize the language model
try:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
except Exception as e:
    st.error(f"Error initializing ChatGroq model: {e}")
    st.stop()

# Ensure the session state for storing question and responses is initialized
if "question_history" not in st.session_state:
    st.session_state.question_history = []

# Process PDFs and create vector store
if uploaded_files:
    try:
        st.markdown("### Processing Documents... Please wait.")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        all_documents = []

        for uploaded_file in uploaded_files:
            file_path = os.path.join("temp", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(file_path)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            split_documents = text_splitter.split_documents(docs[:20])
            all_documents.extend(split_documents)

        vectors = FAISS.from_documents(all_documents, embeddings)
        retriever = vectors.as_retriever()

        prompt_template = ChatPromptTemplate.from_template(
            """
            Answer the question based on the provided context only.
            Please provide the most accurate response based on the question.
            <context>
            {context}
            <context>
            Question: {input}
            """
        )

        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
    except Exception as e:
        st.error(f"Error processing PDFs: {e}")
        st.stop()

    # Query interaction
    query_col1, query_col2 = st.columns([3, 1])
    with query_col1:
        # Display previous questions and answers
        for idx, (q, a) in enumerate(st.session_state.question_history):
            st.markdown(f"<div class='question'>{q}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='answer'>{a}</div>", unsafe_allow_html=True)
            st.markdown("---")

        # User input field with auto-clear after submission
        with st.form("query_form", clear_on_submit=True):
            user_query = st.text_input("Ask a question:")
            submitted = st.form_submit_button("Submit")

            if submitted and user_query.strip():
                try:
                    start = time.process_time()
                    response = retrieval_chain.invoke({"input": user_query})
                    end_time = time.process_time() - start

                    st.session_state.question_history.append((user_query, response.get('answer', "No response generated.")))

                    st.markdown('<div class="response-box">', unsafe_allow_html=True)
                    st.markdown("### AI Response")
                    st.write(response.get('answer', "No response generated."))
                    st.markdown('</div>', unsafe_allow_html=True)

                    with query_col2:
                        st.write(f"Response Time: {end_time:.2f} seconds")
                except Exception as e:
                    st.error(f"Error during query processing: {e}")
else:
    st.warning("Please upload one or more PDF files to start.")

# Footer
st.markdown('<div class="footer">Developed with ❤️ using Streamlit</div>', unsafe_allow_html=True)
