import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
import os

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Streamlit Page Config ---
st.set_page_config(page_title="📚 PDF Q&A Engine", layout="wide")
st.title("📚 Chat with your PDF")
st.write("Upload a PDF and ask questions. The AI will answer based only on the document.")

# Initialize session state for chat history and vector store
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar: Upload PDF ---
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file and st.session_state.vector_store is None:
    # Save PDF locally
    pdf_path = Path("uploaded.pdf")
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    st.sidebar.success("✅ PDF uploaded successfully")

    # --- Indexing PDF into Qdrant ---
    with st.spinner("🔄 Processing PDF..."):
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=400
        )
        split_docs = text_splitter.split_documents(docs)

        # Embeddings
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

        # Store into Qdrant (assuming Qdrant is running at localhost:6333)
        st.session_state.vector_store = QdrantVectorStore.from_documents(
            documents=split_docs,
            url="http://localhost:6333",
            collection_name="learning_vectors",
            embedding=embedding_model
        )
    st.success("📑 PDF indexed successfully!")

# --- Chat Section ---
if st.session_state.vector_store is not None:
    st.subheader("💬 Ask a question about your PDF")
    user_query = st.text_input("Type your question here:")

    if user_query:
        with st.spinner("🤔 Thinking..."):
            # Perform similarity search
            search_results = st.session_state.vector_store.similarity_search(query=user_query, k=4)

            # Build context from retrieved docs
            context = "\n\n".join([
                f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}"
                for result in search_results
            ])

            SYSTEM_PROMPT = f"""
            You are a helpful AI assistant who answers questions based ONLY on the provided context from a PDF.
            Always mention the page number where the answer is found.
            
            Context:
            {context}
            """

            # Call OpenAI Chat Completion
            chat_completion = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_query},
                ]
            )

            answer = chat_completion.choices[0].message.content

            # Append to chat history
            st.session_state.chat_history.append({"question": user_query, "answer": answer})

# --- Display Chat History ---
if st.session_state.chat_history:
    st.subheader("🗨️ Chat History")
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['question']}")
        st.markdown(f"**🤖:** {chat['answer']}")
