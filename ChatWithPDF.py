# 📦 Standard Libraries
import os
from dotenv import load_dotenv
# 📄 PDF Handling
from PyPDF2 import PdfReader

# 🔍 LangChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# 🖥️ UI
import streamlit as st

# 🔐 Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 📚 PDF Text Loading & Chunking
def load_pdf(file_path):
    raw_text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(raw_text)

# 🚀 Streamlit App
def main():
    st.set_page_config(page_title="Chat with PDF (Gemini + Chroma)", layout="wide")
    st.title("📄🤖 Chat with your PDF using Google Gemini")

    pdf_file = st.file_uploader("📤 Upload your PDF", type="pdf")

    if pdf_file:
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.getbuffer())

        st.info("⏳ Reading and chunking the PDF...")
        chunks = load_pdf("temp.pdf")
        st.success(f"✅ Loaded and split into {len(chunks)} chunks.")

        # 🧠 Set up embeddings and Chroma vector store
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )

        vectorstore = Chroma.from_texts(chunks, embedding=embeddings, persist_directory="chroma_store")

        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # 💬 Set up Google Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3
        )

        # 🔗 Combine retriever + Gemini for answering
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )

        # Ask questions
        st.subheader("💬 Ask Questions")
        question = st.text_input("Ask anything about your PDF:")

        if question:
            with st.spinner("Thinking with Gemini..."):
                response = qa_chain.run(question)
            st.markdown("### 📬 Answer:")
            st.write(response)

if __name__ == "__main__":
    main()
