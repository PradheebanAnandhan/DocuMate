import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from PIL import Image
import pandas as pd
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
genai.configure(api_key=GOOGLE_API_KEY)

def add_custom_css():
    st.markdown("""
    <style>
        body { background: linear-gradient(to right, #e3f2fd, #f9fbe7); font-family: 'Arial', sans-serif; color: #333; }
        .stButton button { background-color: #00796b; color: white; font-size: 16px; border-radius: 8px; padding: 10px 24px; transition: background-color 0.3s ease; }
        .stButton button:hover { background-color: #004d40; }
        .stTextInput input { border-radius: 8px; padding: 10px; border: 2px solid #00796b; }
        .stSidebar { background-color: #e8f5e9; padding: 20px; }
        .footer { text-align: center; padding: 10px; position: fixed; left: 0; bottom: 0; width: 100%; background-color: #00796b; color: white; font-size: 14px; }
    </style>
    """, unsafe_allow_html=True)

def add_chatbot_image():
    try:
        st.sidebar.title("DOCUMATE")
        image = Image.open(r"C:\Users\pradh\OneDrive\Pictures\2167071.jpg")
        st.sidebar.image(image, use_column_width=True)
    except FileNotFoundError:
        st.sidebar.warning("Chatbot image not found.")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def extract_pdf_by_page(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    return [page.extract_text() or "" for page in pdf_reader.pages]

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(text_chunks, db_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(db_name)

def get_conversational_chain():
    prompt_template = """
    Your name is DocumateAI
    You are an intelligent assistant designed to provide clear and comprehensive answers to user questions based on the provided context.
    **Context:** {context}
    **Question:** {question}
    **Answer:**
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question, db_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local(db_name, embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    final_output = "".join(response["output_text"])
    st.markdown(f"<p class='typing-effect'>{final_output}</p>", unsafe_allow_html=True)

def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

def plot_word_frequency(text):
    word_freq = Counter(text.split())
    df = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency'])
    return px.bar(df, x='Word', y='Frequency', title='Word Frequency in PDF')

def main():
    st.set_page_config(page_title="DocuMate", page_icon="📚", layout="wide")
    add_custom_css()
    add_chatbot_image()
    st.title("📚 Welcome to DocuMate")
    st.subheader("Upload your PDFs, ask questions, and get instant answers!")
    
    st.sidebar.header("Upload & Process")
    pdf_docs = st.sidebar.file_uploader("Upload your PDF Files", accept_multiple_files=True)

    if pdf_docs:
        if "pdf_pages" not in st.session_state:
            with st.spinner("Processing..."):
                st.session_state.pdf_pages = {}
                for pdf in pdf_docs:
                    pages = extract_pdf_by_page(pdf)
                    st.session_state.pdf_pages[pdf.name] = pages
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, "faiss_index")
                st.session_state.raw_text = raw_text
                st.sidebar.success("PDFs Processed Successfully")
                st.balloons()

    if "pdf_pages" in st.session_state:
        selected_pdf = st.sidebar.selectbox("Choose a PDF", list(st.session_state.pdf_pages.keys()))
        page_count = len(st.session_state.pdf_pages[selected_pdf])

        if page_count > 0:
            selected_page = st.sidebar.slider("Select Page", 1, page_count, 1)
            st.write(f"**Page {selected_page} Content:**")
            st.write(st.session_state.pdf_pages[selected_pdf][selected_page - 1])

        st.plotly_chart(plot_word_frequency(st.session_state.raw_text))
        st.subheader("Word Cloud")
        generate_word_cloud(st.session_state.raw_text)

    st.subheader("Ask a Question from the PDF Files")
    user_question = st.text_input("Your Question", placeholder="Type your question here...")

    if st.button("Submit Question") and user_question and "raw_text" in st.session_state:
        with st.spinner("Getting answer..."):
            user_input(user_question, "faiss_index")

    st.markdown(
        "<div class='footer'>&copy; 2024 DocuMate | Developed with ❤ using Streamlit</div>", 
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
