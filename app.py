import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import camelot
from tabula import read_pdf
import fitz  
from PIL import Image
import io
import tempfile
from docx import Document

st.set_page_config(page_title="Chat PDF with Gemini", page_icon="💬", layout="wide")

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are a friendly and helpful assistant. Answer the user's question based on the context provided below.
    Provide detailed, engaging, and easy-to-understand explanations. If the context does not contain the answer, respond with
    "The answer is not available in the provided context." Do not make up answers.

    When answering:
    1. Start with a summary of the concept.
    2. Provide additional details or examples if available.
    3. End with a friendly prompt for follow-up questions.

    Context:
    {context}

    Question: 
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7) #0.7 will generate more interactive results

    # Define the template for the chain
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def load_faiss_index():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if os.path.exists("faiss_index"):
        faiss_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        return faiss_store
    else:
        return None

def process_and_update_faiss(pdf_docs):
    # Extract text and create new FAISS index
    all_text = ""
    for pdf_file in pdf_docs:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            all_text += page.extract_text()

    if all_text.strip():
        text_chunks = get_text_chunks(all_text)
        get_vector_store(text_chunks)
        # st.success("FAISS index has been updated with the uploaded PDFs.")
    # else:
    #     st.error("No text could be extracted from the uploaded PDFs.")

def extract_tables_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_file.read())
        tables = camelot.read_pdf(temp_pdf.name, pages="all", flavor="stream")  # Use 'stream' for raw tables
        table_data = []
        for table in tables:
            table_data.append(table.df)  # Convert each table to a DataFrame
        return table_data

# Extract Tables using Tabula
def extract_tables_tabula(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_file.read())
        tables = read_pdf(temp_pdf.name, pages="all", multiple_tables=True)
        return tables

# Extract Images from PDFs
def extract_images_from_pdf(pdf_file):
    images = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_file.read())
        with fitz.open(temp_pdf.name) as pdf:
            for page_index in range(len(pdf)):
                for img_index, img in enumerate(pdf[page_index].get_images(full=True)):
                    xref = img[0]
                    base_image = pdf.extract_image(xref)
                    image_bytes = base_image["image"]
                    images.append((page_index + 1, img_index + 1, image_bytes))
    return images

# Display Images
def save_or_display_images(images):
    image_elements = []
    for page, index, image_bytes in images:
        image = Image.open(io.BytesIO(image_bytes))
        image_elements.append((page, index, image))
    return image_elements



# Export conversation to .txt
def export_to_txt(conversation):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write("\n".join(conversation).encode("utf-8"))
        tmp_file_path = tmp_file.name
    return tmp_file_path

def export_to_docx(conversation):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        doc = Document()
        for line in conversation:
            doc.add_paragraph(line)
        doc.save(tmp_file.name)
        tmp_file_path = tmp_file.name
    return tmp_file_path


conversation_history = []

def user_input(user_question):
    global conversation_history  
    
    new_db = load_faiss_index()
    docs = new_db.similarity_search(user_question) if new_db else []

  
    chain = get_conversational_chain()

  
    response = chain({"input_documents": docs, "question": user_question})

   
    answer = response["output_text"]

   
    conversation_history.append(f"User: {user_question}")
    conversation_history.append(f"Answer: {answer}")

   
    st.markdown("### Reply:")
    if "answer is not available in the context" in answer.lower():
        st.write("Sorry, the answer could not be found in the provided context. Please refine your query or try asking something else.")
    else:
        st.markdown(answer)



def save_chat_summary(conversation_history, file_name="conversation.txt"):
    
    with open(file_name, "w") as file:
        file.write("\n".join(conversation_history))
    st.success("Chat summary has been saved.")

   
    with open(file_name, "r") as file:
        st.download_button(
            label="Download Chat Summary",
            data=file.read(),
            file_name=file_name,
            mime="text/plain"
        )


def main():

    with st.sidebar:
        st.title("Menu")
        st.write("Upload your PDF files and process them for a better chat experience.")
        
        # File Uploader
        pdf_docs = st.file_uploader("Upload PDF Files (multiple allowed)", accept_multiple_files=True)
        
        # Table Extraction
        st.subheader("🔍 Extract Tables")
        table_tool = st.selectbox("Select Table Extraction Tool", ["Camelot", "Tabula"])
        extract_tables_btn = st.button("Extract Tables")
        
        # Image Extraction
        st.subheader("🖼 Extract Images")
        extract_images_btn = st.button("Extract Images")
    
    # RESULT GENERATION SECTION
    st.title("Chat with PDF using Gemini 🤖")
    
    if pdf_docs:
        process_and_update_faiss(pdf_docs)  # Process and create FAISS index for new PDFs
        user_question = st.text_input("Ask a Question from the PDF Files", placeholder="Type your question here...")

        if user_question:
            user_input(user_question)

        if extract_tables_btn:
            st.subheader("Extracted Tables")
            for pdf in pdf_docs:
                if table_tool == "Camelot":
                    tables = extract_tables_from_pdf(pdf)
                elif table_tool == "Tabula":
                    tables = extract_tables_tabula(pdf)
                
                for i, table in enumerate(tables):
                    st.write(f"**Table {i + 1}**")
                    st.dataframe(table)
        
        if extract_images_btn:
            st.subheader("Extracted Images")
            for pdf in pdf_docs:
                images = extract_images_from_pdf(pdf)
                image_elements = save_or_display_images(images)
                for page, index, image in image_elements:
                    st.image(image, caption=f"Page {page}, Image {index}")

        
        if st.button("Export Conversation to .txt"):
            if conversation_history:
                txt_file_path = export_to_txt(conversation_history)
                with open(txt_file_path, "r") as txt_file:
                    st.download_button(
                        label="Download Conversation as TXT",
                        data=txt_file.read(),
                        file_name="conversation.txt",
                        mime="text/plain"
                    )
            else:
                st.warning("No conversation history to export.")

        if st.button("Export Conversation to .docx"):
            if conversation_history:
                docx_file_path = export_to_docx(conversation_history)
                with open(docx_file_path, "rb") as docx_file:
                    st.download_button(
                        label="Download Conversation as DOCX",
                        data=docx_file,
                        file_name="conversation.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
            else:
                st.warning("No conversation history to export.")



if __name__ == "__main__":
    main()
