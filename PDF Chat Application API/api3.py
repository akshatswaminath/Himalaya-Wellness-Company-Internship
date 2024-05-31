from fastapi import FastAPI, UploadFile, File, HTTPException
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

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Global variables (optional for caching)
vector_store = None
model = None

def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        try:
            pdf_reader = PdfReader(pdf.file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {e}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def update_vector_store(pdf_filename, text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_filename = f"{pdf_filename}.faiss"

    try:
        # Create and save .faiss file with new text chunks
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(faiss_filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating .faiss file: {e}")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    global model
    if not model:
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(pdf_filename: str, question: str):
    # Check if .faiss file exists, otherwise raise exception
    faiss_filename = f"{pdf_filename}.faiss"
    if not os.path.exists(faiss_filename):
        raise HTTPException(status_code=400, detail="Please process PDFs first!")

    # Load embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load the single faiss file
    try:
        vector_store = FAISS.load_local(faiss_filename, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading .faiss file: {e}")

    # Perform similarity search
    docs = vector_store.similarity_search(question)

    # Get conversational chain
    chain = get_conversational_chain()

    # Get response from conversational chain
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)

    return {"answer": response["output_text"]}

app = FastAPI()

@app.post("/process_pdf")
async def process_pdf(pdf_files: list[UploadFile] = File(...)):
    # Get file name without extension
    pdf_filename = os.path.splitext(pdf_files[0].filename)[0]

    # Update the vector store with new text chunks
    try:
        raw_text = get_pdf_text(pdf_files)
        text_chunks = get_text_chunks(raw_text)
        update_vector_store(pdf_filename, text_chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {e}")

    # Return success message
    return {"message": "Processing successful!"}

@app.post("/ask_question")
async def ask_question(pdf_filename: str, question: str):
    # Call user_input function to compare with the single updated faiss file
    return user_input(pdf_filename, question)
