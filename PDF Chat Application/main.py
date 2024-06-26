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
import whisper

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_from_file(text_file):
    with open(text_file, "r", encoding="utf-8") as file:
        text = file.read()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    st.text("Whisper Model Loaded")

    st.sidebar.success("Transcribing audio file")
    transcription = model.transcribe(audio_file)
    st.sidebar.success("Transcription Completed")
    st.sidebar.markdown(transcription["text"])

    return transcription["text"]

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])




# def main():
#     st.set_page_config("Chat with PDF")
#     st.header("INSIGHT EXTRACTION USING MULTIMODAL ANALYSIS OF CONSUMER VIDEOS")

#     user_question = st.text_input("Ask a Question")

#     audio_file = st.file_uploader("Upload your Video/Audio file and Click on the Submit & Process Button", type=["mp3", "wav", "mp4"])

#     model = whisper.load_model("base")
#     st.text("Whisper Model Loaded")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Menu:")
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):

#                 if audio_file is not None:
#                     st.sidebar.success("Transcribing audio file")
#                     transcription = model.transcribe(audio_file.name)
#                     st.sidebar.success("Transcription Completed")
#                     st.sidebar.markdown(transcription["text"])
#                     #raw_text = get_pdf_text(pdf_docs)
#                     raw_text = transcription["text"]
#                     text_chunks = get_text_chunks(raw_text)
#                     get_vector_store(text_chunks)
#                     st.success("Done")

def main():
    st.set_page_config("Chat with PDF")
    st.header("INSIGHT EXTRACTION USING MULTIMODAL ANALYSIS OF CONSUMER VIDEOS")

    user_question = st.text_input("Ask a Question")

    audio_file = st.file_uploader("Upload your Video/Audio file and Click on the Submit & Process Button", type=["mp3", "wav", "mp4"])

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if audio_file is not None:
                    raw_text = transcribe_audio(audio_file.name)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")


if __name__ == "__main__":
    main()