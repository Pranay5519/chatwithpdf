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
 
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
            
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size  = 1000 , chunk_overlap = 500)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    
    # Check if the index file already exists
    if os.path.exists("faiss_index"):
        # If it exists, load the existing index
        vector_store = FAISS.load_local("faiss_index", embeddings)
        # Update the existing index with new text chunks
        vector_store.add_texts(text_chunks)
    else:
        # If the index file doesn't exist, create a new index
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    # Save the updated or new index
    vector_store.save_local("faiss_index")

    
def get_conversational_chain():
    prompt_template = """ 
     Answer the question between 10 to 15 words from the provided context, make sure to provide all the details,
     If not answer found in the context the provide answer using LLM
     or give a user friendly answer use emoji.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    """
    
    model = ChatGoogleGenerativeAI(model = "gemini-pro", temprature = 0.9)
    prompt = PromptTemplate(template = prompt_template , input_variables = ['context','question'])            
    chain = load_qa_chain(model , chain_type = 'stuff', prompt = prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    print(response)
    return response   

def main():
    st.set_page_config("Chat PDF")
    st.header("-Chatbot-")

    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    # user input 
    user_question = st.chat_input("Type Here ....")
    
    if user_question:
        response = user_input(user_question)
        
        with st.chat_message("user"):
            st.markdown(user_question)
            
        st.session_state.messages.append({'role' : 'user' , 'content' : user_question})
        
        response = response['output_text']
        
        with st.chat_message('assistant'):
            st.markdown(response)
            
        st.session_state.messages.append({'role' : 'assistant' , 'content' : response})

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()