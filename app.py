import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chains import ConversationalRetrievalChain

load_dotenv()

def get_pdf_text(pdf_docs):
  text = ""
  for pdf in pdf_docs:
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
      text += page.extract_text()
    return text

def get_text_chunks(raw_text):
  text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
  )
  chunks = text_splitter.split_text(raw_text)
  return chunks

def get_vectorstore(text_chunks):
  embeddings = OpenAIEmbeddings()
  #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
  vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
  return vectorstore

def get_conversation_chain(vectorstore):
  llm = ChatOpenAI()
  memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
  conversation_chain = ConversationRetrievalChain.from_llm(
    llm = llm,
    retriever = vectorStore.as_retriever(),
    memory = memory
  )     

def main():
    st.set_page_config(page_title="PDFs ChatBot", page_icon=":books:")
    if "conversation" not in st.session_state:
      st.session_state.conversation = None
    
    st.header("Chat with my pdfs :books:")
    st.text_input("Ask a questiont to your document")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
          with st.spinner("Processing"):
            # get pdf text
            raw_text = get_pdf_text(pdf_docs)
            # get the text chunks
            text_chunks = get_text_chunks(raw_text)
            st.write(text_chunks)
            # create vector store
            vectorstore = get_vectorstore(text_chunks)
            # create conversation chain
            st.session_state.conversation = get_conversation_chain(vectorstore)

    st.session_state.conversation

if __name__== '__main__':
    main()
