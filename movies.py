import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from PIL import Image
import os

llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.5)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

logo_path = "./jio.jpeg"
logo_url = "https://pbs.twimg.com/media/D6cxKX_V4AAXWaH.jpg"

if os.path.exists(logo_path):
    logo = Image.open(logo_path)
else:
    logo = logo_url

system_template = r'''
Use the provided movie synopsis, which may be written in Hindi. If written in Hindi, translate it to English to answer the following questions.
Ensure your answers are detailed and written in paragraph form.
You are an expert writer but DO NOT ADD YOUR OWN SCRIPT UNLESS ASKED TO.
Do not answer questions that are not related to movies.
---------------
Context: ```{context}```
'''

user_template = '''
Question: ```{question}```
'''

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(user_template)
]

qa_prompt = ChatPromptTemplate.from_messages(messages)

def load_docx_files(directory_path):
    from langchain.document_loaders import Docx2txtLoader
    documents = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.docx'):
                file_path = os.path.join(root, file)
                print(f'Loading {file_path}')
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
    return documents

def chunk_data(data, chunk_size=900, chunk_overlap=90):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(data)
    return chunks

def create_embedding_chroma(chunks, persist_directory="./chroma_db"):
    from langchain.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    return vector_store

def load_embeddings_chroma(persist_directory="./chroma_db"):
    from langchain.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vector_store

def ask_question(q, chain):
    result = chain.invoke({'question': q})
    return result

if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    if 'vs' not in st.session_state:
        st.session_state['vs'] = None

    st.header("JioStudios ScriptsGPT")
    st.image(logo, width=80)

    # with st.sidebar:
    #     uploaded_folder = st.text_input('Enter the path of the folder containing your files:')
        
    #     if uploaded_folder:
    #         data = load_docx_files(uploaded_folder)
    #         chunks = chunk_data(data)
    #         vector_store = create_embedding_chroma(chunks)
    #         st.session_state['vs'] = vector_store

    q = st.text_input("Enter the question")
    if q:
        if st.session_state['vs'] is not None:
            vector_store = st.session_state['vs']
        else:
            vector_store = load_embeddings_chroma()
            st.session_state['vs'] = vector_store

        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 20})
        crc = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            chain_type="stuff",
            combine_docs_chain_kwargs={'prompt': qa_prompt},
            verbose=False
        )
        answer = ask_question(q, crc)
        st.text_area('Answer:', value=answer['answer'], height=400)