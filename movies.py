import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from PIL import Image
import os
import pinecone

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

logo_path = "./jio.jpeg"
logo_url = "https://pbs.twimg.com/media/D6cxKX_V4AAXWaH.jpg"
#logo_url = "https://upload.wikimedia.org/wikipedia/en/0/02/Dharma_Production_logo.png"

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

def create_embedding_pinecone(chunks, index_name="jio-index"):
    import pinecone
    from langchain_community.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    from pinecone import PodSpec
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    
    # Initialize Pinecone
    pc = pinecone.Pinecone()
    
    # Create Pinecone index
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=PodSpec(
                environment='gcp-starter'
            )
        )

    vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
    return vector_store

def load_embeddings_pinecone(index_name="jio-index"):
    from langchain.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
    
    # Initialize Pinecone 
    vector_store = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
    return vector_store

def ask_question(q, chain):
    result = chain.invoke({'question': q})
    return result['answer']

if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

    if 'vs' not in st.session_state:
        st.session_state['vs'] = None

    st.header("Dharma ScriptsGPT")
    st.image(logo, width=100)

    # with st.sidebar:
    #     uploaded_folder = st.text_input('Enter the path of the folder containing your files:')
        
    #     if uploaded_folder:
    #         data = load_docx_files(uploaded_folder)
    #         chunks = chunk_data(data)
    #         vector_store = create_embedding_pinecone(chunks)
    #         st.session_state['vs'] = vector_store
    st.text_area('Suggestions:', value="Try: Give me 5 latest horror scripts\n or \nScripts similar to Fast and furious or Annabelle", height=100)
    q = st.text_input("Enter the question")
    
    if q:
        with st.spinner("Running..."):
            if st.session_state['vs'] is not None:
                vector_store = st.session_state['vs']
            else:
                vector_store = load_embeddings_pinecone()
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
        st.text_area('Answer:', value=answer, height=300)

        st.divider()
        if 'history' not in st.session_state:
            st.session_state.history = ''
        value = f'Q: {q} \nA: {answer}'
        st.session_state.history = f'{value} \n {"-"*100}\n {st.session_state.history}'
        h = st.session_state.history
        st.text_area(label="Chat history",value=h, key='history',height = 300)
