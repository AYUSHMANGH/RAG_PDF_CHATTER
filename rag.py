import os
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import streamlit as st
load_dotenv()

# Fetching the groq_api_key
groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="PDF RAG CHATTER",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("WELCOME TO RAG PDF CHATTER")

st.sidebar.header("Engine Configuration ")

st.subheader("Upload your documents")

# Add the file uploader widget
uploaded_file = st.file_uploader("Choose a pdf file to save to your docs", type=["pdf"] )

# Check if a file was successfully uploaded
if uploaded_file:
    # Securely build the target path
    file_path = os.path.join("./docs", uploaded_file.name)
    
    # 4. Write the file bytes from RAM to disk
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.success(f"Saved file successfully to: {file_path}")

loader = DirectoryLoader(
    path="./docs",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

# STEP 1 : Loading the Documents
docs = loader.load()

chunkSize =st.sidebar.slider("Set Chunk Size :",20,1000,500)
chunkOverlap =st.sidebar.slider("Set Chunk overlap :",10,500,100)

top_k = st.sidebar.slider("Set Top k results :",1,10,4)
Temp = st.sidebar.slider("Set Temperature :",0.00,1.00,0.2)

#inintialize the spiltter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunkSize,
    chunk_overlap=chunkOverlap,
    separators=["\n\n", "\n", " ", ""]
)

result = splitter.split_documents(docs)

st.sidebar.write("Total chunks are :",len(result))

# NOW THE CREATING THE EMBEDDING AND STORING IN VECTORDB
try:
    @st.cache_resource
    def get_embeddings():
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    embeddings = get_embeddings()

    # Connect to the Chroma collection
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory="Great_chroma_db",
        collection_name="sample"
    )

    # Wipe the collection completely (removes all old chunks)
    vector_store.reset_collection()

    # Add only the newly split chunks
    vector_store.add_documents(result)

except(Exception):
    st.warning("Please upload an file")

# FINAL STEP RETRIEVER

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": top_k}
)

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=Temp,
    api_key=groq_api_key
)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided context .
      If the context is insufficient, just say you don't know .

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)


question = st.text_input("Enter your message")

if question:
    with st.spinner("Searching documents and generating answers..."):
        retrieved_docs = retriever.invoke(question)
        context = "\n\n".join(i.page_content for i in retrieved_docs)

        final_prompt = prompt.invoke({"context": context, "question": question})

        ans = llm.invoke(final_prompt)

        st.write(ans.content)

        with st.expander("Retrived docs:"):
            st.write(context)