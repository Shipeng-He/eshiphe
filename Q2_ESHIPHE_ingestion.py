import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import *


load_dotenv()

loader = PyPDFLoader("C:\\AI_Hackathan\\Question 2\\critical_path_method.pdf")
docs = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

embeddings = AzureOpenAIEmbeddings(
    model=os.environ["OPENAI_MODEL_NAME_EMBEDDING"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    openai_api_version=os.environ["OPENAI_API_VERSION_EMBEDDING"]  # Use the correct version for the LLM Model selected
)

db = FAISS.from_documents(texts, embeddings)

# Write our index to disk.
db.save_local("faiss_index")