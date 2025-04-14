from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load and split your knowledge base
with open("support_knowledgebase.txt", "r", encoding="utf-8") as file:
    text = file.read()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs_split = splitter.create_documents([text])

"""
# Load and split your knowledge base
loader = PyPDFLoader("support_knowledgebase.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs_split = splitter.split_documents(docs)

"""

# Convert to vector store
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(docs_split, embedding)
db.save_local("email_vectorstore")
print("Vector store built and saved successfully.")
# Note: Ensure you have the necessary PDF file in the same directory for this to work.