from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from uuid import uuid4
from urllib.parse import quote
import os
from dotenv import load_dotenv

load_dotenv()

# ✅ PDF hosting base URL
BASE_URL = "http://127.0.0.1:8002/safe-pdf"

# Embeddings setup
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Vector DB setup
connection = "postgresql+psycopg://langchain:langchain@localhost:5432/langchain"
collection_name = "my_docs"
vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

# Load the PDF
file_path = "media/pdfs/international_tourism.pdf"
filename = os.path.basename(file_path)
encoded_filename = quote(filename)  # Encode special characters for URL
pdf_url = f"{BASE_URL}/{encoded_filename}"

loader = PyPDFLoader(file_path)
documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200
)
split_docs = text_splitter.split_documents(documents)

# Add metadata and unique IDs
documents_with_metadata = []
for i, doc in enumerate(split_docs):
    new_doc = Document(
        page_content=doc.page_content,
        metadata={
            "uuid": str(uuid4())[:8],
            "filename": filename,
            "pdf_url": pdf_url,  # ✅ Include full URL to the hosted PDF
        }
    )
    documents_with_metadata.append(new_doc)

# Add to PGVector
vector_store.add_documents(
    documents_with_metadata,
    ids=[doc.metadata["uuid"] for doc in documents_with_metadata]
)

# Optional search test
# results = vector_store.similarity_search("sustainable tourism", k=5)
# for doc in results:
#     print(f"{doc.page_content[:200]}... [source: {doc.metadata['source']}, page: {doc.metadata.get('page_number')}, url: {doc.metadata['pdf_url']}]")
