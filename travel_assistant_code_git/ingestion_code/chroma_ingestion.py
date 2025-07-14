import os
from uuid import uuid4
from urllib.parse import quote
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

# ✅ Base URL for your served PDFs
BASE_URL = "http://127.0.0.1:8002/safe-pdf"

# Setup OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY")
)

# ✅ List your actual PDF file paths here
pdf_files = [
    "media/pdfs/tourism1-1-100.pdf",
    # Add more files here if needed
]

# Prepare final documents
final_documents = []

for file_path in pdf_files:
    filename = os.path.basename(file_path)
    encoded_filename = quote(filename)  # Handle spaces and special characters

    pdf_url = f"{BASE_URL}/{encoded_filename}"

    # Load PDF and split it
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(documents)

    # Add metadata, including a UUID
    for doc in split_docs:
        # Generate a UUID for each document
        doc_uuid = str(uuid4())[:8]
        final_documents.append(Document(
            page_content=doc.page_content,
            metadata={
                "filename": filename,
                "pdf_url": pdf_url,
                "uuid": doc_uuid  # Add the UUID to the metadata
            }
        ))

# Setup Chroma vector store
vector_store = Chroma(
    collection_name="project_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)

# Add documents to vector store
# Now, you don't need to generate a separate list of UUIDs here.
# The UUID is already in the document metadata.
# If you still want to explicitly pass IDs to Chroma's add_documents,
# you would extract them from the metadata.
# For simplicity, if you've put it in metadata, you might not need to pass it as `ids`
# unless Chroma requires it for its internal indexing and you want to ensure consistency.
# If you want to use the UUID from metadata as the ID for Chroma, you can do:
ids_for_chroma = [doc.metadata["uuid"] for doc in final_documents]
vector_store.add_documents(documents=final_documents, ids=ids_for_chroma)


# Test similarity search
results = vector_store.similarity_search_with_score("what is tourism", k=2)
for doc, score in results:
    print(f"[SIM={score:.3f}]")
    print(f"PDF URL: {doc.metadata['pdf_url']}")
    print(f"UUID: {doc.metadata['uuid']}") # Access the UUID from metadata
    print(doc.page_content[:200] + "...\n")