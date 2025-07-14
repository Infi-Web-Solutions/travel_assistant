import os
import time
import concurrent.futures
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from tavily import TavilyClient
from langchain_community.vectorstores import Chroma
from langchain_postgres import PGVector
# Load environment variables
load_dotenv()

# Setup API keys and models
openai_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large",api_key=openai_key)

vector_store_chroma = Chroma(
    collection_name="project_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)
vector_store_pg = PGVector(
    embeddings=embeddings,
    collection_name="my_docs",
    connection="postgresql+psycopg://langchain:langchain@localhost:5432/langchain",
    use_jsonb=True,
)


#tool 1

def vector_db_as_notes(input: dict) -> dict:
    query = input['text']
    start_db_time = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future1 = executor.submit(vector_store_chroma.similarity_search_with_score, query, k=2)
        future2 = executor.submit(vector_store_pg.similarity_search, query, k=2)
        results1_with_score = future1.result()
        results2 = future2.result()
    end_db_time = time.time()
    db_retrieval_time = round(end_db_time - start_db_time, 4)
    docs1 = [doc for doc, _ in results1_with_score]
    docs2 = results2
    combined_docs = docs1 + docs2
    print('db retrieval time', db_retrieval_time)
    return {
        'vector_text': docs1 + docs2,
        'vector_docs': combined_docs,
        'db_retrieval_time': db_retrieval_time
    }



# tool 2


def website_search_runnable_fn(input: dict) -> dict:
    start_web_search_time = time.time()
    client = TavilyClient(api_key=tavily_key)
    query = input['text']
    response = client.search(
        query=query,
        search_depth="advanced",
        include_domains=["https://www.britannica.com/topic/tourism/Day-trippers-and-domestic-tourism", "https://www.britannica.com/explore/israeli-palestinian-conflicts"]
    )
    docs = "\n\n".join(
        f"{r['content']}\n(Source: {r['url']})" for r in response['results'] if 'content' in r and 'url' in r
    )
    end_web_search_time = time.time()
    web_search_time = round(end_web_search_time - start_web_search_time, 4)
    return {
        'website': docs + "\n\ninstead of 'parargraph' write source link: https://www.britannica.com/topic/tourism/Day-trippers-and-domestic-tourism",
        'web_search_time': web_search_time
    }