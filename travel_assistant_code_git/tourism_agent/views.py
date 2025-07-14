import os
import json
import time
import re
import pandas as pd
from django.http import JsonResponse, StreamingHttpResponse, FileResponse, Http404
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.conf import settings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferWindowMemory
from langchain.evaluation import QAEvalChain
import io
from django.http import HttpResponse
from tourism_agent.tools import vector_db_as_notes,website_search_runnable_fn
# Load environment variables
load_dotenv()

# Setup API keys and models
openai_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_KEY")

# Compile regex patterns
URL_PATTERN = re.compile(r'https?://[^\s]+')
UUID_PATTERN = re.compile(r'uuid:\s*([a-f0-9\-]+)', re.IGNORECASE)
PDF_URL_PATTERN = re.compile(r'https?://[^\s]+?\.pdf')

# Model choices
main_rag_model = ChatOpenAI(model="o1-mini", streaming=True)
eval_llm = OpenAI(temperature=0, openai_api_key=openai_key)
parser = StrOutputParser()
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
memory = ConversationBufferWindowMemory(return_messages=True, memory_key="chat_history", k=5)

# Vector stores

# Vector DB search function


vector_db_runnable = RunnableLambda(vector_db_as_notes)

# Web search function


website_search_runnable = RunnableLambda(website_search_runnable_fn)

# Prompt templates
prompt3 = PromptTemplate(
    template=(
        """You are a helpful AI assistant. I will give you three sources of information: chat history, vector DB data, and website data.

        Your job is to:

        Find and provide the answer using one of these sources and provide output inside body (dont include body tag) of html or give me answer without '``` html ' with proper bullet points but for url just use <p> tag, pdf url should be placed along with their respective answers

        Along with the answer, also include:

        The corresponding UUID as uuid: ... dont give me uuid in '**UUID:**' i want it in this format uuid:

        The full PDF URL (ending in .pdf) from the same document you used for the answer.
        make sure you provide pdf url for their corresponding answers

        If you use chat history to give answer donot include both url and uuid   
         
        If the user asks a question unrelated to the data, feel free to reply conversationally — just don't reveal anything about the chat history source.\n\n"
        "Chat History: {chat_history}\n\n"
        "Vector DB:\n{vector}\n\n"
        "Website:\n{website} \n\n question: {text}\n\n
        """
    ),
    input_variables=["chat_history", "vector", "website", "text"]
)

# Load chat history
load_chat_history = RunnableLambda(lambda x: {
    **x,
    "chat_history": memory.load_memory_variables({})["chat_history"]
})

# Define the main chain
context_preparation_chain = (
    load_chat_history |
    RunnableParallel({
        "retrieval_results": RunnableParallel({
            "vector_output": vector_db_runnable,
            "website_output": website_search_runnable,
        }),
        "text": RunnablePassthrough(),
        "chat_history": RunnableLambda(lambda x: {"chat_history": x["chat_history"]})
    }) |
    RunnableLambda(lambda x: {
        "vector": x["retrieval_results"]["vector_output"]["vector_text"],
        "vector_docs": x["retrieval_results"]["vector_output"]["vector_docs"],
        "db_retrieval_time": x["retrieval_results"]["vector_output"].get("db_retrieval_time"),
        "website": x["retrieval_results"]["website_output"]["website"],
        "web_search_time": x["retrieval_results"]["website_output"].get("web_search_time"),
        "text": x["text"],
        "chat_history": x["chat_history"]
    })
)
answer_generation_chain = prompt3 | main_rag_model | parser

def extract_urls(text):
    urls = URL_PATTERN.findall(text)
    return urls

def extract_gold_answer(query, documents):
    """Extract all relevant content from retrieved documents based on the query."""
    relevant_content = []
    for doc in documents:
        # Simple heuristic: check if query keywords are in the document
        if any(keyword.lower() in doc.page_content.lower() for keyword in query.split()):
            relevant_content.append(doc.page_content)
    
    # Join all relevant content, perhaps with a separator
    return "\n\n".join(relevant_content) if relevant_content else ""
import traceback

@csrf_exempt
def chat_with_graph_agent(request):
    if request.method == 'GET':
        return render(request, 'tourism_agent/chat.html')
    print('tavily key >?????',tavily_key)
    print('opemao key ', openai_key)
    if request.method == 'POST':
        try:
            body = json.loads(request.body)
            query = body.get('message', '')
            start_overall_time = time.time()

            context_data = context_preparation_chain.invoke({'text': query})
            retrieved_documents = context_data['vector_docs']
            db_retrieval_time = context_data['db_retrieval_time']
            web_search_time = context_data['web_search_time']

            gold_answer = extract_gold_answer(query, retrieved_documents)
            eval_chain = QAEvalChain.from_llm(llm=eval_llm)

            def event_stream():
                try:
                    full_ai_response = ""
                    for chunk in answer_generation_chain.stream(context_data):
                        full_ai_response += chunk
                        cleaned_chunk = UUID_PATTERN.sub('', chunk).replace('@', '')
                        yield json.dumps({'type': 'chunk', 'data': cleaned_chunk}) + '\n'

                    print('Full AI response:', full_ai_response)

                    extracted_uuids = UUID_PATTERN.findall(full_ai_response)
                    print("Extracted UUIDs:", extracted_uuids)

                    extracted_uuid = extracted_uuids[0] if extracted_uuids else None

                    matching_pages = [
                        doc.page_content for doc in retrieved_documents
                        if 'uuid' in doc.metadata and doc.metadata['uuid'] in extracted_uuids
                    ]
                    page_content_found = ' '.join(matching_pages)
                    if matching_pages:
                        print("Matched content found for UUIDs.")

                    found_urls = extract_urls(full_ai_response)
                    cleaned_urls = [PDF_URL_PATTERN.search(url).group(0) for url in found_urls if PDF_URL_PATTERN.search(url)]
                    pdf_url_found = cleaned_urls[0] if cleaned_urls else ""

                    memory.save_context({"input": query}, {"output": full_ai_response})

                    examples = [{
                        "query": query,
                        "answer": gold_answer,
                        "contexts": [doc.page_content for doc in retrieved_documents]
                    }]
                    predictions = [{
                        "result": full_ai_response
                    }]

                    eval_results = eval_chain.evaluate(examples=examples, predictions=predictions) if gold_answer else []
                    eval_output = eval_results[0].get("results", "No evaluation due to missing gold answer") if eval_results else "No evaluation due to missing gold answer"

                    csv_data = [{
                        "Query": query,
                        "Gold Answer": gold_answer if gold_answer else "No gold answer found",
                        "LLM Answer": full_ai_response,
                        "Evaluation": eval_output,
                        "Contexts": " | ".join([doc.page_content for doc in retrieved_documents]),
                        "UUID": extracted_uuid or "",
                        "PDF URL": pdf_url_found,
                        "DB Retrieval Time": db_retrieval_time,
                        "Web Search Time": web_search_time
                    }]

                    try:
                        csv_file = os.path.join(settings.BASE_DIR, "evaluation_results.csv")
                        print("Saving CSV with data:", csv_data)
                        print("CSV file absolute path:", os.path.abspath(csv_file))
                        df = pd.DataFrame(csv_data)
                        if os.path.exists(csv_file):
                            df.to_csv(csv_file, mode='a', header=False, index=False)
                        else:
                            df.to_csv(csv_file, mode='w', header=True, index=False)
                        print(f"CSV saved successfully at {csv_file}")
                    except Exception as csv_e:
                        print("Error while saving CSV:", csv_e)
                        print(traceback.format_exc())

                    final_cleaned_response = UUID_PATTERN.sub('', full_ai_response).replace('@', '')

                    final_payload = {
                        'type': 'end',
                        'response_time': round(time.time() - start_overall_time, 2),
                        'db_retrieval_time': db_retrieval_time,
                        'web_search_time': web_search_time,
                        'document': [],
                        'full_response': final_cleaned_response,
                        'paragraph_content': page_content_found,
                        'extracted_uuid': extracted_uuid,
                        'pdf_url_found': pdf_url_found
                    }
                    yield json.dumps(final_payload) + '\n'

                except Exception as e:
                    print("Error in event_stream:", e)
                    print(traceback.format_exc())
                    yield json.dumps({'type': 'error', 'message': str(e)}) + '\n'

            return StreamingHttpResponse(event_stream(), content_type="text/event-stream")

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON in request body'}, status=400)
        except Exception as e:
            print("Unhandled error in chat_with_graph_agent:", e)
            print(traceback.format_exc())
            return JsonResponse({'error': str(e)}, status=500)



def safe_pdf_view(request, filename):
    pdf_path = os.path.join(settings.MEDIA_ROOT, 'pdfs', filename)
    if os.path.exists(pdf_path):
        response = FileResponse(open(pdf_path, 'rb'), content_type='application/pdf')
        response["X-Frame-Options"] = "ALLOWALL"
        return response
    else:
        raise Http404("PDF not found")
    




from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

@csrf_exempt
def batch_qa_evaluation(request):
    if request.method == 'POST':
        try:
            body = json.loads(request.body)
            qa_data = body.get('qa_data', [])

            if not qa_data:
                return JsonResponse({'error': 'No QA data provided'}, status=400)

            results = []

            def process_item(idx, item):
                try:
                    query = item.get('question', '')
                    expected_answer = item.get('answer', '')

                    context_data = context_preparation_chain.invoke({'text': query})
                    retrieved_documents = context_data['vector_docs']

                    full_ai_response = ""
                    for chunk in answer_generation_chain.stream(context_data):
                        full_ai_response += chunk

                    eval_chain = QAEvalChain.from_llm(llm=eval_llm)

                    examples = [{
                        "query": query,
                        "answer": expected_answer,
                        "contexts": [doc.page_content for doc in retrieved_documents]
                    }]
                    predictions = [{
                        "result": full_ai_response
                    }]

                    eval_results = eval_chain.evaluate(examples=examples, predictions=predictions) if expected_answer else []
                    eval_output = eval_results[0].get("results", "No evaluation due to missing expected answer") if eval_results else "No evaluation due to missing expected answer"

                    return {
                        "S.No.": idx,
                        "Query": query,
                        "Expected Answer": expected_answer,
                        "LLM Answer": full_ai_response,
                        "Evaluation": eval_output
                    }

                except Exception as e:
                    return {
                        "S.No.": idx,
                        "Query": item.get('question', ''),
                        "Expected Answer": item.get('answer', ''),
                        "LLM Answer": f"Error: {e}",
                        "Evaluation": "Error"
                    }

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(process_item, idx, item) for idx, item in enumerate(qa_data, start=1)]
                for future in as_completed(futures):
                    results.append(future.result())

            # Generate Excel
            df = pd.DataFrame(results)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False)
            output.seek(0)

            response = HttpResponse(output.read(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            response['Content-Disposition'] = 'attachment; filename="qa_evaluation_results.xlsx"'
            return response

        except Exception as e:
            print(traceback.format_exc())
            return JsonResponse({'error': str(e)}, status=500)

    return render(request, 'tourism_agent/evaluation.html')
