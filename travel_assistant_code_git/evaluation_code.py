import os
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.evaluation import QAEvalChain
from langchain_openai import OpenAI
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Setup API keys and models
openai_key = os.getenv("OPENAI_API_KEY")
# Initialize the LLM
llm = ChatOpenAI(model="gpt-4", openai_api_key=openai_key)

# Sample documents for the retriever
docs = [
    Document(page_content="France is a country in Europe. Its capital is Paris.", metadata={}),
    Document(page_content="Paris is known for its cafes and the Eiffel Tower.", metadata={})
]

# Create a FAISS vector store
embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# Define the system prompt for the retrieval chain
system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentences maximum and keep the answer concise. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create the retrieval chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

# Sample input
question = "What is the capital of France?"
gold_answer = "The capital of France is newytork."

result = chain.invoke({"input": question})
llm_generated_answer = result["answer"]

# Format inputs for QAEvalChain
examples = [
    {
        "query": question,
        "answer": gold_answer,
        "contexts": [doc.page_content for doc in result["context"]]
    }
]
predictions = [
    {
        "result": llm_generated_answer
    }
]

# Initialize QAEvalChain
eval_llm = OpenAI(temperature=0, openai_api_key=openai_key)
eval_chain = QAEvalChain.from_llm(llm=eval_llm)

# Run the evaluation
results = eval_chain.evaluate(examples=examples, predictions=predictions)

# Prepare data for CSV
csv_data = []
for i, result in enumerate(results):
    csv_data.append({
        "Query": examples[i]["query"],
        "Gold Answer": examples[i]["answer"],
        "LLM Answer": predictions[i]["result"],
        "Evaluation": result.get("results", "No evaluation provided"),
        "Contexts": " | ".join(examples[i]["contexts"])  # Combine contexts into a single string
    })

# Save to CSV using pandas
df = pd.DataFrame(csv_data)
df.to_csv("evaluation_results.csv", index=False)

# Print confirmation
print("Results saved to 'evaluation_results.csv'")