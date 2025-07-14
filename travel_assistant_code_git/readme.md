
# 🌍 Travel Chat Assistant

This project is a conversational AI assistant designed to provide information related to tourism, leveraging Large Language Models (LLMs) and various data sources. It features real-time chat, document referencing, and a robust evaluation framework for quality assurance.

## ✨ Features

* **Intelligent Chat:** Interact with an AI assistant that can answer your travel-related questions.
* **Multi-Source Information Retrieval:** The assistant can pull information from:
    * **Chat History:** Maintains context from the ongoing conversation.
    * **Vector Database:** Accesses pre-processed document embeddings for relevant information.
    * **Website Search:** Performs real-time web searches to gather up-to-date information (powered by Tavily).
* **Dynamic Referencing:** When the AI provides an answer, it can also display the supporting documents (including PDFs) and highlight the relevant sections.
* **PDF Preview:** View embedded PDF documents directly within the chat interface, with highlighted text corresponding to the AI's answer.
* **Batch QA Evaluation:** An integrated tool for evaluating the AI's performance against a set of questions and expected answers, generating an Excel report.
* **Streaming Responses:** Provides a smooth user experience by streaming AI responses in real-time.

## 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.8+**
* **pip** (Python package installer)
* **Django**
* **Poetry** (recommended for dependency management)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-project-directory>
    ```

2.  **Install dependencies:**
    If you use `pip`:
    ```bash
    pip install -r requirements.txt # You'll need to create this file based on the importsa

3.  **Set up environment variables:**
    Create a `.env` file in the root of your project and add your API keys:

    ```env
    OPENAI_API_KEY="your_openai_api_key_here"
    TAVILY_KEY="your_tavily_api_key_here"
    # Add any other environment variables your Django project needs (e.g., SECRET_KEY)
    ```

4.  **Database Migrations:**
    If your Django project uses a database, apply migrations:
    ```bash
    python manage.py migrate
    ```

5.  **Collect static files (if applicable):**
    ```bash
    python manage.py collectstatic
    ```

### Running the Application

1.  **Start the Django development server:**
    ```bash
    python manage.py runserver
    ```

2.  **Access the application:**
    Open your web browser and navigate to:
    * **Chat Interface:** `http://127.0.0.1:8000/chat/` 
    * **Batch Evaluation Interface:** `http://127.0.0.1:8002/upload-json/`

## ⚙️ Configuration

### API Keys

* **OPENAI_API_KEY**: Required for accessing OpenAI's language models and embeddings. Obtain it from the [OpenAI Platform](https://platform.openai.com/).
* **TAVILY_KEY**: Required for enabling web search capabilities. Obtain it from [Tavily AI](https://tavily.com/).

### Models

* `main_rag_model`: Uses `ChatOpenAI(model="o1-mini", streaming=True)` for the primary RAG (Retrieval-Augmented Generation) model.
* `eval_llm`: Uses `OpenAI(temperature=0, openai_api_key=openai_key)` for evaluation purposes.
* `embeddings`: Uses `OpenAIEmbeddings(model="text-embedding-3-large")` for generating vector embeddings.

### Prompt Template (`prompt3`)

The core of the AI's reasoning is defined in `prompt3`, which instructs the model to act as a helpful AI assistant and leverage chat history, vector database data, and website data to answer questions. It also specifies the output format, including UUIDs and PDF URLs.

## 📁 Project Structure (Implied)

Based on the provided code, the project structure likely includes: