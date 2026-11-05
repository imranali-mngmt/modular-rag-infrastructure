# Modular RAG Infrastructure

A clean, reusable, and modular RAG infrastructure project designed with SOLID principles, provider abstraction, dependency injection, and separated evaluation workflows.

The goal of this project is to build an enterprise-style AI/RAG system step by step, where each component can be tested independently before moving to the next layer.

## Current Features

- Reusable LLM provider architecture
- Supports multiple LLM providers:
  - OpenAI
  - Gemini
  - OpenRouter
  - Groq
- Provider switching through `.env`
- Factory + Registry pattern for LLM providers
- Separate prompt builders using SystemMessage/HumanMessage style messages
- Reusable embedding provider architecture
- Supports:
  - OpenAI embeddings
  - Google Gemini embeddings
- Document ingestion from terminal
- PDF and text loading
- Text chunking with overlap
- Qdrant vector store integration
- Retriever-only testing before generation
- CLI commands:
  - `/ingest`
  - `/query`
  - `/list`
  - `/clear`
  - `/help`
  - `/exit`
- Cohere reranker planned/integrated for better retrieval precision
- Hybrid search with Qdrant dense + sparse vectors planned/in progress
- DeepEval retriever evaluation planned in separate `evals/` folder

## Architecture Goals

This project follows clean architecture ideas:

- `domain` contains interfaces and abstract contracts
- `infrastructure` contains concrete implementations
- `application` contains services/use-cases
- `api/dependencies` contains dependency injection providers
- `langchain/prompts` contains prompt builders
- `evals` contains evaluation code, separate from app source code

## Tech Stack

- Python
- LangChain
- Qdrant Client
- pypdf
- Cohere
- DeepEval
- dotenv
- Optional:
  - FastEmbed for sparse embeddings

## Project Structure

```text
my_infrastructure_project/
├── main.py
├── requirements.txt
├── .env
├── data/
│   ├── raw/
│   └── eval/
├── evals/
│   ├── datasets/
│   ├── scripts/
│   └── results/
├── src/
│   └── app/
│       ├── api/
│       │   └── dependencies/
│       │       └── providers.py
│       ├── application/
│       │   └── services/
│       │       ├── indexing_service.py
│       │       └── retrieval_service.py
│       ├── core/
│       │   ├── config.py
│       │   └── exceptions.py
│       ├── domain/
│       │   └── interfaces/
│       │       ├── llm_interface.py
│       │       ├── embeddings_interface.py
│       │       ├── sparse_embeddings_interface.py
│       │       ├── vectorstore_interface.py
│       │       ├── retriever_interface.py
│       │       └── reranker_interface.py
│       ├── infrastructure/
│       │   ├── llm/
│       │   │   ├── openai_llm.py
│       │   │   ├── gemini_llm.py
│       │   │   ├── openrouter_llm.py
│       │   │   ├── groq_llm.py
│       │   │   └── factory.py
│       │   ├── embeddings/
│       │   │   ├── openai_embeddings.py
│       │   │   ├── google_embeddings.py
│       │   │   ├── sparse_embeddings.py
│       │   │   └── factory.py
│       │   ├── document_processing/
│       │   │   ├── pdf_loader.py
│       │   │   ├── text_loader.py
│       │   │   ├── loader_factory.py
│       │   │   └── text_splitter.py
│       │   ├── vectorstores/
│       │   │   └── qdrant_store.py
│       │   ├── retrievers/
│       │   │   └── vectorstore_retriever.py
│       │   └── rerankers/
│       │       └── cohere_reranker.py
│       └── langchain/
│           └── prompts/
│               └── builders.py