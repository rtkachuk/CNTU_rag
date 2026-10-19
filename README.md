# CNTU_rag

This project is dedicated to university lectures and serves as an example for students learning about RAG (Retrieval-Augmented Generation) search with simple AI models.

## Project Overview

This is a simple implementation of a Retrieval-Augmented Generation (RAG) system that demonstrates how to combine vector databases with language models to answer questions based on document content. The project uses ChromaDB for vector storage and LiteLLM for AI model access.

## Files

### rag.py
Main script implementing a simple RAG (Retrieval-Augmented Generation) system using:
- ChromaDB for vector storage and retrieval
- LiteLLM for embedding and completion services
- Text chunking for processing large documents
- Command-line interface for ingestion and querying

The script provides functionality to:
1. Ingest text files into the vector database
2. Ask questions based on the ingested content
3. Retrieve relevant context from documents and generate answers using AI models

## Requirements

The project requires the following Python packages (listed in requirements.txt):
- chromadb: Vector database for storing document embeddings
- litellm: For embedding and completion services
- openai: OpenAI API client
- other supporting libraries for HTTP, JSON processing, etc.

## Usage

The script can be run from the command line with two main modes:
1. Ingest mode: `python rag.py -i <file>` to ingest a text file
2. Ask mode: `python rag.py -a "<question>"` to ask a question based on ingested content

## Configuration

The project uses environment variables for configuration:
- LITELLM_HOST: Address for your litellm service
- LITELLM_API_KEY: Your API key for litellm
- EMBED_MODEL: Model for generating embeddings (default: "litellm_proxy/qwen-embedding-8b")
- CHAT_MODEL: Model for generating answers (default: "litellm_proxy/qwen-instruct-8b")