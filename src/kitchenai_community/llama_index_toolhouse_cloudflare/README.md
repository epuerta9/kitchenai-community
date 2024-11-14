# KitchenAI Application using LlamaIndex, Toolhouse, and Cloudflare AI

This repository contains a KitchenAI-based application that enables storage, querying, and analysis of documents, specifically designed to handle and analyze RFPs (Request for Proposals) using OpenAI’s language models. It leverages multiple advanced components, including vector storage, metadata management, and natural language processing with OpenAI's models for enhanced AI-driven querying and reporting.

---

## Features

1. **File Storage with Metadata**: Uploads and stores documents in a Chroma vector store with metadata for efficient querying and retrieval.
2. **Document Parsing and Indexing**: Processes documents with tools like token splitting, title extraction, and question-answer extraction to create a summary and vector-based indices.
3. **Advanced Querying**: Provides an agent-based querying interface for extracting insights from RFP documents using context-aware language models.
4. **Customizable Tool Integration**: Allows integration of custom tools and metadata for enhanced flexibility and targeted querying.

---

## Prerequisites

1. **Python 3.11** (ensure compatibility with dependencies)
2. **OpenAI API Key**: Required for language model and embedding services.
3. **ChromaDB**: Vector storage solution for handling vector embeddings.
4. **Environment Variables**:
   - `CLOUDFLARE_API_KEY` and `CLOUDFLARE_API_BASE_URL`: Required for custom LLM API endpoints.
   - `KITCHENAI_DEBUG`: True
   - `TOOLHOUSE_API_KEY`: Key for accessing the Toolhouse API.
   - `OPENAI_API_KEY`: Key for accessing the OpenAI API.

---

## Setup Instructions

1. **Create new environment**:
   ```bash
   python -m venv venv && source venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install kitchenai && pip install -r requirements.txt
   ```

2. **Environment Configuration**:
   Set up your environment variables using export command.

3. **Run the Server**:
   Start the server to enable file uploads and agent-based querying:
   ```bash
   kitchenai init && kitchenai setup && kitchenai dev --module app:kitchen 
   ```

---

## Usage Guide

### 1. File Upload and Storage with Metadata

> **IMPORTANT**: Access the Swagger UI documentation at `/api/docs` to upload, manage files, and query the RFPs. This is the primary interface for file operations.

> **IMPORTANT**: Access the Swagger UI documentation at `/kitchenai-admin` with the `admin@localhost` and `admin` user credentials to manage files.

example data used for this demo: [](https://github.com/epuerta9/kitchenai-community/tree/main/src/kitchenai_community/llama_index_toolhouse_cloudflare/data)



The `chromadb_storage` function processes and stores documents, building indices to facilitate quick and relevant querying.

**Route**: `/storage/file`  
**Description**: Stores uploaded files into ChromaDB with metadata, generating indices for quick retrieval and document summary.  

Example:
```python
@kitchen.storage("file")
def chromadb_storage(dir: str, metadata: dict = {}, *args, **kwargs):
    # Stores files in vector store with metadata
```

#### Key Components

- **Vector Storage**: `ChromaVectorStore` instances for storage with and without metadata.
- **Index Creation**:
   - `VectorStoreIndex`: Creates an index with document transformations like token splitting, title extraction, and question-answer extraction.
   - `StorageContext`: Manages storage across Chroma collections.

### 2. Querying with Agents

The `agent` function sets up a querying agent that allows users to request specific information, targeting both the general document collection and individual RFPs by file label.

**Route**: `/query/agent`  
**Description**: Provides natural language querying capabilities on stored documents with a custom agent, specialized for RFP document analysis.  

Example:
```python
@kitchen.query("agent")
def agent(request, query: Query):
    # Agent-based querying on vectorized documents
```

#### Key Components

- **OpenAI Language Models**: Utilizes OpenAI’s models for chat-based querying.
- **Query Engine**: `QueryEngineTool` instances are set up to route queries to the appropriate document context (e.g., "RFP Collection" or specific RFPs).
- **Agent Setup**:
   - `OpenAIAgent`: Manages the language model-based querying.
   - **Toolhouse Integration**: Integrates with Toolhouse.ai to add and manage tools with custom metadata.
   - **Object Indexing**: Uses `ObjectIndex` for retrieving relevant tools for each query based on similarity.

### 3. How It Works

- **File Storage**:
   - Files are uploaded, parsed, and stored into Chroma vector stores, with metadata added for categorization.
   - Custom document parsers extract structured data and build indices to enhance query performance.
- **Querying**:
   - The `agent` function uses a `VectorStoreIndex` to perform similarity searches on stored RFPs.
   - Each query is processed through a context-aware agent that chooses the relevant data and provides an accurate response based on stored knowledge.
   - Responses include generated text, metadata, and document source references.

---

## Example Workflow

1. **Upload RFP Files**:
   - Use the `/storage/file` endpoint to upload RFPs, which will be processed and indexed.

2. **Query RFP Information**:
   - Use `/query/agent` to make specific queries on the stored RFPs. For example:
   ```json
   {
       "query": "Summarize the construction RFP requirements"
   }
   ```

   - The agent will respond with a structured report, listing requirements and comparisons to other RFPs as needed.

---

## Additional Information

- **Integrations**: This setup is designed to be modular, allowing additional tools and configurations to be easily added through `Toolhouse` and the `KitchenAIApp` interface.
- **Debugging**: Set `KITCHENAI_DEBUG=True` to get detailed logs during development and debugging.

This setup demonstrates how KitchenAI can manage and analyze complex document workflows, making it suitable for RFP management and other document-heavy applications.