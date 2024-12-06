# KitchenAI Default Starter Template

Welcome to the **KitchenAI Default Starter Template**! This template provides a foundational setup for working with KitchenAI, integrating document storage, vector indexing, and querying capabilities, all powered by OpenAI’s language models. With this starter, you’ll quickly set up a KitchenAI app to handle document ingestion, vector storage with metadata, and natural language querying.

---

## Features

- **File Storage with Metadata**: Upload and store files in a Chroma vector store with associated metadata for enhanced query relevance.
- **Document Parsing and Indexing**: Automatically parse and transform documents with tools like token splitting, title extraction, and question-answer extraction to create an efficient index.
- **LLM-Driven Querying**: Query stored documents using OpenAI’s `gpt-4` model for intelligent, context-aware responses.

---

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [File Storage](#file-storage)
  - [Querying](#querying)
- [License](#license)

---

---

> **IMPORTANT**: Access the Swagger UI documentation at `/api/docs` to upload, manage files, and query the RFPs. This is the primary interface for file operations.

> **IMPORTANT**: Access the Django Admin at `/kitchenai-admin` with the `admin@localhost` and `admin` user credentials to manage files.

## Prerequisites

1. **Python 3.11** (ensure compatibility with dependencies)
2. **OpenAI API Key**: Required for language model and embedding services.
3. **ChromaDB**: Vector storage solution for handling vector embeddings.
4. **Environment Variables**:
   - `KITCHENAI_DEBUG`: True
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
## Notebook Usage

The `notebook.ipynb` provides an interactive way to explore and test the functionality:

### Setup

1. **Import Extensions**:
   ```python
   %load_ext kitchenai.contrib.cook
   ```

2. **Set Project**:
   ```python 
   %kitchenai_set_project llama-index-starter
   ```

3. **Import Required Libraries**:
   Use the `%%kitchenai_import` magic command to import necessary libraries:
   ```python
   %%kitchenai_import llama-index-imports
   from llama_index.core import VectorStoreIndex, StorageContext
   from llama_index.vector_stores.chroma import ChromaVectorStore
   from llama_index.llms.openai import OpenAI
   # ... additional imports
   ```



### Global Setup

Configure ChromaDB and OpenAI:

using the `%%kitchenai_setup` magic command:

```python
%%kitchenai_setup
```

### Defining your Functions 

This notebook showcases two functions:

- `simple_vector2`: Parses and stores files in a Chroma vector store using metadata.
- `simple_query`: Queries the stored documents based on user input.  

### Converting your functions to KitchenAI

The `%%kitchenai_create_module` magic command will convert your python function into a KitchenAI module.
It will create a new file in the `app.py` file and add the function to the `kitchen` object.



#KitchenAI Module

# KitchenAI Module: Simple Vector and Query Operations

This module is a **KitchenAI** application that demonstrates how to use the KitchenAI framework for performing vector storage and query tasks. It includes two vector storage functions (`simple-vector`, `simple-vector2`) and two query handlers (`simple-query`, `non-ai`) for various use cases.

---

## Features

- **Vector Storage**:
  - Store documents into a vector database using **Chroma** collections.
  - Parse and preprocess documents with the **LLAMA Parser**.
  - Extract features using tools like **TokenTextSplitter**, **TitleExtractor**, and **QuestionsAnsweredExtractor**.

- **Query Handlers**:
  - Perform natural language queries against a vector index using **OpenAI GPT-4**.
  - Provide a simple query handler that does not rely on AI.

---

## Prerequisites

1. **KitchenAI** installed as a dependency.
2. Access to an OpenAI GPT-4 API key.
3. Chroma vector database installed and configured.

---

## Installation

1. Clone the project.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure the following environment variable:
   - `LLAMA_CLOUD_API_KEY`: Your LLAMA Parser API key.

---

## Components

### **1. Vector Storage**

#### `simple-vector`
Stores documents in the `quickstart` Chroma collection.

```python
@kitchen.storage("simple-vector")
def simple_vector(dir: str, metadata: dict = {}, *args, **kwargs):
    ...
```

- **Input Parameters**:
  - `dir`: Directory containing the files to store.
  - `metadata`: Metadata associated with the documents.
  - `*args, **kwargs`: Additional arguments.

- **Output**: 
  Returns the number of documents successfully processed.

#### `simple-vector2`
Stores documents in the `second_collection` Chroma collection.

```python
@kitchen.storage("simple-vector2")
def simple_vector2(dir: str, metadata: dict = {}, *args, **kwargs):
    ...
```

- **Input Parameters**: Same as `simple-vector`.
- **Output**: 
  Returns the number of documents successfully processed.

---

### **2. Query Handlers**

#### `simple-query`
Handles natural language queries using **GPT-4** and the vector index.

```python
@kitchen.query("simple-query")
def simple_query(data: QuerySchema):
    ...
```

- **Input Parameters**:
  - `data`: A `QuerySchema` object with the following fields:
    - `query`: The query string.

- **Output**:
  - The response generated by GPT-4.

#### `non-ai`
A simple handler that returns a predefined message without relying on AI.

```python
@kitchen.query("non-ai")
def non_ai(data: QuerySchema):
    ...
```

- **Input Parameters**:
  - `data`: A `QuerySchema` object (though not used in this function).
  
- **Output**:
  - The message `"no AI is used in this function"`.

---

## Usage

### Running the Module
1. Start the KitchenAI application:
   ```bash
   kitchenai run
   ```

2. Use the provided handlers and storage methods via the API or KitchenAI CLI.

### Examples

#### Storing Documents
```bash
curl -X POST \
  -F "dir=/path/to/documents" \
  -F "metadata={'category': 'example'}" \
  http://localhost:8000/api/storage/simple-vector
```

#### Querying the Vector Index
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}' \
  http://localhost:8000/api/query/simple-query
```

#### Simple Non-AI Response
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "No AI example"}' \
  http://localhost:8000/api/query/non-ai
```

---

## How It Works

1. **Vector Storage**:
   - Uses the **LLAMA Parser** to load and preprocess documents.
   - Leverages **Chroma VectorStore** to store document embeddings.
   - Applies feature extractors to enrich the stored documents.

2. **Query Handlers**:
   - **`simple-query`**: Uses GPT-4 to process natural language queries and retrieve relevant results.
   - **`non-ai`**: Provides a simple, predefined response without relying on AI.

---

## Extending the Module

1. **Adding a New Storage Handler**:
   Define a new function decorated with `@kitchen.storage("<label>")`.

   Example:
   ```python
   @kitchen.storage("new-storage")
   def new_storage(dir: str, metadata: dict = {}, **kwargs):
       ...
   ```

2. **Adding a New Query Handler**:
   Define a new function decorated with `@kitchen.query("<label>")`.

   Example:
   ```python
   @kitchen.query("new-query")
   def new_query(data: QuerySchema):
       ...
   ```

---

## License

This module is licensed under the MIT License.

--- 

For further information, consult the KitchenAI documentation.