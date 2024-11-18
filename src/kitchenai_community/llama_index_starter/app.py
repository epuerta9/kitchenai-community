from kitchenai.contrib.kitchenai_sdk.kitchenai import KitchenAIApp

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from kitchenai.contrib.kitchenai_sdk.api import QuerySchema, EmbedSchema
from llama_index.llms.openai import OpenAI
import os 
import chromadb
from llama_index.llms.openai import OpenAI
from kitchenai.contrib.kitchenai_sdk.storage.llama_parser import Parser
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor)

from llama_index.core import Document

# create client and a new collection
chroma_client = chromadb.PersistentClient(path="chroma_db")
chroma_collection = chroma_client.get_or_create_collection("quickstart")
llm = OpenAI(model="gpt-4")

kitchen = KitchenAIApp()

import logging

logger = logging.getLogger(__name__)

@kitchen.storage("file")
def chromadb_storage(dir: str, metadata: dict = {}, *args, **kwargs):
    """
    Store uploaded files into a vector store with metadata
    """
    parser = Parser(api_key=os.environ.get("LLAMA_CLOUD_API_KEY", None))

    response = parser.load(dir, metadata=metadata, **kwargs)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # set up ChromaVectorStore and load in data
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
    # quickstart index
    VectorStoreIndex.from_documents(
        response["documents"], storage_context=storage_context, show_progress=True,
            transformations=[TokenTextSplitter(), TitleExtractor(),QuestionsAnsweredExtractor()]
    )
    
    return {"msg": "ok", "documents": len(response["documents"])}


@kitchen.embed("embed")
def embed(instance, metadata: dict = {}):
    """Embed single pieces of text"""
    documents = [Document(text=instance.text)]
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
    VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True,
            transformations=[TokenTextSplitter(), TitleExtractor(),QuestionsAnsweredExtractor()]
    )
    return "ok"


@kitchen.query("simple-query")
def query(request, data: QuerySchema):
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
    )
    query_engine = index.as_query_engine(chat_mode="best", llm=llm, verbose=True)
    response = query_engine.aquery(data.query)

    return {"msg": response.response}