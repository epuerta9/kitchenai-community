from ninja import Schema
from kitchenai.contrib.kitchenai_sdk.kitchenai import KitchenAIApp

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.llms.openai import OpenAI
import os 
import chromadb
from llama_index.llms.openai import OpenAI
from kitchenai.contrib.kitchenai_sdk.storage.llama_parser import Parser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.objects import ObjectIndex
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor)

from toolhouse import Toolhouse, Provider

print(os.environ.get("KITCHENAI_DEBUG", None))

print(os.environ.get("CLOUDFLARE_API_KEY", None))
print(os.environ.get("CLOUDFLARE_API_BASE_URL", None))



# create client and a new collection
chroma_client = chromadb.PersistentClient(path="chroma_db")
chroma_collection = chroma_client.get_or_create_collection("quickstart")
chroma_collection_no_metadata = chroma_client.get_or_create_collection("quickstart_no_metadata")


function_llm = OpenAI(model="gpt-4",  api_base=os.environ.get("CLOUDFLARE_API_BASE_URL", None))
Settings.llm = OpenAI(model="gpt-4", api_base=os.environ.get("CLOUDFLARE_API_BASE_URL", None))
Settings.embed_model = OpenAIEmbedding(api_base=os.environ.get("CLOUDFLARE_API_BASE_URL", None))


# build summary index

class Query(Schema):
    query: str

kitchen = KitchenAIApp()


@kitchen.storage("file")
def chromadb_storage(dir: str, metadata: dict = {}, *args, **kwargs):
    """
    Store uploaded files into a vector store with metadata in two collections: quickstart and summary
    """
    chroma_collection = chroma_client.get_or_create_collection("quickstart")
    chroma_collection_no_metadata = chroma_client.get_or_create_collection("quickstart_no_metadata")
    parser = Parser(api_key=os.environ.get("LLAMA_CLOUD_API_KEY", None))

    response = parser.load(dir, metadata=metadata, **kwargs)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    vector_store_no_metadata = ChromaVectorStore(chroma_collection=chroma_collection_no_metadata)

    # set up ChromaVectorStore and load in data
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    storage_context_no_metadata = StorageContext.from_defaults(vector_store=vector_store_no_metadata)
            
    # quickstart index
    VectorStoreIndex.from_documents(
        response["documents"], storage_context=storage_context, show_progress=True,
            transformations=[TokenTextSplitter(), TitleExtractor(),QuestionsAnsweredExtractor()]
    )

    VectorStoreIndex.from_documents(
        response["documents"], storage_context=storage_context_no_metadata, show_progress=True,
    )

    
    return {"msg": "ok", "documents": len(response["documents"])}


@kitchen.query("agent")
def agent(request, query: Query):
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store
    )
    th = Toolhouse(provider=Provider.LLAMAINDEX)
    th.set_metadata("id", "llamaindex_agent")
    th.set_metadata("timezone", 0)


    vector_query_engine = index.as_query_engine(llm=Settings.llm)


    file_label = query.query

    label_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        filter={"file_label": file_label}
    )


    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="rfp_collection",
                description=(
                    "A catch all tool for any question."
                    "Use a detailed plain text question as input to the tool."
                ),
            ),
        ),
        QueryEngineTool(
            query_engine=label_index.as_query_engine(llm=Settings.llm),
            metadata=ToolMetadata(
                name=file_label,
                description=(
                    f"Provides information about the current working RFP: {file_label}"
                    "Use a detailed plain text question as input to the tool."
                ),
            ),
        ),
    ]

    local_agent = OpenAIAgent.from_tools(
        query_engine_tools,
        llm=function_llm,
        verbose=True,
        system_prompt=f"""\
    You are an expert in analyzing RFPs and breaking down their requirements from previous incoming RFP requests.
    You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
    """,
    )


    obj_index = ObjectIndex.from_objects(
        query_engine_tools + th.get_tools(bundle="llamaindex test"),
        # query_engine_tools,
        index_cls=VectorStoreIndex,
    )

    top_agent = OpenAIAgent.from_tools(
        tool_retriever=obj_index.as_retriever(similarity_top_k=3),
        system_prompt=""" \
    You are an RFP expert designed to help the proposal team understand the requirements of incoming RFPs.
    Please always use the tools provided to answer a question. Do not rely on prior knowledge.\
    """,
        verbose=True,
    )
    response = top_agent.chat(f"build a report on the requirements of the incoming RFP: {file_label}, include a list of requirements and a comparison to the advertising and consultancy rfps")

    return {"msg": response.response, "response_metadata": response.metadata, "response_source": response.source_nodes}

