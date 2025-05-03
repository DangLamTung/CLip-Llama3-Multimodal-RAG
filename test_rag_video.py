
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import SimpleDirectoryReader, StorageContext, ServiceContext

from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.lancedb import LanceDBVectorStore


from llama_index.core import SimpleDirectoryReader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction


import chromadb
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import SimpleDirectoryReader, StorageContext
from chromadb.utils.data_loaders import ImageLoader


from llama_index.core.indices import MultiModalVectorStoreIndex

from llama_index.core import SimpleDirectoryReader, StorageContext
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# set defalut text and image embedding functions
embedding_function = OpenCLIPEmbeddingFunction()
image_loader = ImageLoader()





# create client and a new collection
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection(
    "multimodal_collection",
    embedding_function=embedding_function,
    data_loader=image_loader,
)
from llama_index.core import Settings

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
documents = SimpleDirectoryReader("./data_wiki/").load_data()
print(documents)
# set up ChromaVectorStore and load in data
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)
index.storage_context.persist(persist_dir="./travel")

# load documents
import chromadb
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import SimpleDirectoryReader, StorageContext
from chromadb.utils.data_loaders import ImageLoader
# load from disk
db2 = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db2.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)





retriever_engine = index.as_retriever(
    similarity_top_k=5, image_similarity_top_k=5
)


import json

# metadata_str = json.dumps(metadata_vid)

qa_tmpl_str = (
    "Given the provided information, including relevant images and retrieved context from the video, \
 accurately and precisely answer the query without any additional prior knowledge.\n"
    "Please ensure honesty and responsibility, refraining from any racist or sexist remarks.\n"
    "---------------------\n"
    "Context: {context_str}\n"
    # "Metadata for video: {metadata_str} \n"
    "---------------------\n"
    "Query: {query_str}\n"
    "Answer: "
)


from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.schema import ImageNode
from llama_index.llms.ollama import Ollama


# Set ollama model
Settings.llm = Ollama(model="gemma3:4b", request_timeout=120.0)

llm = Ollama(model="gemma3:4b", request_timeout=120.0)

def retrieve(retriever_engine, query_str):
    retrieval_results = retriever_engine.retrieve(query_str)

    retrieved_image = []
    retrieved_text = []
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_image.append(res_node.node.metadata["file_path"])
        else:
            display_source_node(res_node, source_length=200)
            retrieved_text.append(res_node.text)

    return retrieved_image, retrieved_text


query_str = """
who is batman
"""
from llama_index.core import PromptTemplate
template = (
    "Imagine you are a insurance's assistant and "
    "you answer a recruiter's questions about the  insurance's policy."
    "Here is some context from the insurance's "
    "resume related to the query::\n"
    "-----------------------------------------\n"
    "{context_str}\n"
    "-----------------------------------------\n"
    "Considering the above information, "
    "Please respond to the following inquiry:\n\n"
    "Question: {query_str}\n\n"
    "Answer succinctly and ensure your response is "
    "truth, based on the fact stated in the context."
 
)
img, txt = retrieve(retriever_engine=retriever_engine, query_str=query_str)
qa_template = PromptTemplate(template)
print("retrieved text", txt)

# # Query Data
# import ollama
# response = ollama.chat(model='gemma3:4b', 
#     messages=[{
#         'role': 'user', 
#         'content': 'Describe the image',
#     }],
#     # options={"temperature":0.7}
#     )
# print(response)

 # Query Data
query_engine = index.as_query_engine(llm =llm  )
response = query_engine.query("can llama 2 calculate 1 +1  ")

print(response)