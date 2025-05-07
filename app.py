import streamlit as st
import folium
import osmnx
import networkx as nx
import numpy as np
import pandas as pd
import leafmap.foliumap as leafmap
from streamlit_folium import st_folium
from folium.plugins import Draw
from RAG_websearch import *

from geopy.geocoders import Nominatim
from opendeepsearch import OpenDeepSearchTool
from smolagents import CodeAgent, LiteLLMModel
import os


# from llama_index.core.response.notebook_utils import display_source_node
# from llama_index.core.schema import ImageNode
# from llama_index.llms.ollama import Ollama

# # load documents
# import chromadb
# from llama_index.core.indices import MultiModalVectorStoreIndex
# from llama_index.vector_stores.chroma import ChromaVectorStore
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
# from llama_index.core import StorageContext, load_index_from_storage
# from chromadb.utils.data_loaders import ImageLoader
# from llama_index.core import Settings
# # load from disk
# # db2 = chromadb.PersistentClient(path="./chroma_db")
# # chroma_collection = db2.get_or_create_collection("quickstart")
# # vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# storage_context = StorageContext.from_defaults(persist_dir="./travel")
# index = load_index_from_storage(
#     storage_context=storage_context
# )




# retriever_engine = index.as_retriever(
#     similarity_top_k=5, image_similarity_top_k=5
# )

# # Set ollama model
# Settings.llm = Ollama(model="gemma3:4b", request_timeout=120.0)

# llm = Ollama(model="gemma3:4b", request_timeout=120.0)

# def retrieve(retriever_engine, query_str):
#     retrieval_results = retriever_engine.retrieve(query_str)

#     retrieved_image = []
#     retrieved_text = []
#     for res_node in retrieval_results:
#         if isinstance(res_node.node, ImageNode):
#             retrieved_image.append(res_node.node.metadata["file_path"])
#         else:
#             display_source_node(res_node, source_length=200)
#             retrieved_text.append(res_node.text)

#     return retrieved_image, retrieved_text


# query_str = """
# who is batman
# """
# from llama_index.core import PromptTemplate
# template = (
#     "Imagine tour guide "
#     "you are guiding some tourist on a motorcycle trip."
#     "Here is some context from the motor trips "
#     "resume related to the query: \n"
#     "-----------------------------------------\n"
#     "{context_str}\n"
#     "-----------------------------------------\n"
#     "Considering the above information, "
#     "Please respond to the following inquiry:\n\n"
#     "Question: {query_str}\n\n"
#     "Answer succinctly and ensure your response is "
#     "truth, based on the fact stated in the context."
 
# )
# img, txt = retrieve(retriever_engine=retriever_engine, query_str=query_str)
# qa_template = PromptTemplate(template)
 
# query_engine = index.as_query_engine(llm =llm  )



BASEMAPS = ['Satellite', 'Roadmap', 'Terrain', 'Hybrid', 'OpenStreetMap']
TRAVEL_MODE = ['Drive', 'Walk', 'Bike']
TRAVEL_OPTIMIZER = ['Dijkstra', 'Bellman-Ford', 'Floyd Warshall' ]
ADDRESS_DEFAULT = "[10.7724,106.65922]"


# Set environment variables for API keys


search_agent = OpenDeepSearchTool(model_name="openrouter/google/gemini-2.5-pro-exp-03-25:free", reranker="jina") # Set reranker to "jina" or "infinity"
model = LiteLLMModel(
    "openrouter/google/gemini-2.0-flash-001",
    temperature=0.2
)


locator = Nominatim(user_agent = "myapp")
# location = locator.geocode(address)
   

# Create a session variable
if 'comparing' not in st.session_state:
    st.session_state['comparing'] = False

if '3short' not in st.session_state:
    st.session_state['3short'] = False
 #path folder of the data file


def clear_text():
    st.session_state["go_from"] = ""
    st.session_state["go_to"] = ""
@st.cache_data
def compare_algo():

    return  

@st.cache_data
def short_algo():

    return 

    

st.set_page_config(page_title="🚋 Route finder", layout="wide")

# ====== SIDEBAR ======
with st.sidebar:

 
    st.title("LLM search for travel lover")

    st.markdown("A simple app that perform LLM RAG search on a map.")

    basemap = st.selectbox("Choose basemap", BASEMAPS)
    if basemap in BASEMAPS[:-1]:
        basemap=basemap.upper()

    transport = st.selectbox("Choose transport", TRAVEL_MODE)

    question = st.text_input("Chat", key="go_from")


    btn = st.button('Chat')


    if(btn):
        

        # Step 1: Retrieve information using DuckDuckGo
        retrieved_texts = search_duckduckgo(question)
        print(retrieved_texts[0] )
        if not retrieved_texts:
            print("No relevant information found.")
            retrieved_texts = []
            #return
        
        # Step 2: Generate a response using the retrieved information and ChatGPT
        response = query_engine.query(question)

        print(response)

        # response = generate_response_with_rag(question, retrieved_texts)
        print(response)
    btn1 = st.button('Find Intersting thing (RAG)')
    btn1 = st.button('Find Intersting thing (WebSearch)')

    if(btn1):
 
        location = locator.geocode(st.session_state.marker_location)
        print(location)


        code_agent = CodeAgent(tools=[search_agent], model=model)
        query = "Địa điểm này và khu vực lân cận có điều gì thú vị cho khách du lịch, về ẩm thực, văn hóa, phong cảnh " + str(location)
        result = code_agent.run(query)
        st.write(result)

        # print(location)
 



    # st.info(
    #     "This is an open source project and you are very welcome to contribute your "
    #     "comments, questions, resources and apps as "
    #     "[issues](https://github.com/maxmarkov/streamlit-navigator/issues) or "
    #     "[pull requests](https://github.com/maxmarkov/streamlit-navigator/pulls) "
    #     "to the [source code](https://github.com/maxmarkov/streamlit-navigator). "
    # )




# ====== MAIN PAGE ======

# Initialize session state to store marker location
if "marker_location" not in st.session_state:
    st.session_state.marker_location = [10.7724,106.65922]  # Default location
    st.session_state.zoom = 11  # Default zoom


lat, lon = st.session_state.marker_location

m = leafmap.Map(center=(lat, lon), zoom=16)

m.add_basemap(basemap)


# Function to get position from click coordinates
def get_pos(lat, lng):
    return lat, lng



# Add a marker at the current location in session state
marker = folium.Marker(
    location=st.session_state.marker_location,
    draggable=False
)
marker.add_to(m)

# Render the map and capture clicks
# map = st_folium(m, width=620, height=580, key="folium_map")

data = None
route =  []


map = st_folium(m, height=800, width=1400)
# m.to_streamlit()

# Render the map and capture clicks
# map = st_folium(m, width=620, height=580, key="folium_map")

# Update marker position immediately after each click
if map.get("last_clicked"):
    lat, lng = map["last_clicked"]["lat"], map["last_clicked"]["lng"]
    st.session_state.marker_location = [lat, lng]  # Update session state with new marker location
    st.session_state.zoom = map["zoom"]
    # Redraw the map immediately with the new marker location
    m = folium.Map(location=st.session_state.marker_location, zoom_start=st.session_state.zoom)
    folium.Marker(
        location=st.session_state.marker_location,
        draggable=False
    ).add_to(m)
    map = st_folium(m, height=800, width=1400)

# Display coordinates
st.write(f"Coordinates: {st.session_state.marker_location}")
