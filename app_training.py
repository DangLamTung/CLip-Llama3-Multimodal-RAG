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


from litellm import completion
import json




BASEMAPS = ['Satellite', 'Roadmap', 'Terrain', 'Hybrid', 'OpenStreetMap']
TRAVEL_MODE = ['Drive', 'Walk', 'Bike']
TRAVEL_OPTIMIZER = ['Dijkstra', 'Bellman-Ford', 'Floyd Warshall' ]
ADDRESS_DEFAULT = "[10.7724,106.65922]"


# Set environment variables for API keys

os.environ['GEMINI_API_KEY'] = GOOGLE_API_KEY



search_agent = OpenDeepSearchTool(model_name="gemini/gemini-2.0-flash-001", reranker="jina") # Set reranker to "jina" or "infinity"
model = LiteLLMModel(
    "gemini/gemini-2.0-flash-001",
    temperature=0.2
)

location = []
location_coord = []
locator = Nominatim(user_agent = "myapp")
# location = locator.geocode(address)
response = None

# Create a session variable
if 'comparing' not in st.session_state:
    st.session_state['comparing'] = False

if '3short' not in st.session_state:
    st.session_state['3short'] = False
 #path folder of the data file

if 'location' not in st.session_state:
    st.session_state['location'] = []


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

 
    st.title("Planning and Ingest Data for RAG")

    st.markdown("A simple app that planning trip, create MultiModal RAG database.")

    basemap = st.selectbox("Choose basemap", BASEMAPS)
    if basemap in BASEMAPS[:-1]:
        basemap=basemap.upper()

    transport = st.selectbox("Choose transport", TRAVEL_MODE)

    question = st.text_input("Chat", key="go_from")


    btn = st.button('Prompting your requirement')


    if(btn):
        

        # Step 1: Retrieve information using DuckDuckGo
        retrieved_texts = search_duckduckgo("Địa điểm du lịch nổi tiếng ở gần"+ question)
        print(retrieved_texts[0] )
        if not retrieved_texts:
            print("No relevant information found.")
            retrieved_texts = []
            #return
        
        # Step 2: Generate a response using the retrieved information and ChatGPT
        # response = generate_response_with_rag(question, retrieved_texts)
        # print(response)
    btn1 = st.button('Plan your trip')


    if(btn1):
        compare_algo()
        full_trip =  ""
        for loc in st.session_state.location:
            location = locator.geocode(loc)
            st.write("Planning for this location:" + str(location)  + " ....")
             
            
            retrieved_texts = search_duckduckgo("Địa điểm du lịch nổi tiếng ở gần"+ str(location))
            st.write("Crawling....")
            st.write("Seeing the web: " +retrieved_texts[0])
            code_agent = CodeAgent(tools=[search_agent], model=model)
            query = "Địa điểm này và khu vực lân cận có điều gì thú vị cho khách du lịch, về ẩm thực, văn hóa, phong cảnh," \
                     "lên kế hoạch tham quan khu vực này, nối tiếp chuyến đi với địa điểm trước" \
                     "Tìm các nhà hàng, khách sạn được đánh giá cao nhất trong khu vực với giá cả hợp lý và " \
                     "lưu địa chỉ tọa độ vào câu trả lời" \
                     "TÌm kiếm khoảng 3 nhà hàng và 3 khách sạn " + str(location)
            result = code_agent.run(query)
            print(result)
            full_trip += str(result)


            # full_trip
        code_agent = CodeAgent(tools=[search_agent], model=model)
        query = "Dựa trên những  thông tin  đã thu thập, lên kế hoạch một chuyến phượt từ địa địa điểm đầu tiên đến cuối, đi bằng xe máy, " \
        "đảm bảo 1 ngày di  chuyển không quá " \
        "300km, giữ chi tiết phương thức di chuyển và các  địa điểm thú vị trong chuyến đi gắn kèm," \
        "tọa  độ và đánh giá đã thu được tử các bước tìm kiếm" \
        "lưu địa chỉ tọa độ vào câu trả lời"  + full_trip


        response_dict = completion(
            "gemini/gemini-2.0-flash-001",
            temperature=0.2,
            messages=[{ "content":query,"role": "user"}]
            )
                    

        response = response_dict.choices[0].message.content
        # result = model.run(query)
        # print(response)
        # print(response.choices[0].message.content)
        st.write("Have full trip, Proceessing to Database, see you space cowboy....")

        st.empty()
        # retrieved_texts = search_duckduckgo(location)
        # # print(retrieved_texts[0] )
        # if not retrieved_texts:
        #     print("No relevant information found.")
        #     retrieved_texts = []
        #     #return
        
        # # Step 2: Generate a response using the retrieved information and ChatGPT
        # response = generate_response_with_rag("Find interesting feature in this area, can be interesting landscape, exotic  food, or place to visit", location)
        

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
    # Create a Folium map

@st.cache_data   
def get_pos(lat, lng):
    location_coord.append([lat, lng])
    location.append([lat, lng])
    st.session_state.location.append([lat, lng])
    return lat, lng
m = folium.Map(location=[10.7724,106.65922], zoom_start=16)
m.add_child(folium.ClickForMarker())

    # When the user interacts with the map
map = st_folium(
        m,
        width=1400, height=800,
        key="folium_map"
    )
data = None
if map.get("last_clicked"):
    data = get_pos(map["last_clicked"]["lat"], map["last_clicked"]["lng"])

if response is not None:
    st.write(response)
    with open("road_trip.txt", "w") as f:
        f.write(response)
