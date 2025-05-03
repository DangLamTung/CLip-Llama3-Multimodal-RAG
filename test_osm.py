import streamlit as st
import folium
from streamlit_folium import st_folium
import osmnx as ox
import networkx as nx
from geopy.geocoders import Nominatim

# Set page title
st.title("🚗 OpenStreetMap Routing with Streamlit")

# Initialize geolocator
geolocator = Nominatim(user_agent="streamlit-osm")

# Sidebar for user input
st.sidebar.header("Enter Locations")

start_location = st.sidebar.text_input("Start Location", "San Francisco, CA")
end_location = st.sidebar.text_input("Destination", "Los Angeles, CA")

if st.sidebar.button("Find Route"):
    # Convert addresses to coordinates
    try:
        start_coords = geolocator.geocode(start_location)
        end_coords = geolocator.geocode(end_location)

        if start_coords and end_coords:
            # Get latitude and longitude
            start_point = (start_coords.latitude, start_coords.longitude)
            end_point = (end_coords.latitude, end_coords.longitude)

            # Load the street network graph
            G = ox.graph_from_point(start_point, dist=50000, network_type="drive")

            # Find the nearest nodes in the graph
            orig_node = ox.nearest_nodes(G, start_point[1], start_point[0])
            dest_node = ox.nearest_nodes(G, end_point[1], end_point[0])

            # Find the shortest path
            route = nx.shortest_path(G, orig_node, dest_node, weight="length")

            # Convert route to lat/lon points
            route_map = ox.plot_route_folium(G, route, route_color="blue")

            # Add markers
            folium.Marker(start_point, tooltip="Start", icon=folium.Icon(color="green")).add_to(route_map)
            folium.Marker(end_point, tooltip="Destination", icon=folium.Icon(color="red")).add_to(route_map)

            # Display map
            st_folium(route_map, width=800, height=500)

        else:
            st.error("Could not find one or both locations. Try a different address.")

    except Exception as e:
        st.error(f"Error: {e}")
