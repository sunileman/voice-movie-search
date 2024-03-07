import sys

import streamlit as st
from elasticsearch_llm_cache.elasticsearch_llm_cache import ElasticsearchLLMCache

import variables
from speech_recognition_package.speech_recognition import microphone_to_es_with_avatar
from utils.es_helper import create_es_client
from utils.openai_helper import ini_chat_prompts


# Connect to Elasticsearch
try:
    username = st.secrets['es_username']
    password = st.secrets['es_password']
    cloudid = st.secrets['es_cloudid']
    es = create_es_client(username, password, cloudid)
except Exception as e:
    print("Connection failed", str(e))
    st.error("Error connecting to Elasticsearch. Fix connection and restart app")
    sys.exit(1)

# Init Elasticsearch Cache
variables.cache = ElasticsearchLLMCache(
    es_client=es,
    index_name="llm_cache",
    create_index=False,  # setting only because of Streamlit behavor
)
print(f"_creating Elasticsearch Cache_")

ini_chat_prompts()

# Only want to attempt to create the index on first run
if "index_created" not in st.session_state:
    print("_running create_index_")
    variables.cache.create_index(768)
    # Set the flag so it doesn't run every time
    st.session_state.index_created = True
else:
    print("_index already created, skipping_")

def main():


    # Using the sidebar for the speech button
    c1 = st.container()


    if st.sidebar.button('🎙 Start', key="start", help="Start Speech Recognition"):
        microphone_to_es_with_avatar()

    # Use the toggle and set its value based on the session state
    cachetoggle = st.sidebar.toggle('Activate feature', st.session_state['toggle_state'])

    # Update the session state with the new state of the toggle
    st.session_state['toggle_state'] = cachetoggle




    # Custom CSS to create the TV screen look
    c1.markdown("""
    <style>
    /* Custom styles for the main content area */
    .block-container {
        padding-top: 50px; /* Increase top padding */
        padding-left: 5%; /* Increase left padding */
        padding-right: 5%; /* Increase right padding */
        border: 10px solid black; /* Black border for TV screen effect */
        border-radius: 20px; /* Rounded corners for the TV screen */
        background-color: #333; /* Dark background simulating the off screen */
    }
    /* Adjust the main title style to fit the TV screen theme */
    h1 {
        color: white; /* White text color */
        text-align: center; /* Center alignment */
    }
    </style>
    """, unsafe_allow_html=True)

    # Placeholder content inside the "TV screen"
    c1.markdown("""
    <div style="color: white; text-align: center;">
        <h2>Welcome to Movie Search</h2>
        <p>Press the "🎙 Start" button in the sidebar to begin.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
