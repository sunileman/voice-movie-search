##This app will read documents from elasticsearch index, read a field from the docs
##create a vector representation of that field using azure open ai ada002 model
##then store the new document in a new index.  The embedding will be stored in the field text_embedding.predicted_value

import sys

import openai
import time
from utils.es_helper import create_es_client
from variables import openai_embedding_deployment_name, rate_throttle
from variables import openai_api_type, openai_api_base, openai_api_version
import streamlit as st

from elasticsearch import helpers


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

def get_embedding(input_text):
    # Set API configuration
    openai.api_type = openai_api_type
    openai.api_base = openai_api_base
    openai.api_version = openai_api_version
    openai.azure_endpoint = openai_api_base

    # Set the GPT-3 API key
    openai.api_key = st.secrets['pass']


    response = openai.embeddings.create(input=[input_text], model=openai_embedding_deployment_name).data[0].embedding


    # Throttling as per configuration
    time.sleep(rate_throttle)

    return response




def get_documents_from_index(index_name):
    """
    Fetch all documents from the specified Elasticsearch index.
    """
    results = helpers.scan(es, index=index_name)
    return [doc for doc in results]

def add_document_with_embedding(index_name, source_field_name, document):
    """
    Add a new document to the specified Elasticsearch index with the embedding.
    """
    # Extract the original document data
    doc_data = document['_source']
    # Get the embedding for the 'combined_relevancy' field
    embedding = get_embedding(doc_data[source_field_name])
    # Add the embedding to the 'text_embedding.predicted_value' field
    doc_data['text_embedding'] = {"predicted_value": embedding}
    # Add the new document to the index
    es.index(index=index_name, body=doc_data)


def process_documents(source_index_name, source_field_name, target_index_name):
    # Fetch documents from the source index
    documents = get_documents_from_index(source_index_name)

    # For each document, fetch the embedding and add the new document to the target index
    for doc in documents:
        add_document_with_embedding(target_index_name, source_field_name, doc)


# Call the process_documents function
#uncomment to run.  It is commentted out as streamlit app will try and run this
#process_documents("ordercodes_processed", "combined_relevancy", "ordercodes_openai")