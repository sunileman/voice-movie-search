import streamlit as st

openai_api_type = "azure"
openai_api_base = "https://xxxxx.openai.azure.com"
openai_api_sa_base = "https://xxxxx.openai.azure.com"
openai_api_version = "2023-08-01-preview"
openai_embedding_deployment_name = "ada-002"
openai_completion_deployment_name = "gpt35turbo"

azure_client_api_version="2023-07-01-preview"
azure_client_deployment_name="gpt-35-turbo-16k"


openai_completion_api_version = "2024-02-15-preview"
openai_completion_large_deployment_name = "gpt-4-32k"
#openai.api_key is stored in ./.streamlit/secrets.toml


#https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-text-to-speech?tabs=streaming#long-audio-api
# The service host suffix.
SERVICE_HOST = "xxxx.api.speech.microsoft.com"
NAME = "Movie avatar"
DESCRIPTION = "Movie avatar"
SUBSCRIPTION_KEY = st.secrets['speech_key']
SERVICE_REGION = st.secrets['speech_region']



byom_index_name = 'movies_inferred'
number_of_dims = 768
similarity = "cosine"
rate_throttle = .2
deleteExistingIndex = True
model='sentence-transformers__all-minilm-l6-v2'
elser_model=".elser_model_1"
vector_embedding_field = "text_embedding.predicted_value"
elser_embedding_field = "ml.tokens"

cache=None
