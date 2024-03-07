import base64
import os
import sys
import logging
import azure.cognitiveservices.speech as speechsdk
import streamlit as st
from openai.lib.azure import AzureOpenAI
import time

import variables
from avatar.avatar_helper import submit_synthesis, get_synthesis
from utils.es_helper import create_es_client
from utils.openai_helper import get_chat_guidance, get_chat_guidance_summarized, add_to_cache
from utils.query_helper import  search_products_v2
from variables import openai_api_sa_base, openai_completion_api_version

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

azureclient = AzureOpenAI(
    api_key=st.secrets['sa_pass'],
    api_version=openai_completion_api_version,
    azure_endpoint=openai_api_sa_base
)

if 'video_displayed' not in st.session_state:
    st.session_state.video_displayed = 'first'  # Default to showing the first video

logger = logging.getLogger(__name__)



def recognize_from_microphone():
    speech_config = speechsdk.SpeechConfig(subscription=st.secrets['speech_key'],
                                           region=st.secrets['speech_region'])
    speech_config.speech_recognition_language = "en-US"

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    st.write("Speak into your microphone.")
    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        st.success("Recognized: {}".format(speech_recognition_result.text))
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        st.error("No speech could be recognized.")
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        st.error(
            f"Speech Recognition canceled: {cancellation_details.reason}. Error details: {cancellation_details.error_details}")



def microphone_to_es():
    speech_config = speechsdk.SpeechConfig(subscription=st.secrets['speech_key'],
                                           region=st.secrets['speech_region'])
    speech_config.speech_recognition_language = "en-US"

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    # st.write("Speak into your microphone.")
    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    # Initialize user_query as an empty string in case recognition fails
    user_query = ""

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        user_query = speech_recognition_result.text  # Store the recognized text in user_query

        c = st.container()

        titles = search_products_v2(es, user_query, 'Elser', 1, 200)
        st.session_state.messages = []

        st.session_state.messages.append(
            {"role": "system",
             "content": "You are an AI assistant for Movies.  I supply you with movie titles and you answer questins based on the movie titles I provided. Give me a response for each title I supply. Give me a brief description for each movie."
                        "Include IMDB rating for each movie. Also pretend you have TV guide information for the month of March 2024 and add some information about what channel the move will play on with time, and include that in the table. Lastly your responses should be in markdown format using table format. Make sure the description is short per movie and fits within single column"})

        st.session_state.messages.append(
            {"role": "user", "content": f"Answer this question: {user_query} with the following movies  {titles}"})
        response_text = get_chat_guidance(azureclient)

        c.success("Question: {}".format(user_query))
        c.markdown(response_text, unsafe_allow_html=True)

    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        st.error("No speech could be recognized.")
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        st.error(
            f"Speech Recognition canceled: {cancellation_details.reason}. Error details: {cancellation_details.error_details}")

    # Return the user_query variable to use it outside the function
    return user_query


def build_avatar_response(user_query, titles):
    if st.session_state['toggle_state'] and "cage" in user_query.lower():
        print("build_avatar_response cache_toggle_state and cage")
        st.video(
            'https://sunmanapp.blob.core.windows.net/publicstuff/movies/nick-cage-movies.mp4',
            format="video/mp4", start_time=0)
        return 0
    elif st.session_state['toggle_state'] and "date" in user_query.lower():
        st.video(
            'https://sunmanapp.blob.core.windows.net/publicstuff/movies/movie-release-dates.mp4',
            format="video/mp4", start_time=0)
        return 0
    elif st.session_state['toggle_state'] and ("length" in user_query.lower() or "runtime" in user_query.lower()):
        st.video(
            'https://sunmanapp.blob.core.windows.net/publicstuff/movies/movie-length.mp4',
            format="video/mp4", start_time=0)
        return 0

    if (st.session_state.ini_engage == True):
        st.session_state.avatar_messages.append(
            {"role": "user", "content": f"Provide me short description for each of the following movies {titles}"})
        st.session_state.ini_engage = False
    else:
        st.session_state.avatar_messages.append(
            {"role": "user", "content": f"{user_query}"})

    # # Print the messages array for debugging or logging purposes
    print("build_avatar: Messages being sent to Azure OpenAI:")
    for message in st.session_state.avatar_messages:
        print(f"{message['role'].title()}: {message['content']}")

    avatar_response = get_chat_guidance_summarized(azureclient)

    st.session_state.avatar_messages.append(
        {"role": "system", "content": avatar_response})

    print("build_avatar: Azure OpenAI Response: " + avatar_response)

    with st.spinner('Avatar generation....'):
        job_id = submit_synthesis(avatar_response)
        if job_id is not None:
            while True:
                status, url = get_synthesis(job_id)
                if status == 'Succeeded':
                    print('batch avatar synthesis job succeeded')
                    logger.info('batch avatar synthesis job succeeded')
                    st.video(
                        url,
                        format="video/mp4", start_time=0)
                    break
                elif status == 'Failed':
                    print('batch avatar synthesis job failed')
                    logger.error('batch avatar synthesis job failed')
                    break
                else:
                    print(f'batch avatar synthesis job is still running, status [{status}]')
                    logger.info(f'batch avatar synthesis job is still running, status [{status}]')
                    time.sleep(5)


def speak2text():
    speech_config = speechsdk.SpeechConfig(subscription=st.secrets['speech_key'],
                                           region=st.secrets['speech_region'])
    speech_config.speech_recognition_language = "en-US"

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    # st.write("Speak into your microphone.")
    return speech_recognizer.recognize_once_async().get()


def microphone_to_es_with_avatar():
    speech_recognition_result = speak2text()

    # Initialize user_query as an empty string in case recognition fails
    user_query = ""

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        user_query = speech_recognition_result.text  # Store the recognized text in user_query

        c = st.container()
        c.success("Question: {}".format(user_query))

        if st.session_state.ini_engage:
            print("Calling search_products_v2")
            search_products_v2(es, user_query, 'Elser', 1, 200)
            st.session_state.messages.append(
                {"role": "user", "content": f"Answer this question: {user_query} with the following movies  {st.session_state.titles}"})

            # Query the cache
            cache_response = variables.cache.query(prompt_text=user_query, similarity_threshold=0.6)

            if cache_response:
                print("response from cache")
                c.markdown(cache_response["response"][0], unsafe_allow_html=True)
            else:
                st.session_state.genAIResponse = get_chat_guidance(azureclient)
                add_to_cache(variables.cache, user_query, st.session_state.genAIResponse)
                c.markdown(st.session_state.genAIResponse, unsafe_allow_html=True)

            #st.session_state.genAIResponse = get_chat_guidance(azureclient)

            #add_to_cache(variables.cache, user_query, st.session_state.genAIResponse)

            #c.markdown(st.session_state.genAIResponse, unsafe_allow_html=True)

            st.session_state.ini_engage = False
        else:
            st.session_state.messages.append(
                {"role": "user", "content": f"{user_query}"})

            # Query the cache
            cache_response = variables.cache.query(prompt_text=user_query, similarity_threshold=0.6)

            if cache_response:
                print("response from cache")
                c.markdown(cache_response["response"][0], unsafe_allow_html=True)
            else:
                st.session_state.genAIResponse = get_chat_guidance(azureclient)
                add_to_cache(variables.cache, user_query, st.session_state.genAIResponse)
                c.markdown(st.session_state.genAIResponse, unsafe_allow_html=True)

            #get_chat_guidance(azureclient)

            #add_to_cache(variables.cache, user_query, st.session_state.genAIResponse)

            #c.markdown(st.session_state.genAIResponse, unsafe_allow_html=True)
        build_avatar_response(user_query, st.session_state.titles)
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        st.error("No speech could be recognized.")
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        st.error(
            f"Speech Recognition canceled: {cancellation_details.reason}. Error details: {cancellation_details.error_details}")

    # Return the user_query variable to use it outside the function
    return user_query
