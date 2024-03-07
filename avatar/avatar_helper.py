import json
import logging
import sys

import requests

from variables import SERVICE_HOST, NAME, DESCRIPTION, SERVICE_REGION, SUBSCRIPTION_KEY

logging.basicConfig(stream=sys.stdout, level=logging.INFO,  # set to logging.DEBUG for verbose output
        format="[%(asctime)s] %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p %Z")
logger = logging.getLogger(__name__)



def submit_synthesis(response_text):
    url = f'https://{SERVICE_REGION}.{SERVICE_HOST}/api/texttospeech/3.1-preview1/batchsynthesis/talkingavatar'
    header = {
        'Ocp-Apim-Subscription-Key': SUBSCRIPTION_KEY,
        'Content-Type': 'application/json'
    }

    payload = {
        'displayName': NAME,
        'description': DESCRIPTION,
        "textType": "PlainText",
        'synthesisConfig': {
            "voice": "en-US-JennyNeural",
        },
        # Replace with your custom voice name and deployment ID if you want to use custom voice.
        # Multiple voices are supported, the mixture of custom voices and platform voices is allowed.
        # Invalid voice name or deployment ID will be rejected.
        'customVoices': {
            # "YOUR_CUSTOM_VOICE_NAME": "YOUR_CUSTOM_VOICE_ID"
        },
        "inputs": [
            {
                "text": response_text,
            },
        ],
        "properties": {
            "customized": False, # set to True if you want to use customized avatar
            "talkingAvatarCharacter": "lisa",  # talking avatar character
            "talkingAvatarStyle": "casual-sitting",  # talking avatar style, required for prebuilt avatar, optional for custom avatar, https://learn.microsoft.com/en-us/azure/ai-services/speech-service/text-to-speech-avatar/avatar-gestures-with-ssml
            "videoFormat": "mp4",  # mp4 or webm, webm is required for transparent background
            "videoCodec": "h264",  # hevc, h264 or vp9, vp9 is required for transparent background; default is hevc
            "subtitleType": "soft_embedded",
            "backgroundColor": "#FFFFFFFF", # background color in RGBA format, default is white; can be set to 'transparent' for transparent background
        }
    }

    response = requests.post(url, json.dumps(payload), headers=header)
    if response.status_code < 400:
        print('Batch avatar synthesis job submitted successfully')
        logger.info('Batch avatar synthesis job submitted successfully')
        print(f'Job ID: {response.json()["id"]}')
        logger.info(f'Job ID: {response.json()["id"]}')
        return response.json()["id"]
    else:
        logger.error(f'Failed to submit batch avatar synthesis job: {response.text}')


def get_synthesis(job_id):
    url = f'https://{SERVICE_REGION}.{SERVICE_HOST}/api/texttospeech/3.1-preview1/batchsynthesis/talkingavatar/{job_id}'
    header = {
        'Ocp-Apim-Subscription-Key': SUBSCRIPTION_KEY
    }
    response = requests.get(url, headers=header, timeout=120)
    if response.status_code < 400:
        url = None
        print('Get batch synthesis job successfully')
        logger.debug('Get batch synthesis job successfully')
        print(response.json())
        logger.debug(response.json())
        if response.json()['status'] == 'Succeeded':
            print(f'Batch synthesis job succeeded, download URL: {response.json()["outputs"]["result"]}')
            logger.info(f'Batch synthesis job succeeded, download URL: {response.json()["outputs"]["result"]}')
            url = response.json()["outputs"]["result"]
        return response.json()['status'], url
    else:
        print(f'Failed to get batch synthesis job: {response.text}')
        logger.error(f'Failed to get batch synthesis job: {response.text}')

def list_synthesis_jobs(skip: int = 0, top: int = 100):
    """List all batch synthesis jobs in the subscription"""
    url = f'https://{SERVICE_REGION}.{SERVICE_HOST}/api/texttospeech/3.1-preview1/batchsynthesis/talkingavatar?skip={skip}&top={top}'
    header = {
        'Ocp-Apim-Subscription-Key': SUBSCRIPTION_KEY
    }
    response = requests.get(url, headers=header)
    if response.status_code < 400:
        print(f'List batch synthesis jobs successfully, got {len(response.json()["values"])} jobs')
        logger.info(f'List batch synthesis jobs successfully, got {len(response.json()["values"])} jobs')
        print(f'Failed to get batch synthesis job: {response.text}')
        logger.info(f'Failed to get batch synthesis job: {response.text}')
    else:
        print(f'Failed to list batch synthesis jobs: {response.text}')
        logger.error(f'Failed to list batch synthesis jobs: {response.text}')

