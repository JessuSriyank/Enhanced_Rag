"""import os
import json
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# Ensure the environment variable is set correctly
credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

if not credentials_path:
    raise FileNotFoundError("The environment variable GOOGLE_APPLICATION_CREDENTIALS is not set.")

# Load credentials from file
with open("/home/dtel/Desktop/Grog/client.json") as file:
    creds_data = json.load(file)

# Initialize the OAuth 2.0 flow
flow = InstalledAppFlow.from_client_config(creds_data, scopes=['https://www.googleapis.com/auth/cloud-platform'])

# Run the local server to get the authorization code
creds = flow.run_local_server(port=6060)

# Save the credentials for the next run
with open('token.json', 'w') as token:
    token.write(creds.to_json())"""
    
from google_auth_oauthlib.flow import InstalledAppFlow

# Path to your credentials.json file
credentials_path = '/home/dtel/Desktop/RAG/client.json'

# Path to save the token.json file
token_path = '/home/dtel/Desktop/RAG/token.json'

# OAuth2 flow
flow = InstalledAppFlow.from_client_secrets_file(
    credentials_path, 
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

# Run the flow to authorize and save the credentials
creds = flow.run_local_server(port=6060, access_type='offline', prompt='consent')

# Save the credentials to token.json
with open(token_path, 'w') as token:
    token.write(creds.to_json())

