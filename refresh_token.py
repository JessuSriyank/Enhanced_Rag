import os
import json
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

# Path to your token.json file
token_path = '/home/dtel/Desktop/Grog/token.json'

# Initialize credentials
creds = None

# Load credentials from the token.json file
if os.path.exists(token_path):
    with open(token_path, 'r') as token_file:
        creds = Credentials.from_authorized_user_info(
            json.load(token_file),
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )

# Refresh the credentials if they are expired
if creds and creds.expired and creds.refresh_token:
    creds.refresh(Request())

# Save the refreshed credentials back to token.json
with open(token_path, 'w') as token_file:
    token_file.write(creds.to_json())

print("Credentials refreshed and saved successfully.")
