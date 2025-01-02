import os
from google.oauth2 import service_account

# Path to your credentials.json file
credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

if not credentials_path:
    raise FileNotFoundError("The environment variable GOOGLE_APPLICATION_CREDENTIALS is not set.")

# Load credentials from file
credentials = service_account.Credentials.from_service_account_file(credentials_path)