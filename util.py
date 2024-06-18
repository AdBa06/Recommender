import os
import pandas as pd
from dotenv import load_dotenv
from dotenv import dotenv_values
from langchain.chat_models import AzureChatOpenAI

# Access environment variables
config = dotenv_values('openai.env')

# Check if config has all required keys
required_keys = ['AZURE_API_TYPE', 'AZURE_API_BASE', 'AZURE_API_VERSION', 'AZURE_API_KEY']
for key in required_keys:
    if key not in config:
        raise ValueError(f"Missing required key in config: {key}")

# If all keys are present, print the config
print("Configuration loaded successfully:")
print(config)
# __________________Read in CSV file and ensure match________________________________________
def read_csv_file(file_path='to_llm.csv'):
    try:
        df = pd.read_csv(file_path)
        print("CSV file read successfully!")
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# # Test the function
# if __name__ == "__main__":
#     df = read_csv_file()
#     if df is not None:
#         # Example usage of AzureChatOpenAI
#         chat = AzureChatOpenAI(
#             api_key=api_key,
#             api_base=api_base,
#             deployment_id="YOUR_DEPLOYMENT_ID"  # Replace with your actual deployment ID
#         )

#         response = chat("Hello, how can I help you?")
#         print(response)