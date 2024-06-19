import os
import pandas as pd
from dotenv import load_dotenv
from dotenv import dotenv_values
from langchain_openai import AzureChatOpenAI
from sentence_transformers import SentenceTransformer, util
import torch
import openai
from openai import OpenAI
from openai import AzureOpenAI


#__________________________________________Access OpenAI API________________________________________________
# Function to retrieve Azure API configuration from environment
def get_llm_api_config_details():
    # Access environment variables from openai.env
    config = dotenv_values('openai.env')

    # Check if config has all required keys
    required_keys = ['AZURE_API_TYPE', 'AZURE_API_BASE', 'AZURE_API_VERSION', 'AZURE_API_KEY']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in config: {key}")

    # Print loaded configuration (optional)
    print("Configuration loaded successfully:")
    return config

# Function to initialize Azure ChatOpenAI instance
def get_llm_instance(model='gpt-35-turbo'):   
    config = get_llm_api_config_details()
  
    # Initialize AzureChatOpenAI instance
    llm = AzureChatOpenAI(
        deployment_name=model, 
        model_name=model,
        openai_api_key=config['AZURE_API_KEY'],
        temperature=0, 
        # max_tokens=5000,
        # n=1, 
        azure_endpoint=config['AZURE_API_BASE'],
        openai_api_version=config['AZURE_API_VERSION'],
    )
    return llm

# Function to get completion response from language model
def get_completion(string_prompt_value):
    llm = get_llm_instance()

    try:
        response = llm.invoke(string_prompt_value)  # Use invoke instead of __call__
        return response
    except Exception as e:
        print(f"An error occurred during completion request: {e}")
        return None

# Testing the completion function
# Main function to read the CSV, embed information, and perform similarity search

    # __________________Read in CSV file and ensure match________________________________________
def read_csv_file(file_path='to_llm.csv'):
    try:
        df = pd.read_csv(file_path)
        print("CSV file read successfully!")
        return df.head(10)  # Limit to first 10 rows
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def combine_columns_to_prompt(df, activity_columns, facility_columns):
    df['combined_activities'] = df[activity_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    df['combined_facilities'] = df[facility_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    return df

def get_openai_client():
    config = get_llm_api_config_details()
    client = AzureOpenAI(
        api_key=config['AZURE_API_KEY'],
        api_version=config['AZURE_API_VERSION'],
        azure_endpoint=config['AZURE_API_BASE']
    )
    return client
# Function to embed user information
def embed_text(text, model_name='text-embedding-3-small'):
    client = get_openai_client()
    response = client.embeddings.create(input=[text], model=model_name)
    embedding = response.data[0].embedding
    return embedding


# Function to embed dataframe rows
def embed_dataframe(df, model_name='text-embedding-3-small'):
    df['combined_info'] = df.apply(lambda row: f"MDAS ID: {row['MDAS ID']}, Age: {row['Age']}, Race: {row['Race']}, Activities: {row['combined_activities']}, Facilities: {row['combined_facilities']}", axis=1)
    embeddings = df['combined_info'].apply(lambda x: embed_text(x, model_name=model_name)).tolist()
    return embeddings

# Function to perform similarity search
def perform_similarity_search(user_embedding, embedded_data, top_n=5):
    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(user_embedding, torch.tensor(embedded_data)).numpy()[0]

    # Get indices of top similar users
    top_indices = (-similarities).argsort()[:top_n]

    return top_indices, similarities[top_indices]


def parse_user_query(query):
    # Define some simple rules to extract information
    query_dict = {
        'MDAS ID': '',
        'Age': '',
        'Race': '',
        'Activity 1': '',
        'Activity 2': '',
        'Activity 3': '',
        'Facility 1': '',
        'Facility 2': '',
        'Facility 3': ''
    }
    
    # Rule-based extraction
    query_lower = query.lower()
    if 'malay' in query_lower:
        query_dict['Race'] = 'Malay'
    if 'chinese' in query_lower:
        query_dict['Race'] = 'Chinese'
    if 'indian' in query_lower:
        query_dict['Race'] = 'Indian'
    
    activities = ['badminton', 'swimming', 'running', 'yoga']
    for activity in activities:
        if activity in query_lower:
            if query_dict['Activity 1'] == '':
                query_dict['Activity 1'] = activity.capitalize()
            elif query_dict['Activity 2'] == '':
                query_dict['Activity 2'] = activity.capitalize()
            else:
                query_dict['Activity 3'] = activity.capitalize()

    # You can add more rules to parse other fields if necessary
    return query_dict

# Main function to read CSV, embed information, and perform similarity search
def find_top_matching_users(user_query, csv_file='to_llm.csv', model_name='text-embedding-3-small', top_n=5):
    try:
        # Parse user query
        user_input = parse_user_query(user_query)
        
        # Read CSV file
        df = read_csv_file(csv_file)

        # Combine columns to create prompts
        df = combine_columns_to_prompt(df, ['Activity 1', 'Activity 2', 'Activity 3'], ['Facility 1', 'Facility 2', 'Facility 3'])

        # Embed CSV data
        embedded_data = embed_dataframe(df, model_name=model_name)

        # Create user input text
        user_info = f"MDAS ID: {user_input['MDAS ID']}, Age: {user_input['Age']}, Race: {user_input['Race']}, Activities: {user_input['Activity 1']}, {user_input['Activity 2']}, {user_input['Activity 3']}, Facilities: {user_input['Facility 1']}, {user_input['Facility 2']}, {user_input['Facility 3']}"

        # Embed user input
        user_embedding = embed_text(user_info, model_name=model_name)

        # Perform similarity search
        top_indices, top_scores = perform_similarity_search(user_embedding, embedded_data, top_n=top_n)

        # Get top matching users from dataframe
        top_users = df.iloc[top_indices]

        return top_users, top_scores

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


def export_to_csv(top_users, top_scores, output_file='top_matching_users.csv'):
    try:
        top_users['Similarity Score'] = top_scores
        top_users.to_csv('output.csv', index=False)
        print(f"Top matching users and their scores have been exported to {output_file}")
    except Exception as e:
        print(f"An error occurred while exporting to CSV: {e}")

# Example usage:
if __name__ == "__main__":
    user_query = "I want a Malay who does volunteerism"

    top_users, top_scores = find_top_matching_users(user_query)

    if top_users is not None:
        print("Top Matching Users:")
        print(top_users)
        print("Corresponding Similarity Scores:")
        print(top_scores)
        export_to_csv(top_users, top_scores)