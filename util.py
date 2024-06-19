import os
import pandas as pd
from dotenv import load_dotenv
from dotenv import dotenv_values
from langchain_openai import AzureChatOpenAI
from sentence_transformers import SentenceTransformer, util
import torch

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
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def combine_columns_to_prompt(df, activity_columns, facility_columns):
    df['combined_activities'] = df[activity_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    df['combined_facilities'] = df[facility_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    return df

# New function to embed user information
def embed_user_info(df, model):
    user_info_list = []
    for index, row in df.iterrows():
        user_info = f"MDAS ID: {row['MDAS ID']}, Age: {row['Age']}, Race: {row['Race']}, Activities: {row['combined_activities']}, Facilities: {row['combined_facilities']}"
        user_info_list.append(user_info)
    
    embeddings = model.encode(user_info_list, convert_to_tensor=True)
    return user_info_list, embeddings



# Function to embed user query and perform similarity search
def find_similar_users(user_query, user_info_list, user_embeddings, model):
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(query_embedding, user_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=5)
    
    similar_users = []
    for score, idx in zip(top_results[0], top_results[1]):
        similar_users.append((user_info_list[idx], score.item()))
    return similar_users



if __name__ == "__main__":
    df = read_csv_file('to_llm.csv')
    if df is not None:
        # Combine activities and facilities into separate prompts
        df = combine_columns_to_prompt(df, ['Activity 1', 'Activity 2', 'Activity 3'], ['Facility 1', 'Facility 2', 'Facility 3'])
        
        model = SentenceTransformer('all-MiniLM-L6-v2')  # Choose a suitable embedding model
        user_info_list, user_embeddings = embed_user_info(df, model)
        
        # Example user query
        user_query = "Looking for a person who likes badminton and is in their 30s."
        
        similar_users = find_similar_users(user_query, user_info_list, user_embeddings, model)
        print("Top 5 similar users based on the query:")
        for user_info, score in similar_users:
            print(f"User Info: {user_info}, Similarity Score: {score}")

        # Example completion request
        completion = get_completion(user_query)
        if completion:
            print(completion)
        else:
            print("Failed to get completion response.")
