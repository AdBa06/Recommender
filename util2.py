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
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json

# Load the data from the CSV file
def read_csv_file(file_path='to_llm.csv'):
    try:
        df = pd.read_csv(file_path)
        print("CSV file read successfully!")
        return df.head(50)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Function to retrieve Azure API configuration from environment
def get_llm_api_config_details():
    config = dotenv_values('openai.env')
    required_keys = ['AZURE_API_TYPE', 'AZURE_API_BASE', 'AZURE_API_VERSION', 'AZURE_API_KEY']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing key: {key}")
    print("Configuration loaded:")
    return config

# Initialize Azure ChatOpenAI instance
def get_llm_instance(model='gpt-35-turbo'):
    config = get_llm_api_config_details()
    llm = AzureChatOpenAI(
        deployment_name=model, 
        model_name=model,
        openai_api_key=config['AZURE_API_KEY'],
        temperature=0,
        azure_endpoint=config['AZURE_API_BASE'],
        openai_api_version=config['AZURE_API_VERSION'],
    )
    return llm

# Function to get completion response from language model
def get_completion(string_prompt_value):
    llm = get_llm_instance()
    try:
        response = llm.invoke(string_prompt_value)
        return response
    except Exception as e:
        print(f"An error occurred during completion request: {e}")
        return None

def get_openai_client():
    config = get_llm_api_config_details()
    client = AzureOpenAI(
        api_key=config['AZURE_API_KEY'],
        api_version=config['AZURE_API_VERSION'],
        azure_endpoint=config['AZURE_API_BASE']
    )
    return client

# Embed user information
def embed_text(text, model_name='text-embedding-3-small'):
    client = get_openai_client()
    response = client.embeddings.create(input=[text], model=model_name)
    embedding = response.data[0].embedding
    return embedding

# Embed dataframe rows
def embed_dataframe(df, model_name='text-embedding-3-small'):
    df['combined_info'] = df.apply(lambda row: ' '.join([
        f"MDAS ID: {row['MDAS ID']}",
        f"Age: {row['Age']}" if row['Age'] != 'Unknown' else '',
        f"Race: {row['Race']}" if row['Race'] != 'Unknown' else '',
        f"Activity: {row['Activity 1']}" if row['Activity 1'] != 'Unknown' else '',
    ]), axis=1)
    embeddings = df['combined_info'].apply(lambda x: embed_text(x, model_name=model_name)).tolist()
    return embeddings

# Perform similarity search
def perform_similarity_search(user_embedding, embedded_data, top_n=30):
    similarities = util.pytorch_cos_sim(user_embedding, torch.tensor(embedded_data)).numpy()[0]
    top_indices = (-similarities).argsort()[:top_n]
    return top_indices, similarities[top_indices]

# Calculate keyword matching score
def calculate_keyword_score(query, text, keywords):
    filtered_keywords = {keyword for keyword in keywords if keyword != 'Unknown'}
    query_keywords = set(query.split())
    text_keywords = set(text.split())
    match_count = len(query_keywords.intersection(filtered_keywords))
    return match_count

# Combine keyword and cosine similarity scores
def combine_scores(keyword_score, cosine_similarity, keyword_weight=0.7, cosine_weight=0.3):
    combined_score = (keyword_weight * keyword_score) + (cosine_weight * cosine_similarity)
    return combined_score

# Define Pydantic Model for extracting full query information
class QueryExtraction(BaseModel):
    Race: str = Field(description='The race mentioned in the user query.')
    Activities: list[str] = Field(description='List of activities mentioned in the user query.')
    Age: str = Field(description='Age if mentioned in the user query.', default='')

# Extract information from user query using LLM
def extract_info_from_query(query, model_name='gpt-35-turbo'):
    parser = PydanticOutputParser(pydantic_object=QueryExtraction)
    system_prompt = "Your answer should be in JSON format with no additional words. You are a bot that will assist in extraction of relevant information from user queries."
    user_prompt = f"""Given the following user query:
    {query}
    Extract the race, age (if mentioned), and activities in the following format:
    {parser.get_format_instructions()}
    
    Examples of activities include but are not limited to: badminton, swimming, running, yoga, hiking, cycling, basketball, tennis, etc.
    Examples of races include: Malay, Chinese, Indian, European, Eurasian"""
    llm = get_llm_instance(model=model_name)
    try:
        extraction = llm.invoke(system_prompt + "\n" + user_prompt)
        return parser.parse(extraction.content)
    except Exception as e:
        print(f"An error occurred during information extraction: {e}")
        return None

def get_related_keywords(activity, model_name='gpt-35-turbo'):
    prompt = f"Generate a list of related activities for the given activity: {activity}"
    response = get_completion(prompt)
    related_keywords = response.strip().split(',')
    return [keyword.strip() for keyword in related_keywords]

# Parse user query using the LLM
def parse_user_query_with_llm(query, model_name='gpt-35-turbo'):
    extracted_info = extract_info_from_query(query, model_name)
    activities = extracted_info.Activities if extracted_info else []
    race = extracted_info.Race if extracted_info else ''
    age = extracted_info.Age if extracted_info else ''

    query_dict = {
        'Age': age,
        'Race': race,
        'Activity 1': '',
        'Activity 2': '',
        'Activity 3': '',
        'Facility 1': '',
        'Facility 2': '',
        'Facility 3': ''
    }

    for i, activity in enumerate(activities):
        if i == 0:
            query_dict['Activity 1'] = activity.capitalize()
        elif i == 1:
            query_dict['Activity 2'] = activity.capitalize()
        elif i == 2:
            query_dict['Activity 3'] = activity.capitalize()

    return query_dict

# Filter dataframe based on dictionaries and parsed query
def filter_dataframe(df, query_dict, race_dict, activity_dict):
    race = query_dict['Race']
    activities = [query_dict['Activity 1']]

    if race not in race_dict:
        print(f"No MDAS IDs found for race: {race}")
        return pd.DataFrame()

    race_ids = set(race_dict[race])
    activity_ids = set()

    for activity in activities:
        if activity in activity_dict:
            activity_ids.update(activity_dict[activity])

    common_ids = race_ids.intersection(activity_ids)
    print(f"Number of common MDAS IDs: {len(common_ids)}")

    if not common_ids:
        print("No common MDAS IDs found for the given race and activities.")
        return pd.DataFrame()

    filtered_df = df[df['MDAS ID'].isin(common_ids)]
    return filtered_df

# Main function to read CSV, embed information, and perform similarity search
def find_top_matching_users(user_query, csv_file='to_llm.csv', model_name='text-embedding-ada-002', top_n=30):
    try:
        user_input = parse_user_query_with_llm(user_query)
        print(user_input)
        
        df = read_csv_file(csv_file)

        # Load the dictionaries
        with open('race_dict.json', 'r') as f:
            race_dict = json.load(f)
        with open('activity_dict.json', 'r') as f:
            activity_dict = json.load(f)

        filtered_df = filter_dataframe(df, user_input, race_dict, activity_dict)
        if filtered_df.empty:
            print("No matching users found after filtering.")
            return None, None

        # Embed filtered CSV data
        embedded_data = embed_dataframe(filtered_df, model_name=model_name)

        user_info = ' '.join([
            f"Age: {user_input['Age']}" if user_input['Age'] != 'Unknown' else '',
            f"Race: {user_input['Race']}" if user_input['Race'] != 'Unknown' else '',
            f"Activities: {user_input['Activity 1']}, {user_input['Activity 2']}, {user_input['Activity 3']}" if any([user_input['Activity 1'], user_input['Activity 2'], user_input['Activity 3']]) else '',
            f"Facilities: {user_input['Facility 1']}, {user_input['Facility 2']}, {user_input['Facility 3']}" if any([user_input['Facility 1'], user_input['Facility 2'], user_input['Facility 3']]) else ''
        ])
        print("User info to be embedded:", user_info)

        user_embedding = embed_text(user_info, model_name=model_name)

        top_indices, cosine_similarities = perform_similarity_search(user_embedding, embedded_data, top_n=top_n)

        top_users = filtered_df.iloc[top_indices]
        top_users['Similarity Score'] = cosine_similarities

        return top_users, cosine_similarities

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

def export_to_csv(top_users, top_scores, output_file='top_matching_users.csv'):
    try:
        top_users['Similarity Score'] = top_scores
        top_users.to_csv(output_file, index=False)
        print(f"Top matching users and their scores have been exported to {output_file}")
    except Exception as e:
        print(f"An error occurred while exporting to CSV: {e}")

# Example usage:
if __name__ == "__main__":
    user_query = "I want to have a discussion with Malays about Volunteerism"
    top_users, top_scores = find_top_matching_users(user_query)
    if top_users is not None and top_scores is not None:
        export_to_csv(top_users, top_scores)
    else:
        print("No matching users found.")