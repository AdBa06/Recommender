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

#__________________________________________Access OpenAI API________________________________________________
# Function to retrieve Azure API configuration from environment
def get_llm_api_config_details():
    # Access environment variables from openai.env
    config = dotenv_values('openai.env')

    # Check if config has all required keys
    required_keys = ['AZURE_API_TYPE', 'AZURE_API_BASE', 'AZURE_API_VERSION', 'AZURE_API_KEY']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing  key: {key}")

    # Print loaded configuration (optional)
    print("Configuration loaded:")
    return config

# initialize Azure ChatOpenAI instance
def get_llm_instance(model='gpt-35-turbo'):   
    config = get_llm_api_config_details()
  
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
        response = llm.invoke(string_prompt_value)  # had to switch to using invoke
        return response
    except Exception as e:
        print(f"An error occurred during completion request: {e}")
        return None


    # __________________Read in CSV file and ensure match________________________________________
def read_csv_file(file_path='to_llm.csv'):
    try:
        df = pd.read_csv(file_path)
        print("CSV file read successfully!")
        # return df
        return df.head(30)  # Limit to first 30 rows for now
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
# ______________________________ Combine activities and facilities into one category _______________________________
    
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

#---------------------------------------------- EMBEDDING DATAFRAME ------------------------------------------------
# Function to embed user information
def embed_text(text, model_name='text-embedding-3-small'):
    client = get_openai_client()
    response = client.embeddings.create(input=[text], model=model_name)
    embedding = response.data[0].embedding
    return embedding

# Function to embed dataframe rows
# Function to embed dataframe rows
def embed_dataframe(df, model_name='text-embedding-3-small'):
    df['combined_info'] = df.apply(lambda row: ' '.join([
        f"MDAS ID: {row['MDAS ID']}",
        f"Age: {row['Age']}" if row['Age'] != 'Unknown' else '',
        f"Race: {row['Race']}" if row['Race'] != 'Unknown' else '',
        f"Activities: {row['combined_activities']}" if row['combined_activities'] != 'Unknown' else '',
        f"Facilities: {row['combined_facilities']}" if row['combined_facilities'] != 'Unknown' else ''
    ]), axis=1)
    embeddings = df['combined_info'].apply(lambda x: embed_text(x, model_name=model_name)).tolist()
    
    return embeddings

# Function to perform similarity search
def perform_similarity_search(user_embedding, embedded_data, top_n=5):
    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(user_embedding, torch.tensor(embedded_data)).numpy()[0]

    # Get indices of top similar users
    top_indices = (-similarities).argsort()[:top_n]

    return top_indices, similarities[top_indices]

# Function to calculate keyword matching score
def calculate_keyword_score(query, text, keywords):
    filtered_keywords = {keyword for keyword in keywords if keyword != 'Unknown'}
    
    query_keywords = set(query.split())
    text_keywords = set(text.split())
    
    # Calculate the intersection count with filtered keywords
    match_count = len(query_keywords.intersection(filtered_keywords))
    
    return match_count

# Function to combine keyword and cosine similarity scores
def combine_scores(keyword_score, cosine_similarity, keyword_weight=0, cosine_weight=1):
    combined_score = (keyword_weight * keyword_score) + (cosine_weight * cosine_similarity)
    return combined_score
# --------------------------------------------------- PARSING USER QUERY USING PYDANTIC ---------------------------------
# Define Pydantic Model for extracting full query information
class QueryExtraction(BaseModel):
    Race: str = Field(description='The race mentioned in the user query.')
    Activities: list[str] = Field(description='List of activities mentioned in the user query.')
    Age: str = Field(description='Age if mentioned in the user query.', default='')

# Function to extract information from user query using LLM
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

# Function to parse user query using the LLM
def parse_user_query_with_llm(query, model_name='gpt-35-turbo'):
    extracted_info = extract_info_from_query(query, model_name)
    activities = extracted_info.Activities if extracted_info else []
    race = extracted_info.Race if extracted_info else ''
    age = extracted_info.Age if extracted_info else ''

    # Initialize the query_dict with default values
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

    # Assign extracted activities to the query_dict
    for i, activity in enumerate(activities):
        if i == 0:
            query_dict['Activity 1'] = activity.capitalize()
        elif i == 1:
            query_dict['Activity 2'] = activity.capitalize()
        elif i == 2:
            query_dict['Activity 3'] = activity.capitalize()

    return query_dict

# Main function to read CSV, embed information, and perform similarity search
def find_top_matching_users(user_query, csv_file='to_llm.csv', model_name='text-embedding-ada-002', top_n=5, keyword_weight=0.7, cosine_weight=0.3):
    try:
        # Parse user query
        user_input = parse_user_query_with_llm(user_query)
        print(user_input)
        
        # Read CSV file
        df = read_csv_file(csv_file)

        # Combine columns to create prompts
        df = combine_columns_to_prompt(df, ['Activity 1', 'Activity 2', 'Activity 3'], ['Facility 1', 'Facility 2', 'Facility 3'])

        # Embed CSV data
        embedded_data = embed_dataframe(df, model_name=model_name)

        # Create user input text
        user_info = ' '.join([
            f"Age: {user_input['Age']}" if user_input['Age'] != 'Unknown' else '',
            f"Race: {user_input['Race']}" if user_input['Race'] != 'Unknown' else '',
            f"Activities: {user_input['Activity 1']}, {user_input['Activity 2']}, {user_input['Activity 3']}" if any([user_input['Activity 1'], user_input['Activity 2'], user_input['Activity 3']]) else '',
            f"Facilities: {user_input['Facility 1']}, {user_input['Facility 2']}, {user_input['Facility 3']}" if any([user_input['Facility 1'], user_input['Facility 2'], user_input['Facility 3']]) else ''
        ])
        print("User info to be embedded:", user_info)
        # Embed user input
        user_embedding = embed_text(user_info, model_name=model_name)

        # Perform similarity search
        top_indices, cosine_similarities = perform_similarity_search(user_embedding, embedded_data, top_n=top_n)

        # Calculate keyword scores
        keywords = set(user_info.split())
        keyword_scores = [calculate_keyword_score(user_info, df.iloc[i]['combined_info'], keywords) for i in top_indices]

        # Combine scores
        combined_scores = [combine_scores(kw_score, cos_sim, keyword_weight, cosine_weight) for kw_score, cos_sim in zip(keyword_scores, cosine_similarities)]

        # Sort by combined scores
        sorted_indices = [x for _, x in sorted(zip(combined_scores, top_indices), reverse=True)]
        top_scores = sorted(combined_scores, reverse=True)

        # Get top matching users from dataframe
        top_users = df.iloc[sorted_indices]

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
    user_query = "I want to have a discussion group in volunteerism. Include no Malays"
    top_users, top_scores = find_top_matching_users(user_query)
    if top_users is not None and top_scores is not None:
        export_to_csv(top_users, top_scores)
    else:
        print("No matching users found.")
