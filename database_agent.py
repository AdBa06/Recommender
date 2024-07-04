
import os 
import pandas as pd
from IPython.display import Markdown, display
from langchain_openai import AzureChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from dotenv import load_dotenv, dotenv_values

# Load the data from the CSV file
def read_csv_file(file_path='to_llm.csv'):
    try:
        df = pd.read_csv(file_path)
        print("CSV file read successfully!")
        return df.fillna(value=0)
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
def get_llm_instance(model='gpt-4'):
    config = get_llm_api_config_details()
    llm = AzureChatOpenAI(
        openai_api_version=config['AZURE_API_VERSION'],
        azure_deployment=model,
        azure_endpoint=config['AZURE_API_BASE'],
        openai_api_key=config['AZURE_API_KEY']
    )
    return llm
CSV_PROMPT_PREFIX = """
You are an AI assistant. I need your help in analyzing a dataset of user activities.

Task:
First, import pandas as pd. Then, do the tasks below:

1. Set pandas display options to show all columns.
2. Retrieve column names from the dataset.
3. Identify and list 10 MDAS IDs based on the given criteria.

Example:
If the prompt is "I want to have a discussion with people who love swimming," follow these steps:
- Select the most active swimmers.
- Ensure diversity by including swimmers from different ethnicities.
- You could try to look for "swimming" or "swim" or anything related in the activity columns.

Explanation:
Each selected ID should be chosen based on activity level and diversity. Mention why each ID was chosen.

Example Prompt: I want to have a discussion with people who love swimming.
Example Response:
Here is a list of people you might want to speak to as the Swim frequently and swimming is their activity 1.
MDAS ID: 123, 456, 789, ...

Alternatively, here is another list of people you could speak to as you could talk to minority ethnicities involved
in swimming:
MDAS ID: 222 (Reason: Indian), 333 (Reason: Malay), 444 (Reason: Other etc)

Now, based on the prompt provided, find the 10 most relevant MDAS IDs for two different cases of looking at it like in
the previous example.

Export the 10 rows corresponding to these MDAS IDs to a csv sheet called 'result1.csv' and 'result2.csv' for the two different cases.
do this by creating a dataframe with the relevant rows of IDs, 
then use the pandas to_csv function.
once done, add that you have exported the results in your response
"""

CSV_PROMPT_SUFFIX = """
- Always try another method before giving the final answer.
- Reflect on the answers from both methods and ensure they are consistent.
- Explain the process and reasoning in your final answer.
- Use Markdown for formatting the final response.
- Only use results from your calculations, no prior knowledge.
- Explain how you arrived at the final answer, mentioning relevant column names.
"""

CSV_PROMPT_OUT = """
Please provide the final 10 IDs and their details as follows:

MDAS ID, Reason, Activity 1, Ethnicity
"""


CSV_PROMPT_ADD = """
Export the 10 rows corresponding to these MDAS IDs to a csv sheet called 'result.csv', do this by creating a dataframe with the relevant rows of IDs, 
then use the pandas to_csv function.
once done, add that you have exported the results in your response"""

# Main function to read CSV, invoke LLM, and display response
def main():
    # Load the CSV data
    df = read_csv_file("to_llm.csv")
    if df is None:
        print("Failed to read CSV file.")
        return
    
    # Initialize the Azure ChatOpenAI model
    model = get_llm_instance()

    # Create a Pandas DataFrame agent
    agent = create_pandas_dataframe_agent(llm=model, df=df, verbose=True, allow_dangerous_code=True, handle_parsing_errors=True)

    # Define the user query
    QUESTION = """
    I want to have a discussion with people who like chit chat.
    """

    # Invoke the agent to get the response
    try:
        response = agent.invoke(CSV_PROMPT_PREFIX  + QUESTION + CSV_PROMPT_SUFFIX + CSV_PROMPT_OUT)
        # display(Markdown(response))
        print(response)
    except Exception as e:
        print(f"An error occurred during the agent invocation: {e}")

if __name__ == "__main__":
    main()