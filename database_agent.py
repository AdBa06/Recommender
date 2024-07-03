
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
def get_llm_instance(model='gpt-35-turbo'):
    config = get_llm_api_config_details()
    llm = AzureChatOpenAI(
        openai_api_version=config['AZURE_API_VERSION'],
        azure_deployment=model,
        azure_endpoint=config['AZURE_API_BASE'],
        openai_api_key=config['AZURE_API_KEY']
    )
    return llm

# Define the prompt components
CSV_PROMPT_PREFIX = """
First set the pandas display options to show all the columns,
get the column names, then answer the question.
"""

CSV_PROMPT_ADDITIONAL = """
I need a list of 10 IDs based on the prompt. For this, I want you to decide who might be the most relevant people to talk to based on my prompt.

For example, if I ask to talk to people interested in swimming, you might choose to give me the most active swimmers, such as those who have it as acitivity 1. 
And then, you can also give me IDs of swimmers across different ethnicities for diversity perspective. Present me 2-3 of such cases with the corresponding 10 MDAS IDs for every prompt.



You can come up with any arbitrary reasoning to find these 10IDs, they do not have to be based on my examples above.
"""

CSV_PROMPT_SUFFIX = """
- **ALWAYS** before giving the Final Answer, try another method.
Then reflect on the answers of the two methods you did and ask yourself
if it answers correctly the original question.
If you are not sure, try another method.
- If the methods tried do not give the same result,reflect and
try again until you have two methods that have the same result.
- If you still cannot arrive to a consistent result, say that
you are not sure of the answer.
- If you are sure of the correct answer, create a beautiful
and thorough response using Markdown.
- **DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE,
ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE**.
- **ALWAYS**, as part of your "Final Answer", explain how you got
to the answer on a section that starts with: "\n\nExplanation:\n".
In the explanation, mention the column names that you used to get
to the final answer.
Also, why did you specifically choose those 10 IDs, despite many columns having 'Swim' in them?
"""

CSV_PROMPT_OUT = """
Export the 10 rows corresponding to these MDAS IDs to a csv sheet called 'result.csv'"""

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
    I want to have a discussion with people who love volunteering.
    """

    # Invoke the agent to get the response
    try:
        response = agent.invoke(CSV_PROMPT_PREFIX  + QUESTION + CSV_PROMPT_ADDITIONAL + CSV_PROMPT_SUFFIX + CSV_PROMPT_OUT)
        # display(Markdown(response))
        print(response)
    except Exception as e:
        print(f"An error occurred during the agent invocation: {e}")

if __name__ == "__main__":
    main()