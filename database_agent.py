import os
import pandas as pd
from IPython.display import Markdown, display
from langchain_openai import AzureChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from dotenv import load_dotenv, dotenv_values

from thought_process import thought_process

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

# Perform initial steps manually
def initial_setup(df):
    pd.set_option('display.max_columns', None)
    column_names = df.columns.tolist()
    print(f"Column names: {column_names}")
    return column_names

# Define the templates
def load_templates():
    templates = {
        "prefix": """
        You are an AI assistant. I need your help in analyzing a dataset of user activities.
        Before we do anything, import pandas as pd.
        """,

          "step1": """
       Discern whether in the user query, it is asking for activity or facility related questions.
       If the query asks for things like "Swimming pools", "Gyms", they are interested in facilities
       If it asks for things like "social causes", "group dance" etc, they are interested in activities.
       Use this knowledge when deciding the next step.
        """,
        
          "step2": """
       Recognize that the Activities column in the df is formatted as "Activity1: Frequency, Activity2: Frequency..." and so on,
       so you might have to ensure you are able to read within the entire string when looking for activities. 
       
       Do not split the  activities column, just read from within the larger string to see if anything relevant exists anywhere within,
       and then identify the number right after the relevant activity.

       You need to come up with a list of activities from all the activites in the dataframe that might be related to the prompt. 
       For example here is a good method that works. 
       To identify activities related to water, look for keywords such as "swim", "water", "pool", "dive", "aquatic", etc. in the activities.
       
        Now, filter the dataset to only keep the rows that have relavant activities. 

        """,

        "step3": """
        Step 3: Identify and list 10 MDAS IDs based on the given criteria through multiple methods.

       
        Method 1: Look through the entire activity column, 
        The strings in the activity column are formatted as "Activity 1: Frequency, Activity 2: Frequency etc".
        So you just need to look at the frequency of the activity based on your filtered list to decide who is the most active for a given activity.
        Do not filter by anything else other than the number right after the activity within the string.
        Pick those with the largest numbers once you found the people who do the activity. Pick 10 MDAS IDs.
        
        If I see two options, "Karate: 95, Eating: 12, Badminton: 2" and "Judo: 107, Chit chat: 10", I will pick the second, since the frequency is bigger.
        Follow that thought process.

        Method 2: Pick those who have that activity, but pick people of different ethnicities for diversity. Try to have at least 2 Chinese, 2 Malays, 2 Indians, and 2 Others, but in cases where not all races exist for an activity, it is ok to have more people of races that do exist.

        Example Prompt: I want to have a discussion with people who love martial arts.

        Example Response:
        Here is a list of people you might want to speak to as they frequently engage in martial arts and have a high stake in it:
        MDAS ID: 123, 456, 789, ...

        Alternatively, here is another list of people you could speak to in order to have diversity and differing perspectives:
        MDAS ID: 222, 333, 444 ...
        """,

        "step4": """
        Step 4: Export the 10 rows corresponding to these MDAS IDs to a csv sheet called 'result1.csv' and 'result2.csv' for the two different cases.
        Create a dataframe with the relevant rows of IDs, then use the pandas to_csv function.
        Once done, add that you have exported the results in your response.
        """,

        "guidelines": """
        - always use [python_repl_ast]
        - always do one command at a time if using python, you cannot handle multiple commands at the
        same time
        - you do not need to split the values in the activities column
        - any variables declared need to persist, and be global throughout the entire client execution
        - If the query is about facilities, do the same thing but with the facilities column instead 
          of activities column for analysis. 
        - Always try another method before giving the final answer.
        - Reflect on the answers from both methods and ensure they are consistent.
        - Explain the process and reasoning in your final answer.
        - Use Markdown for formatting the final response.
        - Only use results from your calculations, no prior knowledge.
        - Explain how you arrived at the final answer, mentioning relevant column names.
        """,

        "output_format": """
        Please provide a brief overview of the method you used to find 10 MDAS IDs according to the context,
        and that they can be seen in the output csv file, specify the reason and filename 
        for each method that you used.
        """
    }
    return templates

# Combine templates and question into a single prompt
# Combine templates and question into a single prompt
def build_prompt(templates, question, column_names):
    column_names_str = ", ".join(column_names)
    prompt = (
        templates["prefix"] +
        f"\nThe dataset contains the following columns: {column_names_str}\n" +
        templates["step2"] +
        templates["step3"] +
        templates["step4"] +
        templates["guidelines"] +
        templates["output_format"] +
        "\n\n" +
        # thought_process  # Add the thought process here
        question 
    )
    return prompt

# Main function to read CSV, invoke LLM, and display response
def main():
    # Load the CSV data
    df = read_csv_file("to_llm.csv")
    if df is None:
        print("Failed to read CSV file.")
        return
    
    # Perform initial setup
    column_names = initial_setup(df)
    
    # Initialize the Azure ChatOpenAI model
    model = get_llm_instance()

    # Create a Pandas DataFrame agent
    agent = create_pandas_dataframe_agent(llm=model, df=df, verbose=True, allow_dangerous_code=True, handle_parsing_errors=True)

    # Load templates
    templates = load_templates()

    # Define the user query
    question = """
    I want to have a focus group discussion with people who might be very socially active, who can I speak to?.
    """

    # Build the final prompt
    prompt = build_prompt(templates, question, column_names)

    # Invoke the agent to get the response
    try:
        response = agent.invoke(prompt)
        display(Markdown(response))
        print(response)
    except Exception as e:
        print(f"An error occurred during the agent invocation: {e}")

if __name__ == "__main__":
    main()
