import pandas as pd
import os
from langchain_openai import AzureChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from dotenv import dotenv_values
from dotenv import load_dotenv

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import HumanMessage

#function to pass in a row that contains activity frequency and filter out all 0s
def sort_and_filter_by_frequency(df):
    # Filter out rows where the frequency is 0
    df_filtered = df[df['Frequency'] > 0]
    
    # Sort the DataFrame by the 'Frequency' column in descending order
    df_sorted = df_filtered.sort_values(by='Frequency', ascending=False)
    
    return df_sorted

# Function LLM uses to extract the corresponding frequency. 

def extract_activity_frequency(df, column, keyword):
    # Create a case-insensitive pattern with word boundaries to avoid partial matches
    short_keyword = keyword[:4].lower()
    pattern = rf'(?i)\b{short_keyword}\b.*: (\d+)'
    df['Frequency'] = df[column].str.extract(pattern, expand=False).fillna(0).astype(int)
    return df

#function that the LLM will use to pass in the dataframe, a relevant column and one or more relevant keywords to filter

def filter_by_activity(df, column, keyword):
    # Take the first four letters of the keyword
    short_keyword = keyword[:4].lower()
    return df[df[column].str.contains(short_keyword, case=False, na=False)]

# streaming logic. Inspired from https://github.com/streamlit/StreamlitLangChain/blob/main/streaming_demo.py
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token + " "
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")

# Load the data from the CSV file
def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        return df.fillna(value=0)
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

# Function to retrieve Azure API configuration from environment
def get_llm_api_config_details():
    config = dotenv_values('openai.env')
    required_keys = ['AZURE_API_TYPE', 'AZURE_API_BASE', 'AZURE_API_VERSION', 'AZURE_API_KEY']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing key: {key}")
    return config

    # secrets = {}

    # list_of_secrets = [
    #     'AZURE_API_TYPE',
    #     'AZURE_API_BASE',
    #     'AZURE_API_VERSION',
    #     'AZURE_API_KEY'
    # ]

    # local_secrets = []

    # for s in list_of_secrets:
    #     if os.getenv(s) is None:
    #         local_secrets.append(s)
    #     else:
    #         secrets[s] = os.getenv(s)

    # #running the app locally, will reference a dotenv to get secrets
    # if len(local_secrets) > 0:
    #     dotenv_path = '.env'
    #     load_dotenv(dotenv_path)

    #     for s in local_secrets:
    #         secrets[s] = os.getenv(s)

    # return secrets


# Initialize Azure OpenAI instance
def get_llm_instance(model='gpt-4', temperature=0.8, timeout=1000):
    config = get_llm_api_config_details()
    llm = AzureChatOpenAI(
        openai_api_version=config['AZURE_API_VERSION'],
        azure_deployment=model,
        azure_endpoint=config['AZURE_API_BASE'],
        openai_api_key=config['AZURE_API_KEY'],
        temperature=temperature,
        request_timeout=timeout
    )
    return llm

# Define the templates
def load_templates():
    templates = {
        "prefix": """
        Import pandas as pd. 

        Define the given functions in your context:

        # Function one:
        def sort_and_filter_by_frequency(df):
        # Filter out rows where the frequency is 0
        df_filtered = df[df['Frequency'] > 0]
        
        # Sort the DataFrame by the 'Frequency' column in descending order
        df_sorted = df_filtered.sort_values(by='Frequency', ascending=False)
        
        return df_sorted
        
        # Function two:

        def extract_activity_frequency(df, column, keyword):
            short_keyword = keyword[:4].lower()
            pattern = rf'(?i){short_keyword}.*?: (\d+)'
            df['Frequency'] = df[column].str.extract(pattern, expand=False).fillna(0).astype(int)
            return df
        
        # Function three: 

        def filter_by_activity(df, column, keyword):
            short_keyword = keyword[:4].lower()
            return df[df[column].str.contains(short_keyword, case=False, na=False)]

        I need your help in analyzing a dataset of user activities.
        """,
        "step1": """
        Discern whether in the user query, it is asking for activity or facility related questions.
        If the query asks for things like "Swimming pools", "Gyms", they are interested in facilities.
        If it asks for things like "social causes", "group dance" etc, they are interested in activities.
        
        Sometimes, the query will ask for different things, then you have to look at the relevant columns
        to identify. If I ask for someone old, try to find people's age, if I ask for
        people of high economic class, try to identify a valid column that might help.
        Use this knowledge when deciding the next step.
        """,
        "step2": """
        Filter the dataset to include relevant rows by passing in the dataframe, most relevant column,
        and keywords to the filter_by_activity(df, column, keyword) function. Pass in multiple related keywords if the query
        is not specifically mentioning one activity.
        For example, if asked for "social causes", try to pass in things like "Volunteerism"
        If asked for "History lovers", pass in things like "Museum" etc.
        You have to decide what keyword to use for generalized queries.
        If this filtered dataset is empty, redo the process from the start.

        Next, find the frequency of the relevant activities by calling the extract_activity_frequency(df, column, keyword)
        function, using the same keywords and column you used to filter the dataset.
        If this filtered dataset is empty, redo the process from the start.
        """,
        "step3": """
        use the python_repl_ast tool, execute code one command at a time.

        You need to look at the prompt and decide which two methods of the provided ones below might make the most sense. You may
        also come up with your own methods if none of the below methods work well.
           For example, 
        - If the query is about "the state of Olympic swimming in Singapore", use method 1.
        - If the query is about "barriers to entry in swimming",use method 4.
        - If the query is about "what stops people from swimming more", use method 3.
        Look through all the methods, and decide which 2 are the best to use for the query.
        Identify and list the number of MDAS IDs based on the query.
        Export the results of each method you pick ultimately to "result1.csv" and "result2.csv". 
        
        Method 1: Pick the IDs with the highest frequency. To sort, use the function sort_and_filter_by_frequency(df). If this filtered dataframe is empty, redo the process from the start.
        
        Method 2: Pick those who have that activity, but pick people of different ethnicities for diversity. Try to have at least 2 Chinese, 2 Malays, 2 Indians, and 2 Others, but in cases where not all races exist for an activity, it is ok to have more people of races that do exist.

        Method 3: Pick people who do the activity but infrequently. You are free to decide how to do this.

        Method 4: Pick minorities, such as females, minority ethnicities and so on, with a mix of those who do the activity frequently and those
        who do not. Just for context, in Singapore, 
        the smallest minority races are indians and others, followed by Malays. The Chinese are a majority.

     
        """,
        "guidelines": """
        - ALWAYS use the python_repl_ast tool to execute the python commands
        - if you try the same Action input three times and you run into errors, switch to a different action input.
        - always do one command at a time if using python
        - any variables declared need to persist, and be global throughout the entire client execution
        - Explain the process and reasoning in your final answer.
        - Use Markdown for formatting the final response.
        - Always export results to result1.csv and result2.csv
        """,
        "output_format": """
        Please provide a brief overview of the methods and inferences you used to find the MDAS IDs according to the context.
        A simple explanation of each method, for example "We first identified the most active Swimmers, here's a list of people you could speak to based on that"
        and "we next identified swimmers across different activities, here's a list of people you could speak to for diverse perspectives."
        """
    }
    return templates

# Combine templates and question into a single prompt
def build_prompt(templates, question, column_names):
    column_names_str = ", ".join(column_names)
    prompt = (
        templates["prefix"] +
        'import pandas as pd. Import re' +
        f"\nThe dataset contains the following columns: {column_names_str}\n" +
        templates["step1"] +
        templates["step2"] +
        templates["step3"] +
        templates["guidelines"] +
        templates["output_format"] +
        "\n\n" +
        question +
        'using the python_repl_ast tool only' +
        'executing python code only one command at a time' +
        'never add backticks "`" around the action input' +
        'never use Re'
    )
    return prompt

# Function to invoke LLM with prompt
def invoke_llm(llm, prompt, stream_handler):
    human_message = HumanMessage(content=prompt)
    return llm([human_message], callbacks=[stream_handler])

# Function to create a Pandas DataFrame agent
def create_agent(llm, df):
    return create_pandas_dataframe_agent(llm=llm, df=df, verbose=True, allow_dangerous_code=True, handle_parsing_errors=True)
