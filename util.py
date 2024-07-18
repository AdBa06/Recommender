import pandas as pd
from langchain_openai import AzureChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from dotenv import dotenv_values

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

# Initialize Azure OpenAI instance
def get_llm_instance(model='gpt-4', temperature=1, timeout=600):
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
        Import pandas as pd. Import re. I need your help in analyzing a dataset of user activities.
        """,
        "step1": """
        Discern whether in the user query, it is asking for activity or facility related questions.
        If the query asks for things like "Swimming pools", "Gyms", they are interested in facilities.
        If it asks for things like "social causes", "group dance" etc, they are interested in activities.
        Sometimes, the query will ask for different things, then you have to look at the relevant columns
        to identify things. Like if I ask for someone old, try to find people's age, if I ask for
        people of high economic class, try to identify a valid column that might help.
        Use this knowledge when deciding the next step.
        """,
        "step2": """
        use the python_repl_ast tool, execute code line by line.
        Recognize that the Activities column is formatted as "Activity1: Frequency, Activity2: Frequency..." and so on,
        so you might have to ensure you are able to read within the entire string when looking for activities. 

        You need to come up with a list of activities from all the activities in the dataframe that might be related to the prompt. 
        Do not use direct keyword extraction, you need to keep all relevant rows, even those that
        do not explicitly mention swimming. You might need to look for instances of words like swim, aquatic, pool, dive etc

        Now, filter the dataset to only keep the rows that have relevant activities.
        Do not split the columns up, just keep the relevant rows.
        if your filtered dataset is empty, clearly you did something wrong and you have to reattempt.
        """,
        "step3": """
        use the python_repl_ast tool, execute code one command at a time.

        You need to look at the prompt and decide which two methods of the provided ones below might make the most sense. You may
        also come up with your own methods if none of the below methods work well.
        Identify and list 10 MDAS IDs.
        
        Method 1: Look through the entire relevant column

        The strings in the activity column are formatted as "Volunteering: 20, swimming class: 15 etc".
        You need to figure out the frequency corresponding to swimming, which in the example given is 15.

        If the string is "CNY Getai @ 883: 1, swimming: 1",
        the frequency is not 883, because the frequency corresponding to swimming is 1, after the colon

        If the string is "Swimming challenge 2020: 2", the frequency is not 2020, it is 2. Remember 
        to get the frequency right.

        Use .str.extract('keyword.*: (\d+)', expand=False) to make sure you are capturing all activities named
        differently and not just a fixed keyword. 
         
        The number always comes after the colon, not before.

        Pick those with the largest numbers once you found the people who do the activity. Pick 10 MDAS IDs.
       
        if the frequency column you have created consists of purely 0, there is something wrong with your approach and you need to try again with a different extraction method.
        If the list is empty, something is wrong.
        
        Method 2: Pick those who have that activity, but pick people of different ethnicities for diversity. Try to have at least 2 Chinese, 2 Malays, 2 Indians, and 2 Others, but in cases where not all races exist for an activity, it is ok to have more people of races that do exist.

        Method 3: Pick people who do the activity but infrequently. You are free to decide how to do this.

        Method 4: Pick minorities, such as females, minority ethnicities and so on. Just for context, in Singapore, 
        the smallest minority races are indians and others, followed by Malays. The Chinese are a majority.

        For example, 
        - If the query is about "the state of Olympic swimming in Singapore", look for the most active swimmers, since it makes
        sense to find the most actively involved swimmers for things like competitive swimming and talking about it.
        - If the query is about "barriers to entry in swimming", look for people such as females and minority ethnicities to get their perspectives.
        - If the query is about "what stops people from swimming more", look for people who swim but not too often.
        """,
        "step4": """
        use the python_repl_ast tool, execute code line by line.
        Always Export the 10 rows corresponding to these MDAS IDs using each method you used
        to result1.csv and result2.csv
        Include the entire row from the original csv file, and add in the activity frequency as a new column
        """,
        "guidelines": """
        - ALWAYS use the python_repl_ast tool to execute the python commands
        - Never use re in your searches, use str.extract
        - if you try the same Action input three times and you run into errors, switch to a different action input.
        - always do one command at a time if using python, you cannot handle multiple commands at the same time
        - any variables declared need to persist, and be global throughout the entire client execution
        - If the query is about facilities, do the same thing but with the facilities column instead of activities column for analysis. 
        - Always try another method before giving the final answer.
        - Reflect on the answers from both methods and ensure they are consistent.
        - Explain the process and reasoning in your final answer.
        - Use Markdown for formatting the final response.
        """,
        "output_format": """
        Please provide a brief overview of the methods and inferences you used to find 10 MDAS IDs according to the context.
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
        f"\nThe dataset contains the following columns: {column_names_str}\n" +
        templates["step1"] +
        templates["step2"] +
        templates["step3"] +
        templates["step4"] +
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
def invoke_llm(llm, prompt):
    return llm(prompt)

# Function to create a Pandas DataFrame agent
def create_agent(llm, df):
    return create_pandas_dataframe_agent(llm=llm, df=df, verbose=True, allow_dangerous_code=True, handle_parsing_errors=True)
