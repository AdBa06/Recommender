import os
import pandas as pd
import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from dotenv import dotenv_values

# Load the data from the CSV file
def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        st.success("CSV file read successfully!")
        return df.fillna(value=0)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Function to retrieve Azure API configuration from environment
def get_llm_api_config_details():
    config = dotenv_values('openai.env')
    required_keys = ['AZURE_API_TYPE', 'AZURE_API_BASE', 'AZURE_API_VERSION', 'AZURE_API_KEY']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing key: {key}")
    st.write("Configuration loaded:")
    return config

# Initialize Azure OpenAI instance
def get_llm_instance(model='gpt-4'):
    config = get_llm_api_config_details()
    llm = AzureChatOpenAI(
        openai_api_version=config['AZURE_API_VERSION'],
        azure_deployment=model,
        azure_endpoint=config['AZURE_API_BASE'],
        openai_api_key=config['AZURE_API_KEY']
    )
    return llm

# Perform initial setup
def initial_setup(df):
    pd.set_option('display.max_columns', None)
    column_names = df.columns.tolist()
    st.write(f"Column names: {column_names}")
    return column_names

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
        Identify and list 10 MDAS IDs based on the given criteria through multiple methods.
        
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

        """,
        "step4": """
        use the python_repl_ast tool, execute code line by line.
        Always Export the 10 rows corresponding to these MDAS IDs for the two different cases in 2 different csv files, 
        which are result1.csv and result2.csv. Do not create any other csv file names.
        Include the entire row from the original csv file, and add in the activity frequency as a new column
        """,
        "guidelines": """
        - ALWAYS use the python_repl_ast tool to execute the python commands
        - Never use re in your searches, use str.extract
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

# Main function to read CSV, invoke LLM, and display response
def main():
    st.title("MCCY Citizen Recommender")
    st.subheader("Finding the best citizens to reach out to for Focus group discussions")
    st.markdown("**Disclaimer:** The recommendations are to be evaluated with manual checking, and not to be taken as 100% true")

    # Upload CSV
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = read_csv_file(uploaded_file)
        if df is not None:
            st.write("Data Preview:")
            st.dataframe(df)  # Display the entire dataframe
            
            # Perform initial setup
            column_names = df.columns.tolist()
            
            # Initialize the Azure ChatOpenAI model
            model = get_llm_instance()

            # Create a Pandas DataFrame agent
            agent = create_pandas_dataframe_agent(llm=model, df=df, verbose=True, allow_dangerous_code=True, handle_parsing_errors=True)

            # Initialize chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []

            # Display chat history
            for chat in st.session_state.chat_history:
                st.markdown(chat)

            # Define the user query
            query = st.text_input("Enter your query:")

            if query:
                try:
                    # Display loading indicator
                    with st.spinner("Processing..."):
                        # Load templates
                        templates = load_templates()

                        # Build the final prompt
                        prompt = build_prompt(templates, query, column_names)

                        # Invoke the agent and get response
                        response = agent.invoke(prompt)

                        # Real-time processing and thought display
                        st.subheader("AI Thought Process")
                        thought_process = response.split('\n')
                        for thought in thought_process:
                            st.markdown(thought)

                        # Access the elements directly from the response dictionary
                        user_query = response.get('input', 'User query not found')
                        output = response.get('output', 'Output not found')

                        st.subheader("AI Response:")
                        st.markdown(output)

                        # Read the results from the CSV files
                        result1 = pd.read_csv('result1.csv')
                        result2 = pd.read_csv('result2.csv')

                        # Store results in session state to avoid re-reading the files unnecessarily
                        st.session_state.result1 = result1
                        st.session_state.result2 = result2

                    # Display the results as tables
                    st.subheader("Result 1: Top 10 MDAS IDs based on activity frequency")
                    st.dataframe(st.session_state.result1)

                    st.subheader("Result 2: Top 10 MDAS IDs with diversity")
                    st.dataframe(st.session_state.result2)

                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
