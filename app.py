import os
import pandas as pd
import re
import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from dotenv import load_dotenv, dotenv_values

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
    st.write(f"Column names: {column_names}")
    return column_names

# Define the templates
def load_templates():
    templates = {
        "prefix": """
        Import pandas as pd. I need your help in analyzing a dataset of user activities.
        """,
        "step1": """
        Discern whether in the user query, it is asking for activity or facility related questions.
        If the query asks for things like "Swimming pools", "Gyms", they are interested in facilities.
        If it asks for things like "social causes", "group dance" etc, they are interested in activities.
        Use this knowledge when deciding the next step.
        """,
        "step2": """
        Recognize that the Activities column in the df is formatted as "Activity1: Frequency, Activity2: Frequency..." and so on,
        so you might have to ensure you are able to read within the entire string when looking for activities. 
        Do not split the activities column, just read from within the larger string to see if anything relevant exists anywhere within,
        and then identify the number right after the relevant activity.
        You need to come up with a list of activities from all the activities in the dataframe that might be related to the prompt. 
        For example here is a good method that works. 
        To identify activities related to water, look for keywords such as "swim", "water", "pool", "dive", "aquatic", etc. in the activities.
        Now, filter the dataset to only keep the rows that have relevant activities.
        """,
        "step3": """
        Identify and list 10 MDAS IDs based on the given criteria through multiple methods.
        Method 1: Look through the entire activity column, 
        The strings in the activity column are formatted as "Activity 1: Frequency, Activity 2: Frequency etc".
        So you just need to look at the frequency of the activity based on your filtered list to decide who is the most active for a given activity.
        Do not filter by anything else other than the number right after the activity within the string.
        Pick those with the largest numbers once you found the people who do the activity. Pick 10 MDAS IDs.
        For example, use this method: df_badminton['badminton_freq'] = df_badminton['Activities'].str.extract('Badminton Court: (\d+)', expand=False).fillna(0).astype(int)   
       
        
        Method 2: Pick those who have that activity, but pick people of different ethnicities for diversity. Try to have at least 2 Chinese, 2 Malays, 2 Indians, and 2 Others, but in cases where not all races exist for an activity, it is ok to have more people of races that do exist.
        Example Prompt: I want to have a discussion with people who love martial arts.
        Example Response:
        Here is a list of people you might want to speak to as they frequently engage in martial arts and have a high stake in it:
        MDAS ID: 123, 456, 789, ...
        Alternatively, here is another list of people you could speak to in order to have diversity and differing perspectives:
        MDAS ID: 222, 333, 444 ...
        """,
        "step4": """
        Export the 10 rows corresponding to these MDAS IDs for the two different cases in 2 different csv files, 
        which are result1.csv and result2.csv. Do not create any other csv file names.
        """,
        "guidelines": """
        - Use the python_repl_ast tool to execute the python commands
        - always do one command at a time if using python, you cannot handle multiple commands at the same time
        - any variables declared need to persist, and be global throughout the entire client execution
        - If the query is about facilities, do the same thing but with the facilities column instead of activities column for analysis. 
        - Always try another method before giving the final answer.
        - Reflect on the answers from both methods and ensure they are consistent.
        - Explain the process and reasoning in your final answer.
        - Use Markdown for formatting the final response.
        """,
        "output_format": """
        Please provide a brief overview of the methods and inferences you used to find 10 MDAS IDs according to the context
        """
    }
    return templates

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
        question +
        'using tool python_repl_ast'
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
            st.dataframe(df.head())

            # Perform initial setup
            column_names = initial_setup(df)
            
            # Initialize the Azure ChatOpenAI model
            model = get_llm_instance()

            # Create a Pandas DataFrame agent
            agent = create_pandas_dataframe_agent(llm=model, df=df, verbose=True, allow_dangerous_code=True, handle_parsing_errors=True)

            # Initialize chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            if 'result1' not in st.session_state:
                st.session_state.result1 = None
            if 'result2' not in st.session_state:
                st.session_state.result2 = None

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

                        response = agent.invoke(prompt)
                        st.session_state.chat_history.append(f"**User:** {query}\n\n**AI:** {response}\n\n")
                        for chat in st.session_state.chat_history:
                            st.markdown(chat)

                        # Display the relevant rows directly within the app
                        if st.session_state.result1 is None or st.session_state.result2 is None:
                            result1 = pd.read_csv('result1.csv')
                            result2 = pd.read_csv('result2.csv')
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
