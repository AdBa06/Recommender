import os
import pandas as pd
import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from dotenv import load_dotenv, dotenv_values
from io import BytesIO

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

CSV_PROMPT_PREFIX = """
You are an AI assistant. I need your help in analyzing a dataset of user activities.

Task:
First, import pandas as pd. Then, do the tasks below:

1. Set pandas display options to show all columns.
2. Retrieve column names from the dataset.
3. Identify and list 10 MDAS IDs based on the given criteria through multiple methods.

Example:
If the prompt is "I want to have a discussion with people who love swimming," follow these steps:

Method 1: Look for the most Active Swimmers
Do not directly filter by "Swimming", you have to look for related words like "swimming"
or "swim" or even "pool", you can make this decision yourself.
Explanation:
For activity level, look at the Activity column. Each activity a person does is listed with a 
number representing how many times they took part in it. For example, if a person has "swimming: 20, Food: 2", it means
they swam 20 times and were involved in food activites twice. You just need to find the people with the
largest values corresponding to the activities, you do not need to count anything yourself. 

Pick those with the largest numbers once you found the people who do the activity. Pick 10 MDAS IDs.

Method 2:
Pick those who have that activity, but pick people of different ethnicities for diversity. For this method,
the number of times someone did that activity does not matter, feel free to include all ethnicities even 
if it means they only did it once. Include multiple members of each ethnicity to make up 10 total IDs

Example Prompt: 
I want to have a discussion with people who love swimming.

Example Response:
Here is a list of people you might want to speak to as they swim very frequently and have a high stake in Swimming
MDAS ID: 123, 456, 789, ...

Alternatively, here is another list of people you could speak to in order for having diversity and differing perspectives.
MDAS ID: 222, 3333, 444 ...

Export the 10 rows corresponding to these MDAS IDs to a csv sheet called 'result1.csv' and 'result2.csv' for the two different cases.
do this by creating a dataframe with the relevant rows of IDs, 
then use the pandas to_csv function.
once done, add that you have exported the results in your response

Remember that the two methods are suggestions, you are free to look at it in terms of age, ethnicity, or any
other method you believe would be very relevant depending on the prompt.
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

# Main function to read CSV, invoke LLM, and display response
def main():
    st.title("CSV Analysis with AI Assistant")

    # Upload CSV
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = read_csv_file(uploaded_file)
        if df is not None:
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            # Initialize the Azure ChatOpenAI model
            model = get_llm_instance()

            # Create a Pandas DataFrame agent
            agent = create_pandas_dataframe_agent(llm=model, df=df, verbose=True, allow_dangerous_code=True, handle_parsing_errors=True)

            # Define the user query
            query = st.text_input("Enter your query:")
            
            if query:
                try:
                    response = agent.invoke(CSV_PROMPT_PREFIX + query + CSV_PROMPT_SUFFIX + CSV_PROMPT_OUT)
                    st.markdown(response)

                    # Save the result1.csv and result2.csv as BytesIO objects to be downloadable
                    result1_buffer = BytesIO()
                    result2_buffer = BytesIO()
                    result1 = pd.read_csv('result1.csv')
                    result2 = pd.read_csv('result2.csv')
                    result1.to_csv(result1_buffer, index=False)
                    result2.to_csv(result2_buffer, index=False)

                    st.download_button(
                        label="Download result1.csv",
                        data=result1_buffer.getvalue(),
                        file_name='result1.csv',
                        mime='text/csv',
                    )

                    st.download_button(
                        label="Download result2.csv",
                        data=result2_buffer.getvalue(),
                        file_name='result2.csv',
                        mime='text/csv',
                    )

                except Exception as e:
                    st.error(f"An error occurred during the agent invocation: {e}")

if __name__ == "__main__":
    main()
