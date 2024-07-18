import os
import pandas as pd
import streamlit as st
from dotenv import dotenv_values
from util import read_csv_file, get_llm_instance, create_agent, load_templates, build_prompt, invoke_llm

# Custom CSS for background and input box styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def set_custom_css():
    css = """
    <style>
    .stApp {
        background-color: #F5E1FF;
    }
    .stTextInput input {
        background-color: white;
        color: black;
    }
    header { /* This will target the header and make it match the background */
        background-color: #F5E1FF;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def main():
    st.title("MCCY Citizen Recommender")
    st.subheader("Finding the best citizens to reach out to for Focus group discussions")
    st.markdown("**Disclaimer:** The recommendations are to be evaluated with manual checking, and not to be taken as 100% true")

    # Apply custom CSS
    set_custom_css()

    # Upload CSV
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = read_csv_file(uploaded_file)
        if df is not None:
            st.success("CSV file read successfully!")
            st.write("Data Preview:")
            st.dataframe(df)  # Display the entire dataframe

            # Perform initial setup
            column_names = df.columns.tolist()

            # Initialize the Azure ChatOpenAI model
            model = get_llm_instance()

            # Create a Pandas DataFrame agent
            agent = create_agent(llm=model, df=df)

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
                        response = invoke_llm(agent, prompt)

                        # Real-time processing and thought display
                        st.subheader("AI Thought Process")
                        st.markdown(response)

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
                    st.subheader("First recommended group to talk to")
                    st.dataframe(st.session_state.result1)

                    st.subheader("Second recommended group to talk to")
                    st.dataframe(st.session_state.result2)

                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
