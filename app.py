import os
import pandas as pd
import streamlit as st
from dotenv import dotenv_values
from util import read_csv_file, get_llm_instance, create_agent, load_templates, build_prompt, invoke_llm, StreamHandler

# Custom CSS for background and input box styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def set_custom_css():
    css = """
    <style>
    .stApp {
        background-color: #EBDAB7;
        color: black; /* Set default text color to black */
    }
    .stTextInput input {
        background-color: white;
        color: black;
    }
    header { /* This will target the header and make it match the background */
        background-color: #F5E1FF;
        color: black; 
    }
    .message-box {
        background-color: #fff;
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        font-size: 16px;
        color: black;
        max-width: 100%;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def retry_prompt(agent, prompt, stream_handler):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with st.spinner(f"Processing... Attempt {attempt + 1}/{max_retries}"):
                response = invoke_llm(agent, prompt, stream_handler)
                return response
        except Exception as e:
            st.error(f"An error occurred: {e}")
            if attempt < max_retries - 1:
                st.warning("Retrying...")
            else:
                st.error("Failed after several attempts.")
                return None

def main():
    st.title("CitizenGoWhere")
    st.subheader("Finding the best citizens to reach out to for Focus Group Discussions")
    st.markdown("**Disclaimer:** Please perform manual verification of AI-generated responses")

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
                st.markdown(f'<div class="message-box">{chat}</div>', unsafe_allow_html=True)

            # Define the user query
            query = st.text_input("Enter your query:")

            if query:
                # Retry the prompt in case of error
                try:
                    # Load templates
                    templates = load_templates()

                    # Build the final prompt
                    prompt = build_prompt(templates, query, column_names)

                    # Placeholder for real-time thought process
                    thought_process_placeholder = st.empty()

                    # Create a StreamHandler instance
                    stream_handler = StreamHandler(thought_process_placeholder, display_method='markdown')

                    # Retry the prompt
                    response = retry_prompt(agent, prompt, stream_handler)

                    if response:
                        # Real-time processing and thought display
                        # st.subheader("AI Thought Process")
                        # st.markdown(f'<div class="message-box">{response}</div>', unsafe_allow_html=True)

                        user_query = response.get('input', 'User query not found')
                        output = response.get('output', 'Output not found')

                        st.subheader("AI Response:")
                        st.markdown(f'<div class="message-box">{output}</div>', unsafe_allow_html=True)

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
                    else:
                        st.warning("No response from the model.")

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()