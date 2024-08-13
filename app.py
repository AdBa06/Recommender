import os
import pandas as pd
import streamlit as st
from dotenv import dotenv_values
from util import read_csv_file, get_llm_instance, create_agent, load_templates, build_prompt, invoke_llm, StreamHandler

def get_secrets():
    # Mock function to return secrets. Replace with actual secret fetching mechanism.
    return {'app_pw': 'mccy0108'}

def check_password():
    """Returns `True` if the user had a correct password."""    
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        pw = st.session_state["password"]
        secrets = get_secrets()

        if pw == secrets['app_pw']:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        # Show input form if password is not correct.
        with st.form(key='login'):
            st.text_input("Password", type="password", key="password")
            submit_button = st.form_submit_button('Submit', on_click=password_entered)
            if st.session_state["password_correct"] is False and submit_button:
                st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

# Custom CSS for background and input box styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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
    st.title("Citizen Recommender")
    st.subheader("Finding the best citizens to reach out to for Focus Group Discussions")
    st.markdown("**Disclaimer:** Please perform manual verification of AI-generated responses")

# FAQ SECTION
    with st.expander("FAQ"):
        st.markdown("""
        **Dataset usage**
        - Dataset needs to be in CSV format 
        - Ensure the column headings are at the first row of the dataset
        - For optimal accuracy, ensure dataset has column headings with meaning
        - Data is not collected or stored by the application

        **Prompting**
        - For ease of processing, input custom group size taking into account buffer on user end
        - Stacking queries will lead to longer processing times 
        - Encouraged to prompt as directly as possible
        
        **Interface**
        - All key sections are expandible or collapsible
        - If model fails due to an error, a rerun usually fixes the issue
        """)

    if check_password():
        df = None
        # Test Mode button
        if st.button("Test Mode"):
            test_mode = True
        else:
            test_mode = False

        if not test_mode:
            # Upload CSV
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                df = read_csv_file(uploaded_file)
        else:
            # Load the CSV file for test mode
            df = read_csv_file("merged_df.csv")
            st.success("Test Mode: Loaded merged_df.csv successfully!")

        if df is not None:
            st.success("CSV file read successfully!")
            with st.expander("Data Preview"):
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
            query = st.text_input("Enter your query:", placeholder="e.g., I want to talk to active swimmers about their experiences")

            # Add group size selection
            st.markdown("**Select Group Size:**")
            group_size = None
            group_size = st.number_input("Enter custom group size", min_value=1, step=1)

            if query and group_size and st.button("Search"):
                # Retry the prompt in case of error
                try:
                    # Load templates
                    templates = load_templates()

                    # Include group size in the prompt
                    prompt = build_prompt(templates, f"{query} for a group of {group_size} people", column_names)

                    # Expander for real-time thought process (streaming content)
                    with st.expander("AI Thought Process"):
                        thought_process_placeholder = st.empty()

                        # Create a StreamHandler instance
                        stream_handler = StreamHandler(thought_process_placeholder, display_method='markdown')

                        # Retry the prompt
                        response = retry_prompt(agent, prompt, stream_handler)

                    if response:
                        user_query = response.get('input', 'User query not found')
                        output = response.get('output', 'Output not found')

                        st.markdown("**Final Answer**")
                        st.markdown(f'<div class="message-box">{output}</div>', unsafe_allow_html=True)

                        # Read the results from the CSV files
                        result1 = pd.read_csv('result1.csv')
                        result2 = pd.read_csv('result2.csv')

                        # Store results in session state to avoid re-reading the files unnecessarily
                        st.session_state.result1 = result1
                        st.session_state.result2 = result2

                        # Display the results as tables under expandable sections
                        with st.expander("First recommended group to talk to"):
                            st.dataframe(st.session_state.result1)

                        with st.expander("Second recommended group to talk to"):
                            st.dataframe(st.session_state.result2)

                    else:
                        st.warning("No response from the model.")

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()
