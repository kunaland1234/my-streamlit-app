import streamlit as st
import pandas as pd
import os
from openai_client import OpenAIClient
from data_processor import DataProcessor
from utils import is_data_analysis_question, clean_generated_code
import json
from PIL import Image
import io

# Set up the page
st.set_page_config(
    page_title="AI-Powered Data Analyst v1.0",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "df" not in st.session_state:
        st.session_state.df = None
    if "openai_client" not in st.session_state:
        st.session_state.openai_client = None
    if "data_processor" not in st.session_state:
        st.session_state.data_processor = DataProcessor()
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gpt-4o-mini"
    if "api_key_configured" not in st.session_state:
        st.session_state.api_key_configured = False

def initialize_openai_client(api_key: str, model: str):
    """Initialize the OpenAI client"""
    try:
        st.session_state.openai_client = OpenAIClient(api_key, model)
        return True
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        # Add more detailed error information
        import traceback
        st.error("Full error traceback:")
        st.code(traceback.format_exc())
        return False

def handle_api_error(response: str):
    """Check if the response is an error message and handle it"""
    if response.startswith("Error:"):
        st.error(response)
        return True
    return False

def main():
    st.title("🤖 AI-Powered Data Analyst v1.0")
    st.markdown("Upload your CSV, ask questions about your data, and get instant analysis results.")
    
    # Initialize session state
    init_session_state()
    
    # API key setup - using secrets.toml or environment variable
    api_key = None
    if "openai_api_key" in st.secrets:
        api_key = st.secrets["openai_api_key"]
        st.session_state.api_key_configured = True
    elif os.environ.get("OPENAI_API_KEY"):
        api_key = os.environ.get("OPENAI_API_KEY")
        st.session_state.api_key_configured = True
    else:
        st.sidebar.error("OpenAI API key not found")
        st.info("Please add your OpenAI API key to Streamlit secrets or set OPENAI_API_KEY environment variable")
        st.session_state.api_key_configured = False
    
    if not st.session_state.api_key_configured:
        return
    
    # Model selection in sidebar
    st.sidebar.header("Model Selection")
    model_options = [
        "gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4",
        "gpt-3.5-turbo"
    ]
    selected_model = st.sidebar.selectbox(
        "Select AI Model",
        options=model_options,
        index=model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
    )
    
    if st.session_state.openai_client is None or st.session_state.selected_model != selected_model:
        if initialize_openai_client(api_key, selected_model):
            st.session_state.selected_model = selected_model
            st.sidebar.success(f"Using model: {selected_model}")
    
    # File upload section
    if not st.session_state.data_loaded:
        st.header("Step 1: Upload Your Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success("Data loaded successfully!")
                
            except Exception as e:
                st.error(f"Error loading CSV file: {e}")
    
    # If data is loaded, show data preview and analysis section
    if st.session_state.data_loaded and st.session_state.openai_client:
        # Data Preview Section
        st.header("Data Preview")
        
        # First 10 rows
        st.subheader("First 10 Rows")
        st.dataframe(st.session_state.df.head(10))
        
        # Data Information in columns
        st.subheader("Data Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Shape**")
            st.write(st.session_state.df.shape)
            
        with col2:
            st.write("**Columns**")
            st.write(list(st.session_state.df.columns))
            
        with col3:
            st.write("**Data Types**")
            st.write(st.session_state.df.dtypes.astype(str))
        
        # Missing values and basic stats
        col4, col5 = st.columns(2)
        
        with col4:
            st.write("**Missing Values**")
            st.write(st.session_state.df.isnull().sum())
            
        with col5:
            st.write("**Basic Statistics**")
            st.write(st.session_state.df.describe())
        
        # Analysis Section
        st.header("Data Analysis")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and "code" in message:
                    st.code(message["content"], language="python")
                    if "output" in message and message["output"]:
                        st.text("Output:")
                        st.text(message["output"])
                    if "visualization" in message and message["visualization"]:
                        try:
                            if isinstance(message["visualization"], io.BytesIO):
                                message["visualization"].seek(0)
                                st.image(message["visualization"])
                        except Exception as e:
                            st.error(f"Error displaying image: {e}")
                else:
                    st.markdown(message["content"])
        
        # User input
        if prompt := st.chat_input("Ask a question about your data..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Check if this is a data analysis question
            if not is_data_analysis_question(prompt):
                with st.chat_message("assistant"):
                    response = "I can only answer questions about the data you've provided. Please ask about your dataset."
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                return
            
            # Process the data question
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your data..."):
                    # Get analysis code from AI
                    code_response = st.session_state.openai_client.generate_analysis_code(
                        prompt, st.session_state.df
                    )
                    
                    if handle_api_error(code_response):
                        return
                    
                    # Clean the code (remove markdown if present)
                    clean_code = clean_generated_code(code_response)
                    
                    # Display the generated code
                    st.code(clean_code, language="python")
                    
                    # Execute the code
                    result = st.session_state.data_processor.safe_execute_code(
                        clean_code, st.session_state.df
                    )
                    
                    if result and result['success']:
                        # Display output
                        if result['output']:
                            st.text("Output:")
                            st.text(result['output'])
                        
                        # Check if there are visualizations
                        visualization = None
                        if 'visualization' in result and result['visualization']:
                            try:
                                # Convert BytesIO to image
                                if isinstance(result['visualization'], io.BytesIO):
                                    result['visualization'].seek(0)
                                    st.image(result['visualization'])
                                    visualization = result['visualization']
                            except Exception as e:
                                st.error(f"Error displaying image: {e}")
                        
                        # Add to message history
                        message_data = {
                            "role": "assistant", 
                            "content": clean_code,
                            "output": result['output'],
                            "code": True
                        }
                        
                        if visualization:
                            message_data["visualization"] = visualization
                            
                        st.session_state.messages.append(message_data)
                        
                    else:
                        st.error("Error executing analysis code")
                        if result:
                            st.error(f"Error: {result.get('error', 'Unknown error')}")
                            if result.get('output'):
                                st.text("Output before error:")
                                st.text(result.get('output'))
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"Error executing code: {result.get('error', 'Unknown error') if result else 'No result'}"
                        })

if __name__ == "__main__":
    main()
