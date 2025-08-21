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

def initialize_openai_client(api_key: str, model: str):
    """Initialize the OpenAI client"""
    try:
        st.session_state.openai_client = OpenAIClient(api_key, model)
        return True
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        return False

def handle_api_error(response: str):
    """Check if the response is an error message and handle it"""
    if response.startswith("Error:"):
        st.error(response)
        return True
    return False

def get_api_key():
    """Get API key from Streamlit secrets or environment variable"""
    # Try to get from Streamlit secrets first
    try:
        if hasattr(st, 'secrets') and "openai_api_key" in st.secrets:
            return st.secrets["openai_api_key"]
    except Exception:
        pass
    
    # Try to get from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    # If neither works, ask user to input
    return None

def main():
    st.title("🤖 AI-Powered Data Analyst v1.0")
    st.markdown("Upload your CSV, ask questions about your data, and get instant analysis results.")
    
    # Initialize session state
    init_session_state()
    
    # API key setup
    api_key = get_api_key()
    
    if not api_key:
        st.sidebar.warning("⚠️ OpenAI API Key Required")
        st.sidebar.markdown("""
        **For Deployment:**
        Add your OpenAI API key as a secret in Streamlit Cloud:
        1. Go to your app settings
        2. Click on "Secrets"
        3. Add: `openai_api_key = "your-api-key-here"`
        
        **For Local Development:**
        Set environment variable: `OPENAI_API_KEY=your-api-key-here`
        """)
        
        # Fallback: Allow manual input (not recommended for production)
        api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
        
        if not api_key:
            st.info("Please add your OpenAI API key to continue.")
            return
    
    # Model selection in sidebar
    st.sidebar.header("Model Selection")
    model_options = [
        "gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"
    ]
    selected_model = st.sidebar.selectbox(
        "Select AI Model",
        options=model_options,
        index=model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
    )
    
    if st.session_state.openai_client is None or st.session_state.selected_model != selected_model:
        if initialize_openai_client(api_key, selected_model):
            st.session_state.selected_model = selected_model
            st.sidebar.success(f"✅ Using model: {selected_model}")
    
    # File upload section
    if not st.session_state.data_loaded:
        st.header("📁 Step 1: Upload Your Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type=["csv"], 
            help="Upload a CSV file to analyze your data"
        )
        
        if uploaded_file is not None:
            try:
                # Show progress
                with st.spinner("Loading data..."):
                    df = pd.read_csv(uploaded_file)
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                
                st.success(f"✅ Data loaded successfully! ({df.shape[0]} rows, {df.shape[1]} columns)")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error loading CSV file: {e}")
                st.info("Please make sure your file is a valid CSV format.")
    
    # If data is loaded, show data preview and analysis section
    if st.session_state.data_loaded and st.session_state.openai_client:
        # Data Preview Section
        st.header("📋 Data Preview")
        
        # Create tabs for better organization
        tab1, tab2, tab3 = st.tabs(["Sample Data", "Data Info", "Statistics"])
        
        with tab1:
            st.subheader("First 10 Rows")
            st.dataframe(st.session_state.df.head(10), use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Rows", st.session_state.df.shape[0])
                st.metric("Columns", st.session_state.df.shape[1])
                
            with col2:
                st.write("**Column Names:**")
                st.write(list(st.session_state.df.columns))
                
            st.write("**Data Types:**")
            st.dataframe(st.session_state.df.dtypes.to_frame('Data Type'), use_container_width=True)
            
            st.write("**Missing Values:**")
            missing_data = st.session_state.df.isnull().sum()
            if missing_data.sum() > 0:
                st.dataframe(missing_data[missing_data > 0].to_frame('Missing Count'), use_container_width=True)
            else:
                st.success("No missing values found!")
        
        with tab3:
            st.write("**Basic Statistics:**")
            st.dataframe(st.session_state.df.describe(), use_container_width=True)
        
        # Analysis Section
        st.header("🔍 Data Analysis")
        st.markdown("Ask questions about your data and get instant analysis with visualizations!")
        
        # Display chat messages
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and message.get("code"):
                    st.code(message["content"], language="python")
                    if "output" in message and message["output"]:
                        st.text("📊 Output:")
                        st.text(message["output"])
                    if "visualization" in message and message["visualization"]:
                        try:
                            st.image(message["visualization"], use_column_width=True)
                        except Exception as e:
                            st.error(f"Error displaying visualization: {e}")
                else:
                    st.markdown(message["content"])
        
        # User input
        if prompt := st.chat_input("Ask a question about your data... (e.g., 'Show me the distribution of values' or 'Create a correlation matrix')"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Check if this is a data analysis question
            if not is_data_analysis_question(prompt):
                with st.chat_message("assistant"):
                    response = "I can only answer questions about the data you've provided. Please ask about your dataset, such as statistical analysis, visualizations, or data insights."
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                return
            
            # Process the data question
            with st.chat_message("assistant"):
                with st.spinner("🤖 Analyzing your data..."):
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
                        if result.get('output'):
                            st.text("📊 Analysis Results:")
                            st.text(result['output'])
                        
                        # Check if there are visualizations
                        visualization = None
                        if 'visualization' in result and result['visualization']:
                            try:
                                # Convert BytesIO to image
                                if isinstance(result['visualization'], io.BytesIO):
                                    result['visualization'].seek(0)
                                    st.image(result['visualization'], use_column_width=True)
                                    visualization = result['visualization']
                            except Exception as e:
                                st.error(f"Error displaying visualization: {e}")
                        
                        # Add to message history
                        message_data = {
                            "role": "assistant", 
                            "content": clean_code,
                            "output": result.get('output', ''),
                            "code": True
                        }
                        
                        if visualization:
                            message_data["visualization"] = visualization
                            
                        st.session_state.messages.append(message_data)
                        
                    else:
                        st.error("❌ Error executing analysis code")
                        if result:
                            st.error(f"Error details: {result.get('error', 'Unknown error')}")
                            if result.get('output'):
                                st.text("Output before error:")
                                st.text(result.get('output', ''))
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"Error executing code: {result.get('error', 'Unknown error') if result else 'No result'}"
                        })
        
        # Add a reset button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()
