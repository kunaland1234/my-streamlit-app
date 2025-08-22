import streamlit as st
import pandas as pd
import os
from openai_client import OpenAIClient
from data_processor import DataProcessor
from utils import clean_generated_code
import json
from PIL import Image
import io

# Set up the page
st.set_page_config(
    page_title="AI-Powered Data Analyst Pro v1.1",
    page_icon="🤖",
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
    if "analysis_count" not in st.session_state:
        st.session_state.analysis_count = 0

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

def display_data_insights(df: pd.DataFrame):
    """Display comprehensive data insights"""
    st.header("📊 Data Insights")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    
    with col2:
        st.metric("Total Columns", len(df.columns))
    
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    with col4:
        memory_usage = df.memory_usage(deep=True).sum()
        st.metric("Memory Usage", f"{memory_usage / 1024 / 1024:.2f} MB")
    
    # Data quality overview
    st.subheader("Data Quality Overview")
    
    # Column Information
    st.write("**Column Information**")
    col_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        unique_count = df[col].nunique()
        
        col_info.append({
            'Column': col,
            'Type': dtype,
            'Nulls': f"{null_count} ({null_pct:.1f}%)",
            'Unique': unique_count
        })
    
    col_df = pd.DataFrame(col_info)
    st.dataframe(col_df, use_container_width=True)
    
    # Sample Data - Displayed separately
    st.write("**Sample Data**")
    st.dataframe(df.head(), use_container_width=True)

def main():
    st.title("🤖 AI-Powered Data Analyst Pro v1.1")
    st.markdown("### Upload your CSV and have intelligent conversations about your data!")
    
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
    
    # Enhanced sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_options = [
            "gpt-4o-mini", "gpt-4o", "gpt-4o-2024-11-20", "gpt-4-turbo", 
            "gpt-4", "gpt-3.5-turbo"
        ]
        selected_model = st.selectbox(
            "🧠 Select AI Model",
            options=model_options,
            index=model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0,
            help="Choose the AI model for analysis. GPT-4o models are recommended for better code generation."
        )
        
        if st.session_state.openai_client is None or st.session_state.selected_model != selected_model:
            if initialize_openai_client(api_key, selected_model):
                st.session_state.selected_model = selected_model
                st.success(f"✅ Using: {selected_model}")
        
        # Analysis statistics
        if st.session_state.data_loaded:
            st.header("📈 Session Stats")
            st.metric("Analyses Completed", st.session_state.analysis_count)
            
            if st.button("🔄 Reset Conversation", help="Clear conversation history"):
                st.session_state.messages = []
                if st.session_state.openai_client:
                    st.session_state.openai_client.clear_conversation_history()
                st.rerun()
            
            if st.button("🗑️ Clear All Data", help="Remove dataset and reset everything"):
                for key in ['messages', 'data_loaded', 'df', 'analysis_count']:
                    st.session_state[key] = [] if key == 'messages' else (False if key == 'data_loaded' else (None if key == 'df' else 0))
                if st.session_state.openai_client:
                    st.session_state.openai_client.clear_conversation_history()
                st.rerun()
    
    # File upload section
    if not st.session_state.data_loaded:
        st.header("📁 Step 1: Upload Your Dataset")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose a CSV file", 
                type="csv",
                help="Upload your CSV file to start analyzing your data with AI"
            )
            
            if uploaded_file is not None:
                try:
                    # Show loading spinner
                    with st.spinner("Loading your data..."):
                        df = pd.read_csv(uploaded_file)
                        
                        # Validate the dataframe
                        validation = st.session_state.data_processor.validate_dataframe(df)
                        
                        if not validation['valid']:
                            st.error(f"Data validation failed: {validation['error']}")
                            return
                        
                        st.session_state.df = df
                        st.session_state.data_loaded = True
                        
                        # Clear conversation history for new dataset
                        st.session_state.messages = []
                        st.session_state.analysis_count = 0
                        if st.session_state.openai_client:
                            st.session_state.openai_client.clear_conversation_history()
                        
                        st.success(f"✅ Data loaded successfully! {len(df):,} rows × {len(df.columns)} columns")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error loading CSV file: {e}")
                    st.info("Make sure your CSV file is properly formatted and not corrupted.")
        
        with col2:
            st.info("""**Tips for best results:**
            - Ensure your CSV has clear column headers
            - Remove any completely empty rows/columns
            - Check for proper encoding (UTF-8 recommended)
            - File size should be reasonable for processing
            """)
    
    # Main analysis section
    if st.session_state.data_loaded and st.session_state.openai_client:
        # Display data insights
        display_data_insights(st.session_state.df)
        
        st.header("🔍 Interactive Data Analysis")
        st.markdown("Ask questions about your data in natural language. I can handle follow-up questions and remember our conversation!")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    if "error_message" in message:
                        st.error(message["content"])
                    elif "code" in message:
                        st.code(message["content"], language="python")
                        if "output" in message and message["output"]:
                            st.text("📋 Output:")
                            st.text(message["output"])
                        if "visualization" in message and message["visualization"]:
                            try:
                                st.image(message["visualization"], use_column_width=True)
                            except Exception as e:
                                st.error(f"Error displaying image: {e}")
                    else:
                        st.markdown(message["content"])
                else:
                    st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about your data... 💬"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Check if this is a data-related question using enhanced logic
            try:
                is_data_related, reason = st.session_state.openai_client.is_data_related_question(
                    prompt, st.session_state.df
                )
                
                if is_data_related:
                    # Show thinking indicator
                    with st.chat_message("assistant"):
                        with st.spinner("🤔 Analyzing your data..."):
                            # Generate analysis code
                            code = st.session_state.openai_client.generate_analysis_code(
                                prompt, 
                                st.session_state.df, 
                                st.session_state.messages
                            )
                            
                            # Clean the generated code
                            cleaned_code = clean_generated_code(code)
                            
                            # Execute the code
                            result = st.session_state.data_processor.safe_execute_code(
                                cleaned_code, st.session_state.df
                            )
                            
                            # Handle the result
                            if result['success']:
                                # Display code
                                st.code(cleaned_code, language="python")
                                
                                # Display output if any
                                if result['output']:
                                    st.text("📋 Output:")
                                    st.text(result['output'])
                                
                                # Display visualization if any
                                if result['visualization']:
                                    st.image(result['visualization'], use_column_width=True)
                                
                                # Add to message history
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": cleaned_code,
                                    "output": result['output'],
                                    "visualization": result['visualization']
                                })
                                
                                # Increment analysis count
                                st.session_state.analysis_count += 1
                            else:
                                # Handle error
                                error_msg = f"❌ Error executing code: {result['error']}"
                                st.error(error_msg)
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": error_msg,
                                    "error_message": True
                                })
                else:
                    # Not data-related
                    response = "I'm designed to help with data analysis questions. Please ask something about your dataset."
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.info(response)
                    
            except Exception as e:
                error_msg = f"❌ Error processing your request: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg,
                    "error_message": True
                })

if __name__ == "__main__":
    main()
