import re
import pandas as pd
from typing import Dict, Any, List

def clean_generated_code(code: str) -> str:
    """Enhanced code cleaning with better handling of different formats"""
    if not code or not isinstance(code, str):
        return ""
    
    # Remove markdown code blocks
    if "```python" in code:
        # Extract code between ```python and ```
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, code, re.DOTALL)
        if matches:
            code = matches[0]
    elif "```" in code:
        # Extract code between ``` and ```
        pattern = r'```\s*(.*?)\s*```'
        matches = re.findall(pattern, code, re.DOTALL)
        if matches:
            code = matches[0]
    
    # Remove any remaining markdown artifacts
    code = re.sub(r'^```.*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'```$', '', code, flags=re.MULTILINE)
    
    # Clean up common issues
    lines = code.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Skip empty lines at the beginning
        if not cleaned_lines and not line.strip():
            continue
        
        # Remove leading/trailing whitespace but preserve indentation
        cleaned_line = line.rstrip()
        
        # Skip lines that are just markdown or comments that don't add value
        if cleaned_line.strip().startswith('#') and any(word in cleaned_line.lower() for word in ['here', 'code', 'analysis', 'let', 'now']):
            continue
            
        cleaned_lines.append(cleaned_line)
    
    # Remove trailing empty lines
    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()
    
    return '\n'.join(cleaned_lines)

def extract_column_mentions(question: str, df: pd.DataFrame) -> List[str]:
    """Extract column names mentioned in the question"""
    question_lower = question.lower()
    mentioned_columns = []
    
    for col in df.columns:
        # Check for exact matches (case insensitive)
        if col.lower() in question_lower:
            mentioned_columns.append(col)
        
        # Check for partial matches for multi-word columns
        col_words = col.lower().split('_')
        if len(col_words) > 1:
            if all(word in question_lower for word in col_words):
                mentioned_columns.append(col)
    
    return list(set(mentioned_columns))  # Remove duplicates

def extract_value_mentions(question: str, df: pd.DataFrame) -> Dict[str, List[str]]:
    """Extract specific values mentioned in the question that exist in the data"""
    question_lower = question.lower()
    mentioned_values = {}
    
    for col in df.columns:
        if df[col].dtype == 'object':  # Text/categorical columns
            unique_vals = df[col].dropna().unique()
            col_mentions = []
            
            for val in unique_vals:
                val_str = str(val).lower()
                if val_str in question_lower and len(val_str) > 2:  # Avoid short matches
                    col_mentions.append(str(val))
            
            if col_mentions:
                mentioned_values[col] = col_mentions
    
    return mentioned_values

def suggest_analysis_types(question: str, df: pd.DataFrame) -> List[str]:
    """Suggest appropriate analysis types based on question and data structure"""
    question_lower = question.lower()
    suggestions = []
    
    # Basic analysis suggestions
    basic_keywords = {
        'summary': ['summary statistics', 'data overview', 'basic info'],
        'missing': ['missing values analysis', 'data quality check'],
        'distribution': ['distribution analysis', 'histograms'],
        'correlation': ['correlation analysis', 'relationship analysis'],
        'outliers': ['outlier detection', 'anomaly analysis']
    }
    
    for keyword, analyses in basic_keywords.items():
        if keyword in question_lower:
            suggestions.extend(analyses)
    
    # Visualization suggestions
    viz_keywords = {
        'plot': 'visualization',
        'chart': 'chart creation',
        'graph': 'graph generation',
        'visual': 'data visualization'
    }
    
    for keyword, analysis in viz_keywords.items():
        if keyword in question_lower:
            suggestions.append(analysis)
    
    # Column-specific suggestions
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if 'compare' in question_lower and len(numeric_cols) >= 2:
        suggestions.append('comparative analysis')
    
    if 'group' in question_lower and len(categorical_cols) > 0:
        suggestions.append('grouped analysis')
    
    return list(set(suggestions))  # Remove duplicates

def format_error_message(error: str, context: Dict[str, Any] = None) -> str:
    """Format error messages to be more user-friendly"""
    error_lower = error.lower()
    
    # Common error patterns and their user-friendly messages
    if 'keyerror' in error_lower:
        return "❌ Column name not found. Please check the spelling and use exact column names (case-sensitive)."
    
    if 'indexerror' in error_lower:
        return "❌ Index out of range. The data might not have enough rows for this operation."
    
    if 'typeerror' in error_lower:
        return "❌ Data type mismatch. This operation might not be suitable for the selected columns."
    
    if 'valueerror' in error_lower:
        return "❌ Invalid value encountered. Please check if the data contains the expected values."
    
    if 'import' in error_lower:
        return "❌ Required library not available. Some advanced features might not be supported."
    
    if 'syntax' in error_lower:
        return "❌ Code syntax error. Please try rephrasing your question."
    
    # Return original error if no pattern matches
    return f"❌ Error: {error}"

def get_helpful_suggestions(df: pd.DataFrame, error_context: str = None) -> List[str]:
    """Provide helpful suggestions based on the dataset and any errors"""
    suggestions = [
        "📊 Try: 'Show me a summary of the data'",
        "🔍 Try: 'What columns are available?'",
        "📈 Try: 'Create a simple visualization'",
        "🧮 Try: 'Show basic statistics'"
    ]
    
    # Add column-specific suggestions
    if len(df.columns) > 0:
        first_col = df.columns[0]
        suggestions.append(f"📋 Try: 'Analyze the {first_col} column'")
    
    # Add suggestions based on data types
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 1:
        suggestions.append(f"🔗 Try: 'Show correlation between numeric columns'")
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        suggestions.append(f"📊 Try: 'Show distribution of {categorical_cols[0]}'")
    
    return suggestions
