from openai import OpenAI
import os
import pandas as pd
from typing import List, Dict, Any
import json

class OpenAIClient:
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        if api_key:
            self.client = OpenAI(api_key=api_key)
        elif os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError("OpenAI API key not provided")
        self.model = model
        self.conversation_history = []
        self.data_context = {}
    
    def update_data_context(self, df: pd.DataFrame):
        """Update the data context with comprehensive information about the dataset"""
        self.data_context = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample": df.head(5).to_dict('records'),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "unique_values": {col: df[col].nunique() for col in df.columns},
            "basic_stats": df.describe().to_dict() if not df.empty else {}
        }
    
    def is_data_related_question(self, question: str, df: pd.DataFrame) -> tuple[bool, str]:
        """Enhanced logic to determine if question is data-related with context awareness"""
        question_lower = question.lower()
        
        # Check if question mentions specific columns from the dataset
        column_mentions = [col.lower() for col in df.columns if col.lower() in question_lower]
        
        # Enhanced data analysis keywords
        data_analysis_keywords = [
            'analy', 'data', 'dataset', 'statistic', 'model', 'pattern', 'trend',
            'correlation', 'regression', 'hypothesis', 'test', 'visualization',
            'plot', 'chart', 'graph', 'insight', 'missing', 'null', 'outlier',
            'clean', 'process', 'transform', 'feature', 'variable', 'column',
            'average', 'mean', 'sum', 'count', 'max', 'min', 'median', 'mode',
            't-test', 'chi-square', 'anova', 'p-value', 'standard deviation',
            'variance', 'distribution', 'histogram', 'box plot', 'scatter plot',
            'show', 'display', 'find', 'what', 'how', 'why', 'compare', 'filter',
            'sort', 'group', 'aggregate', 'summary', 'describe', 'explore',
            'relationship', 'difference', 'similar', 'top', 'bottom', 'highest',
            'lowest', 'best', 'worst', 'most', 'least', 'total', 'percentage'
        ]
        
        # Contextual keywords (words that might be data-related in context)
        contextual_keywords = [
            'this', 'these', 'those', 'it', 'them', 'result', 'output', 'above',
            'previous', 'earlier', 'before', 'more', 'tell me', 'explain'
        ]
        
        # Check for data analysis keywords
        has_analysis_keywords = any(keyword in question_lower for keyword in data_analysis_keywords)
        
        # Check for contextual keywords (suggesting follow-up questions)
        has_contextual_keywords = any(keyword in question_lower for keyword in contextual_keywords)
        
        # Check if it's a follow-up question based on conversation history
        is_followup = len(self.conversation_history) > 0 and has_contextual_keywords
        
        # Check for specific value mentions that might be in the data
        value_mentions = []
        for col in df.columns:
            if df[col].dtype == 'object':  # categorical columns
                unique_vals = df[col].dropna().unique()
                for val in unique_vals:
                    if str(val).lower() in question_lower:
                        value_mentions.append(val)
        
        # Decision logic
        if column_mentions:
            return True, f"Question mentions dataset columns: {column_mentions}"
        elif value_mentions:
            return True, f"Question mentions data values: {value_mentions}"
        elif has_analysis_keywords:
            return True, "Contains data analysis keywords"
        elif is_followup:
            return True, "Appears to be a follow-up question about previous analysis"
        elif has_contextual_keywords and len(self.conversation_history) > 0:
            return True, "Contextual question with conversation history"
        else:
            # If none of the above, it's likely not data-related
            return False, "No clear data analysis intent detected"
    
    def generate_analysis_code(self, question: str, df: pd.DataFrame, conversation_context: List[Dict] = None) -> str:
        """Generate Python code with conversation context and enhanced understanding"""
        
        # Update data context
        self.update_data_context(df)
        
        # Build conversation context
        context_messages = []
        if conversation_context:
            recent_context = conversation_context[-6:]  # Last 6 messages for context
            for msg in recent_context:
                if msg["role"] == "user":
                    context_messages.append(f"User asked: {msg['content']}")
                elif msg["role"] == "assistant" and "output" in msg:
                    context_messages.append(f"Previous analysis result: {msg.get('output', '')[:200]}...")
        
        conversation_context_str = "\n".join(context_messages) if context_messages else "No previous conversation"
        
        system_prompt = f"""
        You are an expert data analyst AI that ONLY analyzes the provided dataset. You have perfect memory of the conversation and can answer follow-up questions intelligently.

        CRITICAL RULES:
        1. You can ONLY answer questions about the provided dataset
        2. NEVER use external knowledge - only analyze the actual data provided
        3. The dataframe is called `df` and contains the user's data
        4. Always consider the conversation context for follow-up questions
        5. If asked "why" or "explain" about results, analyze the data to find patterns
        6. Generate executable Python code that directly answers the question
        7. Include comprehensive error handling and data validation
        8. Use clear variable names and detailed comments
        9. Always print results clearly with context
        10. For visualizations, use matplotlib/seaborn with proper styling
        11. Always import necessary libraries at the top (pandas, numpy, matplotlib, seaborn)
        12. For value counts visualization, use bar plots with proper labels

        CONVERSATION CONTEXT:
        {conversation_context_str}

        DATA INFORMATION:
        - Shape: {self.data_context['shape']}
        - Columns: {self.data_context['columns']}
        - Data Types: {self.data_context['dtypes']}
        - Numeric Columns: {self.data_context['numeric_columns']}
        - Categorical Columns: {self.data_context['categorical_columns']}
        - Missing Values: {self.data_context['missing_values']}
        - Unique Values per Column: {self.data_context['unique_values']}

        CODE REQUIREMENTS:
        - Start with necessary imports (pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns)
        - Add data validation checks
        - Handle missing values appropriately
        - Provide detailed output with explanations
        - For visualizations: use plt.figure(figsize=(12, 8)), add titles, labels, and styling
        - Always end visualizations with plt.tight_layout() and plt.show()
        - Return ONLY executable Python code, no markdown formatting
        
        EXAMPLE PATTERNS:
        
        For counting/grouping:
        ```python
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Validate data exists
        if df.empty:
            print("Dataset is empty")
        else:
            # Your analysis here
            result = df['column'].value_counts()
            print(f"Analysis Results:\\n{{result}}")
            
            # For visualization
            plt.figure(figsize=(12, 8))
            result.plot(kind='bar')
            plt.title('Value Counts for Column')
            plt.xlabel('Values')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        ```
        
        For follow-up questions like "explain this" or "why":
        ```python
        # Analyze the data to understand patterns
        # Look for correlations, distributions, outliers
        # Provide data-driven explanations
        ```
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}"}
        ]
        
        # Add conversation history for context
        self.conversation_history.append({"role": "user", "content": question})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=3000
            )
            
            generated_code = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": generated_code})
            
            return generated_code
            
        except Exception as e:
            error_msg = f"Error: Failed to generate analysis code. {str(e)}"
            self.conversation_history.append({"role": "assistant", "content": error_msg})
            return error_msg
    
    def clear_conversation_history(self):
        """Clear conversation history when new dataset is loaded"""
        self.conversation_history = []
        self.data_context = {}
