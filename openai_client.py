from openai import OpenAI
import os
import pandas as pd
import json

class OpenAIClient:
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        # Try to get API key from parameter, then environment
        final_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not final_api_key:
            raise ValueError("OpenAI API key not provided")
        
        # Initialize client with only the API key
        self.client = OpenAI(api_key=final_api_key)
        self.model = model
    
    def generate_analysis_code(self, question: str, df: pd.DataFrame) -> str:
        """Generate Python code to answer the data question"""
        
        # Prepare data information
        data_info = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample": df.head(5).to_dict('records')
        }
        
        system_prompt = """
        You are a data analysis expert. Your role is to generate Python code to answer questions about datasets.
        
        Rules:
        1. Only generate code that answers the specific question about the provided data
        2. The dataframe is called `df` and is already loaded
        3. Include all necessary imports (pandas, numpy, scipy, matplotlib, seaborn, etc.)
        4. For statistical tests, use appropriate libraries (scipy.stats, etc.)
        5. Include clear comments explaining what each part of the code does
        6. Output the results using print statements
        7. For visualizations, use matplotlib or seaborn and save the plot to 'output_plot.png'
        8. Return only the Python code without any additional text or markdown formatting
        9. Make sure the code is syntactically correct and can be executed directly
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\n\nData Information:\n{json.dumps(data_info, indent=2)}"}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: Failed to generate analysis code. {str(e)}"
