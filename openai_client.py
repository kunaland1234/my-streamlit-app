from openai import OpenAI
import os
import pandas as pd
from typing import List, Dict, Any
import json

class OpenAIClient:
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        # Initialize the client with just the API key
        if api_key:
            self.client = OpenAI(api_key=api_key)
        elif os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            raise ValueError("OpenAI API key not provided")
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
        
        Example for "count of movies vs TV shows":
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Count the occurrences of each type
        type_counts = df['type'].value_counts()
        
        # Print the results
        print("Count of each type:")
        for idx, count in type_counts.items():
            print(f"{idx}: {count}")
            
        # Create a bar chart
        plt.figure(figsize=(10, 6))
        type_counts.plot(kind='bar')
        plt.title('Count of Movies vs TV Shows')
        plt.xlabel('Type')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('output_plot.png')
        plt.show()
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\n\nData Information:\n{json.dumps(data_info, indent=2)}"}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,  # Low temperature for more deterministic code
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: Failed to generate analysis code. {str(e)}"
