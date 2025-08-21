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
    
    def generate_analysis_code(self, question: str, df: pd.DataFrame) -> str:
        """Generate Python code to answer the data question"""
        
        # Prepare data information with error handling
        try:
            data_info = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "sample": df.head(3).to_dict('records'),  # Reduced to 3 rows to save tokens
                "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
            }
        except Exception as e:
            return f"Error: Failed to prepare data information. {str(e)}"
        
        system_prompt = """
You are an expert data analyst. Generate Python code to answer questions about datasets.

CRITICAL RULES:
1. The dataframe is called `df` and is already loaded
2. Always import required libraries at the beginning
3. Use appropriate statistical methods and visualizations
4. Include clear comments explaining the analysis
5. Use print() statements to display results
6. For plots, use plt.show() to display them
7. Handle potential errors (missing values, data types, etc.)
8. Return ONLY executable Python code without markdown formatting
9. Keep code concise but comprehensive
10. Use modern pandas/numpy/matplotlib syntax

AVAILABLE LIBRARIES:
- pandas as pd
- numpy as np 
- matplotlib.pyplot as plt
- seaborn as sns
- scipy.stats
- sklearn (for machine learning tasks)

EXAMPLE CODE STRUCTURE:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Analysis code here
print("Analysis results:")
# Your analysis

# Visualization if needed
plt.figure(figsize=(10, 6))
# Your plot code
plt.title('Your Title')
plt.show()
```

Generate code that directly answers the user's question with appropriate analysis and visualization.
"""
        
        user_prompt = f"""
Question: {question}

Dataset Information:
- Shape: {data_info['shape']}
- Columns: {data_info['columns']}
- Data Types: {data_info['dtypes']}
- Sample Data: {json.dumps(data_info['sample'], indent=1)}

Generate Python code to answer this question about the dataset.
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=2500,
                timeout=30
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower():
                return "Error: OpenAI API rate limit exceeded. Please try again in a moment."
            elif "quota" in error_msg.lower():
                return "Error: OpenAI API quota exceeded. Please check your API usage."
            elif "timeout" in error_msg.lower():
                return "Error: Request timed out. Please try a simpler question or try again."
            else:
                return f"Error: Failed to generate analysis code. {error_msg}"
