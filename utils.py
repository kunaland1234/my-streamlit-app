# utils.py
import re
from typing import Dict, Any

def is_data_analysis_question(question: str) -> bool:
    """Check if the question is related to data analysis"""
    data_analysis_keywords = [
        'analy', 'data', 'statistic', 'model', 'pattern', 'trend',
        'correlation', 'regression', 'hypothesis', 'test', 'visualization',
        'plot', 'chart', 'graph', 'insight', 'missing', 'null', 'outlier',
        'clean', 'process', 'transform', 'feature', 'variable', 'column',
        'average', 'mean', 'sum', 'count', 'max', 'min', 'median', 'mode',
        'movie', 'film', 'country', 'year', 'director', 'actor', 'genre',
        't-test', 'chi-square', 'anova', 'p-value', 'standard deviation',
        'variance', 'distribution', 'histogram', 'box plot', 'scatter plot'
    ]
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in data_analysis_keywords)

def clean_generated_code(code: str) -> str:
    """Remove markdown code blocks from generated code"""
    # Remove ```python and ``` markers
    if code.startswith("```python"):
        code = code.replace("```python", "").strip()
    if code.endswith("```"):
        code = code.replace("```", "").strip()
    
    # Remove any remaining markdown code blocks
    code = re.sub(r'```.*?```', '', code, flags=re.DOTALL)
    
    return code
