import pandas as pd
import numpy as np
import io
import ast
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, Any
import os

class DataProcessor:
    def __init__(self):
        pass
    
    def safe_execute_code(self, code: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Safely execute the generated analysis code"""
        # Create a copy of the dataframe to avoid modifying the original
        local_df = df.copy()
        
        # Prepare the execution environment
        exec_globals = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': None,
            'scipy': None,
            'stats': None,
            'df': local_df
        }
        
        # Add necessary imports that might be referenced
        try:
            import seaborn as sns
            import scipy
            from scipy import stats
            
            exec_globals.update({
                'sns': sns,
                'scipy': scipy,
                'stats': stats
            })
        except ImportError as e:
            return {
                'success': False,
                'error': f"Missing required library: {e}",
                'output': ''
            }
        
        # Capture stdout to get print statements
        output_capture = io.StringIO()
        
        # Store visualization in memory
        visualization = None
        
        try:
            # Parse the code to check for unsafe operations
            tree = ast.parse(code)
            
            # Check for unsafe operations
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        if alias.name in ['os', 'sys', 'subprocess', 'shutil', 'glob']:
                            return {
                                'success': False,
                                'error': f"Unsafe import detected: {alias.name}",
                                'output': output_capture.getvalue()
                            }
                
                if isinstance(node, ast.Call):
                    if (isinstance(node.func, ast.Attribute) and 
                        isinstance(node.func.value, ast.Name) and
                        node.func.value.id in ['os', 'sys', 'subprocess', 'shutil']):
                        return {
                            'success': False,
                            'error': "Unsafe function call detected",
                            'output': output_capture.getvalue()
                        }
            
            # Execute the code safely
            import sys
            old_stdout = sys.stdout
            sys.stdout = output_capture
            
            try:
                # Replace plt.show() with a function that captures the figure
                original_show = plt.show
                
                def show_override():
                    nonlocal visualization
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    visualization = buf
                    plt.close()
                
                plt.show = show_override
                
                # Execute the code
                exec(code, exec_globals)
            finally:
                sys.stdout = old_stdout
                plt.show = original_show
            
            # Get the output from print statements
            output_text = output_capture.getvalue()
            
            # If no visualization was captured by our override, check if there are any figures
            if visualization is None and plt.get_fignums():
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                visualization = buf
                plt.close('all')
            
            return {
                'output': output_text,
                'visualization': visualization,
                'success': True
            }
            
        except Exception as e:
            # Close any open figures
            plt.close('all')
            return {
                'success': False,
                'error': str(e),
                'output': output_capture.getvalue()
            }