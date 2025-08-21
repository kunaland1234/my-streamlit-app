import pandas as pd
import numpy as np
import io
import ast
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, Any
import os
import sys
import warnings

class DataProcessor:
    def __init__(self):
        # Suppress warnings for cleaner output
        warnings.filterwarnings('ignore')
        
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
            'df': local_df,
            'warnings': warnings
        }
        
        # Add necessary imports that might be referenced
        try:
            import seaborn as sns
            import scipy
            from scipy import stats
            import sklearn
            from sklearn import preprocessing, model_selection, metrics
            
            exec_globals.update({
                'sns': sns,
                'scipy': scipy,
                'stats': stats,
                'sklearn': sklearn,
                'preprocessing': preprocessing,
                'model_selection': model_selection,
                'metrics': metrics
            })
        except ImportError as e:
            # Continue without the missing library
            print(f"Warning: Some libraries not available: {e}")
        
        # Capture stdout to get print statements
        output_capture = io.StringIO()
        
        # Store visualization in memory
        visualization = None
        
        try:
            # Parse the code to check for unsafe operations
            tree = ast.parse(code)
            
            # Check for unsafe operations
            unsafe_modules = ['os', 'sys', 'subprocess', 'shutil', 'glob', 'pickle', 'exec', 'eval']
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        if alias.name in unsafe_modules:
                            return {
                                'success': False,
                                'error': f"Unsafe import detected: {alias.name}",
                                'output': ''
                            }
                
                if isinstance(node, ast.Call):
                    if (isinstance(node.func, ast.Attribute) and 
                        isinstance(node.func.value, ast.Name) and
                        node.func.value.id in unsafe_modules):
                        return {
                            'success': False,
                            'error': "Unsafe function call detected",
                            'output': ''
                        }
            
            # Execute the code safely
            old_stdout = sys.stdout
            sys.stdout = output_capture
            
            try:
                # Set matplotlib backend for non-interactive environment
                plt.switch_backend('Agg')
                
                # Replace plt.show() with a function that captures the figure
                original_show = plt.show
                
                def show_override():
                    nonlocal visualization
                    if plt.get_fignums():  # Check if there are any figures
                        try:
                            buf = io.BytesIO()
                            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                            buf.seek(0)
                            visualization = buf
                        except Exception as e:
                            print(f"Warning: Could not save plot: {e}")
                        finally:
                            plt.close('all')
                
                plt.show = show_override
                
                # Execute the code with timeout protection
                exec(code, exec_globals)
                
            finally:
                sys.stdout = old_stdout
                plt.show = original_show
            
            # Get the output from print statements
            output_text = output_capture.getvalue()
            
            # If no visualization was captured by our override, check if there are any figures
            if visualization is None and plt.get_fignums():
                try:
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                    buf.seek(0)
                    visualization = buf
                except Exception as e:
                    print(f"Warning: Could not save remaining plots: {e}")
                finally:
                    plt.close('all')
            
            # Clean up any remaining figures
            plt.close('all')
            
            return {
                'output': output_text,
                'visualization': visualization,
                'success': True
            }
            
        except SyntaxError as e:
            plt.close('all')
            return {
                'success': False,
                'error': f"Syntax Error: {str(e)}",
                'output': output_capture.getvalue()
            }
        except NameError as e:
            plt.close('all')
            return {
                'success': False,
                'error': f"Name Error: {str(e)}. Check if all required libraries are imported.",
                'output': output_capture.getvalue()
            }
        except KeyError as e:
            plt.close('all')
            return {
                'success': False,
                'error': f"Key Error: {str(e)}. Check if the column name exists in the dataset.",
                'output': output_capture.getvalue()
            }
        except ValueError as e:
            plt.close('all')
            return {
                'success': False,
                'error': f"Value Error: {str(e)}",
                'output': output_capture.getvalue()
            }
        except Exception as e:
            plt.close('all')
            return {
                'success': False,
                'error': f"Execution Error: {str(e)}",
                'output': output_capture.getvalue()
            }
