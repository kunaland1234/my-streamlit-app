import pandas as pd
import numpy as np
import io
import ast
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import os
import traceback
import sys

class DataProcessor:
    def __init__(self):
        # Set up matplotlib and seaborn defaults
        plt.style.use('default')
        sns.set_palette("husl")
        
    def safe_execute_code(self, code: str, df: pd.DataFrame, max_retries: int = 2) -> Dict[str, Any]:
        """Safely execute the generated analysis code with automatic retry and error handling"""
        
        for attempt in range(max_retries + 1):
            try:
                result = self._execute_single_attempt(code, df)
                if result['success']:
                    return result
                elif attempt < max_retries:
                    # Try to fix common issues automatically
                    code = self._fix_common_code_issues(code, result.get('error', ''))
                    continue
                else:
                    return result
            except Exception as e:
                if attempt < max_retries:
                    code = self._fix_common_code_issues(code, str(e))
                    continue
                else:
                    return {
                        'success': False,
                        'error': f"Failed after {max_retries + 1} attempts: {str(e)}",
                        'output': ''
                    }
    
    def _execute_single_attempt(self, code: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute code in a single attempt"""
        # Create a copy of the dataframe to avoid modifying the original
        local_df = df.copy()
        
        # Prepare the execution environment with comprehensive libraries
        exec_globals = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'df': local_df,
            '__builtins__': __builtins__
        }
        
        # Add additional libraries safely
        try:
            import scipy
            from scipy import stats
            import warnings
            
            exec_globals.update({
                'scipy': scipy,
                'stats': stats,
                'warnings': warnings
            })
            
            # Suppress warnings for cleaner output
            warnings.filterwarnings('ignore')
            
        except ImportError as e:
            return {
                'success': False,
                'error': f"Missing required library: {e}",
                'output': ''
            }
        
        # Capture stdout and stderr
        output_capture = io.StringIO()
        error_capture = io.StringIO()
        
        # Store visualization in memory
        visualization = None
        
        try:
            # Enhanced security check
            if not self._is_code_safe(code):
                return {
                    'success': False,
                    'error': "Code contains potentially unsafe operations",
                    'output': ''
                }
            
            # Execute the code safely with output capture
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            
            try:
                sys.stdout = output_capture
                sys.stderr = error_capture
                
                # Override plt.show() to capture visualizations
                original_show = plt.show
                
                def show_override():
                    nonlocal visualization
                    if plt.get_fignums():  # Check if there are any figures
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300, facecolor='white')
                        buf.seek(0)
                        visualization = buf
                        plt.close('all')  # Close all figures to free memory
                
                plt.show = show_override
                
                # Execute the code
                exec(code, exec_globals)
                
                # Check for any remaining figures
                if visualization is None and plt.get_fignums():
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300, facecolor='white')
                    buf.seek(0)
                    visualization = buf
                    plt.close('all')
                
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                plt.show = original_show
            
            # Get the output and any errors
            output_text = output_capture.getvalue()
            error_text = error_capture.getvalue()
            
            # Combine output and errors if any
            full_output = output_text
            if error_text:
                full_output += f"\nWarnings/Errors:\n{error_text}"
            
            return {
                'output': full_output,
                'visualization': visualization,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            # Close any open figures
            plt.close('all')
            
            # Get detailed error information
            error_details = traceback.format_exc()
            output_text = output_capture.getvalue()
            
            return {
                'success': False,
                'error': f"{str(e)}\n\nDetailed traceback:\n{error_details}",
                'output': output_text
            }
    
    def _is_code_safe(self, code: str) -> bool:
        """Enhanced security check for code safety"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        
        dangerous_patterns = [
            # File system operations
            'open(', 'file(', 'input(', 'raw_input(',
            # System operations
            'os.', 'sys.exit', 'sys.quit', 'quit(', 'exit(',
            # Network operations
            'urllib', 'requests', 'socket', 'http',
            # Subprocess operations
            'subprocess', 'popen', 'system(',
            # Dangerous builtins
            'exec(', 'eval(', 'compile(', '__import__',
            # File operations
            'shutil', 'glob', 'pathlib'
        ]
        
        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                return False
        
        # AST-based checks for more sophisticated attacks
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ['os', 'sys', 'subprocess', 'shutil', 'glob', 'pathlib', 'urllib', 'requests']:
                        return False
            
            if isinstance(node, ast.ImportFrom):
                if node.module in ['os', 'sys', 'subprocess', 'shutil', 'glob', 'pathlib', 'urllib', 'requests']:
                    return False
        
        return True
    
    def _fix_common_code_issues(self, code: str, error_msg: str) -> str:
        """Attempt to fix common code issues automatically"""
        fixed_code = code
        
        # Fix missing imports
        if "seaborn" in error_msg or "sns" in error_msg or "cannot access local variable 'sns'" in error_msg:
            if "import seaborn as sns" not in fixed_code:
                fixed_code = "import seaborn as sns\n" + fixed_code
        
        if "scipy" in error_msg or "stats" in error_msg:
            if "from scipy import stats" not in fixed_code:
                fixed_code = "from scipy import stats\n" + fixed_code
        
        if "numpy" in error_msg or "np." in error_msg:
            if "import numpy as np" not in fixed_code:
                fixed_code = "import numpy as np\n" + fixed_code
        
        if "pandas" in error_msg or "pd." in error_msg:
            if "import pandas as pd" not in fixed_code:
                fixed_code = "import pandas as pd\n" + fixed_code
        
        if "matplotlib" in error_msg or "plt." in error_msg:
            if "import matplotlib.pyplot as plt" not in fixed_code:
                fixed_code = "import matplotlib.pyplot as plt\n" + fixed_code
        
        # Fix common syntax issues
        if "invalid syntax" in error_msg.lower():
            # Try to fix common indentation issues
            lines = fixed_code.split('\n')
            fixed_lines = []
            for line in lines:
                if line.strip():  # Non-empty line
                    fixed_lines.append(line)
            fixed_code = '\n'.join(fixed_lines)
        
        # Add error handling for common operations
        if "KeyError" in error_msg and "column" not in fixed_code.lower():
            fixed_code = """
# Add column existence check
missing_cols = [col for col in df.columns if col not in df.columns]
if missing_cols:
    print(f"Warning: These columns don't exist: {missing_cols}")
    print(f"Available columns: {list(df.columns)}")

""" + fixed_code
        
        return fixed_code
    
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataframe and return useful information"""
        if df is None:
            return {'valid': False, 'error': 'DataFrame is None'}
        
        if df.empty:
            return {'valid': False, 'error': 'DataFrame is empty'}
        
        return {
            'valid': True,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'has_nulls': df.isnull().any().any()
        }
