import os
import sys
import inspect
from pathlib import Path
from pandas import read_csv

def get_directory() -> str:
    """
    Returns the directory of the file this function is called from.
    (Should) work in both regular Python scripts and Jupyter notebooks.
    """
    # For Jupyter notebooks:
    try:
        import ipynbname
        notebook_path = ipynbname.path()
        return notebook_path.parent
    except Exception:
        pass

    # Standard script: use __file__ of the caller's frame
    import inspect
    frame = inspect.stack()[1]
    caller_file = frame.filename

    if caller_file and caller_file != "<stdin>" and not caller_file.startswith("<ipython"):
        return os.path.abspath(os.path.dirname(caller_file))

    # Final fallback: current working directory
    return os.path.abspath(os.getcwd())

def find_path(target_path, start_dir="."):
    """
    Search for a file or folder in:
    1. Current directory and subdirectories (downwards)
    2. Parent directories (upwards)
    3. Adjacent (sibling) directories

    target_path: The relative path of the folder/file (e.g., 'myfolder/mydata.csv').
    start_dir: The directory to start the search from (default is current directory).

    returns: The absolute path if found, else None.
    """
    start_dir = os.path.abspath(start_dir)  # Convert to absolute path

    # Step 1: Search Downwards (Current Directory & Subdirectories)
    for root, _, files in os.walk(start_dir):
        candidate_path = os.path.join(root, target_path)
        if os.path.exists(candidate_path):
            return os.path.abspath(candidate_path)

    # Step 2: Search Upwards (Parent Directories)
    current_dir = start_dir
    while current_dir != os.path.dirname(current_dir):  # Stop at root directory
        current_dir = os.path.dirname(current_dir)  # Move one level up
        candidate_path = os.path.join(current_dir, target_path)
        if os.path.exists(candidate_path):
            return os.path.abspath(candidate_path)

    # Step 3: Search in Adjacent (Sibling) Directories
    parent_dir = os.path.dirname(start_dir)  # Get parent directory
    for sibling in os.listdir(parent_dir):
        sibling_path = os.path.join(parent_dir, sibling, target_path)
        if os.path.isdir(os.path.join(parent_dir, sibling)) and os.path.exists(sibling_path):
            return os.path.abspath(sibling_path)

    print(f"Error: '{target_path}' not found in any directories.")
    return None

def file_load(file):
    # Define working directory & read path
    raw_data_path = Path('.')  # Tuple of all subdirectories below CWD
    csv_location = str(list(raw_data_path.glob("**/" + file))[0])
    return read_csv(csv_location)

def create_folder(folder):
    is_exist = os.path.exists(folder)
    if not is_exist:
        # Create a new directory because it does not exist
        os.makedirs(folder)

###########################################################################
# - LOADING DATA -------------------------------------------------------- #
###########################################################################

