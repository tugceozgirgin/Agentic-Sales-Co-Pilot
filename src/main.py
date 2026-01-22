import sys
import subprocess
from pathlib import Path
from src.database.semantic_db import SemanticDatabase
from src.database.structured_db import StructuredDatabase


def run():
    """
    Main entry point: initialize databases and run the Streamlit app.
    """
    SemanticDatabase.initialize_and_populate()
    

    StructuredDatabase.initialize_and_populate()
    
    print("[INFO] Starting Streamlit application...")
    app_path = Path(__file__).parent / "app.py"
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to start Streamlit application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()
