import subprocess
import sys
import time


def main():
    print("Starting AI Research Assistant services...")
    
    # Start FastAPI Backend
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    )
    
    time.sleep(2)  # Give backend time to spin up
    
    # Start Streamlit Frontend
    frontend_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "frontend/streamlit_app.py"]
    )
    
    try:
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down services...")
        backend_process.terminate()
        frontend_process.terminate()


if __name__ == "__main__":
    main()