import subprocess
import time
import sys
import uvicorn


def main():
    """Launches FastAPI backend server and Streamlit UI process concurrently."""
    print("🚀 Starting Multi-Agent AI System...")
    
    # Launch Streamlit process
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run", "app/ui/app.py",
        "--server.port=8501", "--server.address=0.0.0.0"
    ]
    streamlit_process = subprocess.Popen(streamlit_cmd)
    
    print("✨ Streamlit UI running on http://localhost:8501")
    print("⚡ Starting FastAPI Backend on http://localhost:8000")
    
    try:
        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        streamlit_process.terminate()


if __name__ == "__main__":
    main()