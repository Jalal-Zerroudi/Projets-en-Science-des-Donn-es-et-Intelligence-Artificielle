import subprocess
import time

# Lancer le backend FastAPI via uvicorn
backend = subprocess.Popen([
    "uvicorn",
    "backend.B_App:app", 
    "--host", "127.0.0.1",
    "--port", "8000",
    "--reload"              
])

time.sleep(5)  # donne un peu de temps au backend pour démarrer

# Lancer le frontend Streamlit
frontend = subprocess.Popen([
    "python",
    "frontend/F_App.py"
])
