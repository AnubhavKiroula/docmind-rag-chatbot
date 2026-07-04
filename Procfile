# Procfile for Render Web Service Deployment
# WHY: Tells Render PaaS how to execute the FastAPI backend server dynamically binding to Render's assigned $PORT environment variable.

web: uvicorn main:app --host 0.0.0.0 --port $PORT
