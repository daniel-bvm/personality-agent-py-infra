from python:3.12-slim 

copy requirements.txt requirements.txt
run pip install --no-cache-dir -r requirements.txt

workdir /workspace
env APP_ENV=production

copy server.py server.py
copy app app

cmd ["python", "server.py"]