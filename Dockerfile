from python:3.12-alpine 

copy requirements.txt requirements.txt
run pip install --no-cache-dir -r requirements.txt

copy server.py server.py
copy app app

workdir /workspace
env APP_ENV=production

cmd ["python", "server.py"]