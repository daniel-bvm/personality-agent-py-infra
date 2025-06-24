# personality-agent-py-infra
python version of personality agent infra

## DEBUGGING

```bash
docker network create agent 
docker build -t devtool devtools && docker run --rm -it --network agent --name devtool -e PORT=8010 devtool
```

```bash
docker build -t test . && docker run --rm -it -p 8000:80 --network agent -e BACKEND_BASE_URL=http://devtool:8010 -e AUTHORIZATION_TOKEN="idk" --env-file=.env test
```