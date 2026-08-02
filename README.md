# SuperKart Sales Forecasting - Model Deployment Project

This repository contains an end-to-end machine learning deployment project for SuperKart sales forecasting.

The project includes:

- Data analysis and preprocessing
- Machine learning model training
- Model serialization
- Flask backend API
- Streamlit frontend app
- Dockerfiles
- Docker Compose
- GitHub Codespaces deployment

## Project Structure

```text
superkart-model-deployment/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── superkart_best_model.joblib
│
├── frontend/
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── .devcontainer/
│   └── devcontainer.json
│
├── docker-compose.yml
├── Batch_Data_SuperKart.csv
└── README.md
```

## Deployment using GitHub Codespaces

### Step 1: Open the repository in GitHub Codespaces

Open the GitHub repository and create a new Codespace.

### Step 2: Build and run the containers

From the repository root, run:

```bash
docker compose up --build
```

This command builds and starts:

- Backend Flask API on port `5000`
- Frontend Streamlit app on port `8501`

### Step 3: Open forwarded ports

In the Codespaces **Ports** tab, check:

- Port `5000` for backend
- Port `8501` for frontend

Set both ports to **Public** if the evaluator needs to access the URLs.

You can also run:

```bash
gh codespace ports visibility 5000:public 8501:public
```

### Step 4: Test backend health

```bash
curl http://127.0.0.1:5000/health
```

Expected response:

```json
{
  "model_loaded": true,
  "status": "healthy"
}
```

### Step 5: Open frontend

Open the forwarded URL for port `8501`.

The Streamlit app supports:

- Single prediction
- Batch prediction

For Docker Compose deployment, the frontend backend URL should be:

```text
http://backend:5000
```

## Stopping the containers

To stop the running containers, press `CTRL + C`.

Then run:

```bash
docker compose down
```
