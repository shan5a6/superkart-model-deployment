# SuperKart Backend API

This folder contains the Flask backend API for the SuperKart sales forecasting model.

## Files

- `app.py` - Flask API application
- `requirements.txt` - Python dependencies
- `Dockerfile` - Backend container definition
- `superkart_best_model.joblib` - Serialized machine learning model

## Endpoints

- `/` - API information
- `/health` - Health check
- `/predict` - Single prediction
- `/batch_predict` - Batch prediction

## Container Port

The backend API runs on port `5000`.

## Docker Compose

The backend is started through the root-level `docker-compose.yml` file.
