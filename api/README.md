# Stock Prediction API

FastAPI backend with JWT authentication for stock price predictions using LSTM with attention mechanism.

## Setup

1. Install dependencies:
```bash
cd api
pip install -r requirements.txt
```

2. Run the server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## Endpoints

### Authentication
- `POST /api/auth/login` - Login and get JWT token
- `GET /api/auth/verify` - Verify token

### Predictions
- `POST /api/predict` - Get stock predictions (requires authentication)

## Demo Credentials
- Username: `demo`
- Password: `demo123`

## API Documentation
Visit `http://localhost:8000/docs` for interactive API documentation.
