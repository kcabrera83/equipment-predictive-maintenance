# Deployment Guide - Equipment Predictive Maintenance

## Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python scripts/train.py

EXPOSE 5003

CMD ["python", "app.py"]
```

### Build and Run
```bash
docker build -t equipment-predictive-maintenance .
docker run -p 5003:5003 equipment-predictive-maintenance
```

### Docker Compose
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "5003:5003"
    environment:
      - FLASK_DEBUG=0
    volumes:
      - ./outputs:/app/outputs
    restart: unless-stopped
```

```bash
docker-compose up -d
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| FLASK_DEBUG | Enable debug mode | 1 |
| PORT | Server port | 5003 |
| HOST | Server host | 0.0.0.0 |

## Manual Deployment

### Prerequisites
- Python 3.8+
- pip

### Steps
```bash
git clone https://github.com/kcabrera83/equipment-predictive-maintenance.git
cd equipment-predictive-maintenance
pip install -r requirements.txt
python scripts/train.py
python scripts/evaluate.py  # optional
python app.py
```

## Production Considerations

### Gunicorn (Recommended)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5003 app:app
```

### Security
- Set `DEBUG=False` in production
- Use HTTPS with a reverse proxy
- Add authentication for prediction endpoints
- Validate sensor input ranges

### Monitoring
- Monitor `/api/health` for uptime
- Track failure prediction distribution
- Alert on high anomaly detection rates
- Monitor RUL predictions for equipment fleet

### Performance
- Models lazy-loaded on first request
- Pre-load models at startup for faster responses
- Consider batch prediction for fleet monitoring

## CI/CD
GitHub Actions workflow runs on every push:
1. Model training
2. Predictions
3. Full evaluation
4. API tests (5 endpoints)

## API Self-Documentation
Access OpenAPI docs at: `http://localhost:5003/api/docs`
