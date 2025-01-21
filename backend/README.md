# Vehicle Detection Backend

This is the backend service for the Vehicle Detection system, built with FastAPI. It provides REST APIs for vehicle detection, processing video streams, and managing detection results.

## Running with Docker (Recommended)

The easiest way to run the backend is using Docker Compose from the root directory:

```bash
docker compose up --build
```

This will start:
- Backend API server at `http://localhost:8000`
- PostgreSQL database at `localhost:5432` -> not being used right now
- PgAdmin interface at `http://localhost:5050` -> not being used right now
- LocalStack (S3 emulator) at `http://localhost:4566`


## API Documentation

Once running, API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Development with Docker

### Useful Commands

```bash
# Build and start all services
docker compose up --build

# Start specific service
docker compose up backend

# View logs
docker compose logs -f backend

# Stop all services
docker compose down

# Reset everything (including volumes)
docker compose down -v
```