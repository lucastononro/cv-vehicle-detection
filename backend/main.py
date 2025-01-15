from fastapi import FastAPI
from app import create_app
from app.api.v1 import v1_router

# Create FastAPI app instance
app = create_app()

# Include API routers
app.include_router(v1_router, prefix="/api")

# Optional: Add startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """
    Initialize services on startup
    """
    pass

@app.on_event("shutdown")
async def shutdown_event():
    """
    Clean up resources on shutdown
    """
    pass

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload
        workers=1     # Use single worker for development
    ) 