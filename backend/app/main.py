from dotenv import load_dotenv
from fastapi import FastAPI

from app.core.config import ENV_FILE
from generation import router as generation_router
from segmentation import router as segmentation_router

load_dotenv(ENV_FILE)

app = FastAPI()
app.include_router(generation_router)
app.include_router(segmentation_router)


@app.get("/")
def root():
    return {
        "message": "Smart Decorator API Running",
    }
