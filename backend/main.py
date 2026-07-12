from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

from generation import router as generation_router
from segmentation import router as segmentation_router

load_dotenv(Path(__file__).with_name(".env"))

app = FastAPI()
app.include_router(generation_router)
app.include_router(segmentation_router)


@app.get("/")
def root():
    return {
        "message": "Smart Decorator API Running",
    }
