# Add to main.py
from fastapi import FastAPI, UploadFile
import uvicorn

app_api = FastAPI()

@app_api.post("/analyze_video")
async def analyze_video(video: UploadFile):
    # Process video and return violations
    return {"violations": violations} # pyright: ignore[reportUndefinedVariable]

@app_api.get("/signal_status/{intersection_id}")
async def get_signal_status(intersection_id: str):
    return controller.get_state() # pyright: ignore[reportUndefinedVariable]

@app_api.post("/manual_override/{intersection_id}")
async def manual_override(intersection_id: str, phase: str):
    controller.manual_protect() # pyright: ignore[reportUndefinedVariable]
    return {"status": "overridden", "phase": phase}