import os
from pathlib import Path
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from typing import Optional
import logging
from fastapi.logger import logger as fastapi_logger
from data.Models.models import load_and_preprocess_data, EmissionsPredictor
from .llm_story import Story

app = FastAPI()

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Changed to http for local development
        "https://nasaspaceappschallenge2024-2.onrender.com",
        "https://syang0624.github.io",
        "http://101.101.218.177",  # Changed to http if not served over HTTPS
        "https://*.onrender.com",  # Wildcard for Render subdomains
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
fastapi_logger.handlers = logging.getLogger("uvicorn").handlers

# Use environment variables for configuration
PORT = int(os.getenv('PORT', 8000))
HOST = os.getenv('HOST', '0.0.0.0')

# Use relative path for data files
base_path = Path(__file__).parent.parent / "data" / "Models"
file_url = os.getenv('DATA_FILE_PATH', str(base_path))

# Initialize EmissionsPredictor and load data in startup event
@app.on_event("startup")
async def startup_event():
    global predictor, combined_df, metrics, initial_ghg, initial_story
    predictor = EmissionsPredictor()
    transport_df = load_and_preprocess_data(Path(file_url) / 'TransportX2.csv')
    electricity_df = load_and_preprocess_data(Path(file_url) / 'ElectricityX3.csv')
    agriculture_df = load_and_preprocess_data(Path(file_url) / 'AgricultureX1.csv')
    combined_df = predictor.preprocess_data(transport_df, electricity_df, agriculture_df)
    metrics = predictor.train(combined_df)
    logger.info("Model trained successfully")
    
    # Initialize global variables
    initial_year = 2000
    initial_ghg = predictor.predict_emissions('CA', initial_year, 0, 0, 0)
    initial_story = story_generator.get_result(
        year=initial_year,
        ghg_level=initial_ghg,
        certificate_level=None
    )

# New GHG router
ghg_router = APIRouter()

class InputData(BaseModel):
    x1: float  # trees
    x2: float  # miles
    x3: float  # electricity
    year: int
    state: Optional[str] = 'CA'

class OutputData(BaseModel):
    GHG: float
    story: str
    year: int
    certificate_level: Optional[str] = None
    state_max_emissions: Optional[float] = None
    model_accuracy: Optional[dict] = None

story_generator = Story()

# Global variable for data storage
current_data = None

@ghg_router.post("/input")
async def input_data(data: InputData):
    global current_data
    try:
        ghg = predictor.predict_emissions(
            state=data.state,
            year=data.year,
            trees=data.x1,
            miles=data.x2,
            electricity=data.x3
        )
        state_max = predictor.get_state_max(data.state)
        model_accuracy = predictor.get_model_accuracy()

        current_data = OutputData(
            GHG=ghg,
            story="",
            year=data.year,
            certificate_level=None,
            state_max_emissions=state_max,
            model_accuracy=model_accuracy
        )

        return {"message": "Data processed successfully"}
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@ghg_router.get("/output")
async def get_output():
    global current_data
    if not current_data:
        return {"error": "No data available"}

    current_data.certificate_level = None
    if current_data.year >= 2020:
        if current_data.GHG < 3000:
            current_data.certificate_level = "Gold"
        elif current_data.GHG < 4000:
            current_data.certificate_level = "Silver"
        else:
            current_data.certificate_level = "Bronze"

    story = story_generator.get_result(
        year=current_data.year,
        ghg_level=current_data.GHG,
        certificate_level=current_data.certificate_level
    )
    current_data.story = story

    logger.info(f"Year: {current_data.year}, GHG: {current_data.GHG}, Certificate Level: {current_data.certificate_level}")

    return current_data

@ghg_router.get("/initial")
async def get_initial_data():
    try:
        initial_ghg_float = float(initial_ghg)
        state_max_emissions = float(predictor.get_state_max('CA'))
        model_accuracy = {key: float(value) for key, value in predictor.get_model_accuracy().items()}

        return {
            "year": initial_year,
            "GHG": initial_ghg_float,
            "story": initial_story,
            "certificate_level": None,
            "state_max_emissions": state_max_emissions,
            "model_accuracy": model_accuracy
        }
    except Exception as e:
        logger.error(f"Error in /ghg/initial endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# Include the new router in the app
app.include_router(ghg_router, prefix="/ghg")

@app.get("/")
async def root():
    return {"message": "Welcome to the API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)