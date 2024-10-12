from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from typing import Optional
import logging
from data.Models.models import load_and_preprocess_data, EmissionsPredictor
from .llm_story import Story




app = FastAPI()


# CORS middleware setup
#allow_origins=["https://your-frontend-url.onrender.com"],
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Local development (if needed)
        "https://nasaspaceappschallenge2024-2.onrender.com/NASASpaceAppsChallenge2024/",
        "https://syang0624.github.io",  # GitHub Pages deployment (if needed)
        "http://101.101.218.177"  # External IP address, 
        "https://syang0624.github.io/NASASpaceAppsChallenge2024"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# GZip compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize EmissionsPredictor
predictor = EmissionsPredictor()
#file_url = "../../data/Models/"
file_url = "/Users/godsonajodo/Desktop/Hackathon/NASASpaceAppsChallenge2024/data/Models/"



transport_df = load_and_preprocess_data(file_url + 'TransportX2.csv')
electricity_df = load_and_preprocess_data(file_url + 'ElectricityX3.csv')
agriculture_df = load_and_preprocess_data(file_url + 'AgricultureX1.csv')


# Preprocess data and train the model
combined_df = predictor.preprocess_data(transport_df, electricity_df, agriculture_df)
metrics = predictor.train(combined_df)

logger.info("Model trained successfully")

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

# Global variables for data storage
initial_year = 2000
initial_ghg = predictor.predict_emissions('CA', initial_year, 0, 0, 0)
initial_story = story_generator.get_result(
    year=initial_year,
    ghg_level=initial_ghg,
    certificate_level=None
)
current_data = OutputData(GHG=initial_ghg, story=initial_story, year=initial_year, certificate_level=None)

@ghg_router.post("/input")
async def input_data(data: InputData):
    global current_data
    try:
        # Make prediction using the trained model
        ghg = predictor.predict_emissions(
            state=data.state,
            year=data.year,
            trees=data.x1,
            miles=data.x2,
            electricity=data.x3
        )

        # Get the maximum emissions for the state
        state_max = predictor.get_state_max(data.state)

        # Get model accuracy metrics
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
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@ghg_router.get("/output")
async def get_output():
    global current_data
    if not current_data:
        return {"error": "No data available"}

    # Reset certificate_level
    current_data.certificate_level = None

    if current_data.year >= 2020:
        # Certification level logic (example)
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

    # Logging
    logger.info(f"Year: {current_data.year}, GHG: {current_data.GHG}, Certificate Level: {current_data.certificate_level}")

    return current_data
@ghg_router.get("/initial")
async def get_initial_data():
    try:
        # Convert numpy types to Python native types
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
        logger.error(f"Error in /ghg/initial endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")



# Include the new router in the app
app.include_router(ghg_router, prefix="/ghg")

@app.get("/")
async def root():
    return {"message": "Welcome to the API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)