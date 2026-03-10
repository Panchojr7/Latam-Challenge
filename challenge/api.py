import os
import fastapi
import logging
import pandas as pd
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Union
from challenge.model import DelayModel
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

app = fastapi.FastAPI()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:   %(asctime)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

class AirlineCarrier(Enum):
    """Enumeration of airline carriers supported by the delay prediction model."""

    AEROLINEAS_ARGENTINAS = "Aerolineas Argentinas"
    AEROMEXICO = "Aeromexico"
    AIR_CANADA = "Air Canada"
    AIR_FRANCE = "Air France"
    ALITALIA = "Alitalia"
    AMERICAN_AIRLINES = "American Airlines"
    AUSTRAL = "Austral"
    AVIANCA = "Avianca"
    BRITISH_AIRWAYS = "British Airways"
    COPA_AIR = "Copa Air"
    DELTA_AIR = "Delta Air"
    GOL_TRANS = "Gol Trans"
    GRUPO_LATAM = "Grupo LATAM"
    IBERIA = "Iberia"
    JETSMART_SPA = "JetSmart SPA"
    K_L_M = "K.L.M."
    LACSA = "Lacsa"
    LATIN_AMERICAN_WINGS = "Latin American Wings"
    OCEANAIR_LINHAS_AEREAS = "Oceanair Linhas Aereas"
    PLUS_ULTRA_LINEAS_AEREAS = "Plus Ultra Lineas Aereas"
    QANTAS_AIRWAYS = "Qantas Airways"
    SKY_AIRLINE = "Sky Airline"
    UNITED_AIRLINES = "United Airlines"

class FlightCategory(Enum):
    """Enumeration of flight categories (International vs. National)."""

    INTERNATIONAL = "I"
    NATIONAL = "N"

class CalendarMonth(Enum):
    """Enumeration of calendar months represented as Month integers."""

    JAN = 1
    FEB = 2
    MAR = 3
    APR = 4
    MAY = 5
    JUN = 6
    JUL = 7
    AUG = 8
    SEP = 9
    OCT = 10
    NOV = 11
    DEC = 12

class FlightSchema(BaseModel):
    """Pydantic model defining the data contract for an individual flight record."""

    OPERA: AirlineCarrier
    TIPOVUELO: FlightCategory
    MES: CalendarMonth

class FlightRequestBatch(BaseModel):
    """Pydantic model defining the expected API request structure for a batch of flights."""

    flights: List[FlightSchema]

def validate_payload_structure(payload: Dict[str, Any]) -> bool:
    """Validates if the incoming request payload matches the expected FlightRequestBatch schema.
    
    Args:
        payload (Dict[str, Any]): The raw JSON data received from the API request.

    Returns:
        bool: True if the payload matches the schema and contains allowed values; 
            False if validation fails.
    """
    try:
        FlightRequestBatch.parse_obj(payload)
        flag = True
    except ValidationError as error:
        logger.error(f"Validation Error: {error}")    
        flag = False
    return flag

async def initialize_and_train_model() -> None:
    """Async method to initialize and train the DelayModel during the API 
    startup sequence, ensuring the predictor is ready for serving.
    """

    dataset_path = Path(os.getcwd(), "data/data.csv")
    raw_data = pd.read_csv(filepath_or_buffer=dataset_path)
    predictor = DelayModel()
    train_features, train_target = predictor.preprocess(raw_data, "delay")
    predictor.fit(train_features, train_target)


@app.on_event("startup")
async def on_startup() -> None:
    """Initialization hook to trigger data loading and model training when the API starts."""
    await initialize_and_train_model()
    logger.info("API started")


@app.get("/health", status_code=200)
async def get_health() -> Dict[str, str]:
    """ API Health Monitoring Endpoint.

    Returns:
        dict: A confirmation message indicating the service is operational.
    """
    return {"status": "OK"}

@app.post("/predict", status_code=200)
async def post_predict(payload: Dict[str, Any]) -> Dict[str, Union[str, List[int]]]:
    """Prediction Endpoint to determine the likelihood of flight delays.

    Args:
        payload (Dict[str, Any]): A dictionary containing a list of flight objects 
                                structured according to the batch schema.

    Returns:
        Dict[str, Union[str, List[int]]]: A dictionary with a list of integers 
                        containing the delay prediction for each flight requested, 
                        or a string with an error message if the request payload is invalid.
    """

    if not validate_payload_structure(payload):
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid data structure or unknown column value received"}
        )
    model_engine = DelayModel()
    input_df = pd.DataFrame(payload["flights"])
    processed_x = model_engine.preprocess(input_df)
    results = model_engine.predict(processed_x)
    logger.info(f"Predictions generated: {results}")
    return {"predict": results}
