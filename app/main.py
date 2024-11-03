import os, requests, uvicorn
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from WeatherInput import WeatherInput
from globals.globals import *
from globals.logger import *
import predict
import train


app = FastAPI()

origins = [
    "*"   #TODO: temporary only
    # "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health(request: Request):
    response={"status":"ok"}
    return response


@app.post("/weather")
def weather(input_data: WeatherInput):
    LOG.debug("WEATHER endpoint activated.")
    try:
        y_pred = predict.predict(input_data)
        response = {"RainTomorrow": int(y_pred)}
        LOG.info("WEATHER endpoint SUCCESS.")
        return response
    
    except Exception as e:
        LOG.error(f"WEATHER: endpoint error.")
        LOG.exception(e)
        return {}


#MAIN:
if __name__ == '__main__':
    if not os.path.exists(os.path.join(SAVE_DIR, MODELNAME)):
        print("TRAINING THE MODEL. This might take a few minutes...")
        LOG.info("TRAINING THE MODEL. This might take a few minutes...")
        train.perform_training()
    
    print("Model already trained. STARTING THE ENDPOINT")
    LOG.info("Model already trained. STARTING THE ENDPOINT")
    uvicorn.run(app, host='0.0.0.0', port=3001)
