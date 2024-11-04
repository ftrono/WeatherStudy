import os, uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from classes.WeatherInput import WeatherInput
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
        response = predict.make_prediction(input_data.model_dump())
        return response
    
    except Exception as e:
        LOG.error(f"WEATHER: endpoint error.")
        LOG.exception(e)
        return {"error": "endpoint_error"}


#MAIN:
if __name__ == '__main__':
    if not os.path.exists(os.path.join(SAVE_DIR, MODELNAME)):
        # Train first:
        print("TRAINING THE MODEL. This might take a few minutes...")
        LOG.info("TRAINING THE MODEL. This might take a few minutes...")
        ret = train.perform_training()
        if ret == 0:
            LOG.info("STARTING THE ENDPOINT")
            uvicorn.run(app, host='0.0.0.0', port=3001)
    else:
        # Directly load endpoint:
        print("Model already trained. STARTING THE ENDPOINT")
        LOG.info("Model already trained. STARTING THE ENDPOINT")
        uvicorn.run(app, host='0.0.0.0', port=3001)
