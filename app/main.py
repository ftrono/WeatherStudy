import os, requests, uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.logger import *


class Item(BaseModel):
    item_name: str

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


@app.post("/weather")
def weather(local_info: Item):
    LOG.debug("WEATHER endpoint activated.")
    try:
        data = local_info.model_dump()
        #TODO
        
        response = {}
        LOG.info("WEATHER endpoint SUCCESS.")
        return response
    
    except Exception as e:
        LOG.error(f"WEATHER: endpoint error.")
        LOG.exception(e)
        return {}


#MAIN:
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=3001)