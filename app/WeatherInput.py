from pydantic import BaseModel

class WeatherInput(BaseModel):
    Date: str = None
    Location: str = None
    MinTemp: float
    MaxTemp: float
    Rainfall: float
    Evaporation: float = None
    Sunshine: float = None
    WindGustDir: str
    WindGustSpeed: float
    WindDir9am: str
    WindDir3pm: str
    WindSpeed9am: float
    WindSpeed3pm: float
    Humidity9am: float
    Humidity3pm: float
    Pressure9am: float
    Pressure3pm: float
    Cloud9am: float = None
    Cloud3pm: float = None
    Temp9am: float
    Temp3pm: float
    RainToday: str = None