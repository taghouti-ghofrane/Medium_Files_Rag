"""
Weather query tool
"""
import logging
import requests
import os
import re
import json
from typing import Dict, Any, List, Callable, Optional, Tuple

from config.settings import AMAP_API_KEY

# Configure logging
logger = logging.getLogger(__name__)

class WeatherService:
    """
    Weather query service based on Amap API
    """
    
    # Amap API endpoints
    WEATHER_API_URL = "https://restapi.amap.com/v3/weather/weatherInfo"
    GEO_API_URL = "https://restapi.amap.com/v3/geocode/geo"
    
    # 1. Initialize weather query service
    def __init__(self, api_key: str):
        """
        
        Args:
            api_key: Amap API key
        """
        self.api_key = api_key
        # logger.info("Weather query service initialized successfully")
    

    # 2. Get city code
    def get_city_code(self, city_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Args:
            city_name: City name
            
        Returns:
            Tuple[Optional[str], Optional[str]]: (adcode, city name) tuple, returns (None, None) if query fails
        """
        try:
            params = {
                "key": self.api_key,
                "address": city_name,
                "output": "JSON"
            }
            
            response = requests.get(self.GEO_API_URL, params=params)
            data = response.json()
            
            if data["status"] == "1" and data["count"] != "0":
                geocode = data["geocodes"][0]
                return geocode["adcode"], geocode["city"] or geocode["district"]
            else:
                logger.warning(f"City not found: {city_name}, API returned: {data}")
                return None, None
                
        except Exception as e:
            logger.error(f"Error occurred when getting city code: {str(e)}")
            return None, None
    

    # 3. Query weather information
    def query_weather(self, city: str, extensions: str = "all") -> Dict[str, Any]:
        """
        
        Args:
            city: City name or code
            extensions: Weather type, base-current weather, all-forecast weather (next 3 days)
            
        Returns:
            Dict[str, Any]: Weather information dictionary
        """
        result = {
            "status": "error",
            "data": None,
            "message": ""
        }
        
        try:
            # First try to get city code
            city_code, city_name = city, city
            if not city.isdigit():
                city_code, city_name = self.get_city_code(city)
                
            if not city_code:
                result["message"] = f"Cannot find city: {city}"
                return result
                
            # Request weather API
            params = {
                "key": self.api_key,
                "city": city_code,
                "extensions": extensions,
                "output": "JSON"
            }
            
            response = requests.get(self.WEATHER_API_URL, params=params)
            data = response.json()
            
            if data["status"] == "1":
                result["status"] = "success"
                result["data"] = data
                
                # Add a formatted summary
                if extensions == "base":
                    lives = data.get("lives", [])
                    if lives:
                        weather_info = lives[0]
                        result["summary"] = self._format_current_weather(weather_info, city_name)
                else:
                    forecasts = data.get("forecasts", [])
                    if forecasts and forecasts[0].get("casts"):
                        result["summary"] = self._format_forecast_weather(forecasts[0], city_name)
            else:
                result["message"] = f"Weather query failed, API returned: {data}"
                
            return result
            
        except Exception as e:
            logger.error(f"Error occurred when querying weather: {str(e)}")
            result["message"] = f"Error occurred when querying weather: {str(e)}"
            return result
            


    # 4. Format current weather information
    def _format_current_weather(self, weather: Dict[str, Any], city_name: str) -> str:
        return (
            f"{city_name} current weather: {weather.get('weather')}, temperature {weather.get('temperature')}°C, "
            f"humidity {weather.get('humidity')}%, {weather.get('winddirection')} wind {weather.get('windpower')} level. "
            f"Data release time: {weather.get('reporttime')}"
        )
        

    # 5. Format forecast weather information
    def _format_forecast_weather(self, forecast: Dict[str, Any], city_name: str) -> str:
        result = f"{city_name} future weather forecast:\n"
        
        for cast in forecast.get("casts", []):
            date = cast.get("date")
            day_weather = cast.get("dayweather")
            night_weather = cast.get("nightweather")
            day_temp = cast.get("daytemp")
            night_temp = cast.get("nighttemp")
            day_wind = f"{cast.get('daywind')} wind {cast.get('daypower')} level"
            night_wind = f"{cast.get('nightwind')} wind {cast.get('nightpower')} level"
            
            result += (
                f"{date}: Day {day_weather} {day_temp}°C {day_wind}, "
                f"Night {night_weather} {night_temp}°C {night_wind}\n"
            )
            
        return result




class WeatherTools:
    """
    Weather query tool based on Amap API
    """
    
    def __init__(self, api_key: str):
        """
        Initialize weather tool
        
        Args:
            api_key: Amap API key
        """
        self.weather_service = WeatherService(api_key)
        # logger.info("Weather query tool initialized successfully")
    
    def query_weather(self, city: str) -> str:
        """
        Query weather forecast for specified city
        
        Args:
            city: City name to query
            
        Returns:
            str: Weather information
        """
        result = self.weather_service.query_weather(city)
        if result["status"] == "success" and "summary" in result:
            return result["summary"]
        else:
            return f"Failed to get weather information for {city}: {result.get('message', 'Unknown error')}" 