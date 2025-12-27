from typing import Annotated, Union
import requests
TOKEN = "738b541a5f7a"

from fastmcp import FastMCP
mcp = FastMCP(
    name="Tools-MCP-Server",
    instructions="""This server contains some api of tools.""",
)

from transformers import pipeline
classifier = pipeline("text-classification", model="../../../../../models/google-bert/bert-base-chinese")
    
@mcp.tool
def text_classification(text: Annotated[str, "Text to classify"]):
    """Get the text's classification."""
    try:
        result = classifier(text)
        return result
    except Exception as e:
        return {"error": str(e)}
    
@mcp.tool
def get_city_weather(city_name: Annotated[str, "The Pinyin of the city name (e.g., 'beijing' or 'shanghai')"]):
    """Retrieves the current weather data using the city's Pinyin name."""
    try:
        response=requests.get(f"https://whyta.cn/api/tianqi?key={TOKEN}&city={city_name}")
        print(f"🔧 响应状态码: {response.status_code}")
        print(f"🔧 响应内容: {response.text}")
        
        # 检查响应状态
        if response.status_code != 200:
            return f"API 请求失败，状态码: {response.status_code}"
        
        # 解析 JSON
        data = response.json()
        print(f"🔧 解析后的数据: {data}")
        
        # 检查数据结构
        if "data" not in data:
            return f"API 返回数据格式异常: {data}"
        
        weather_data = data["data"]
        return weather_data
        
    except requests.exceptions.RequestException as e:
        return f"网络请求错误: {str(e)}"
    except ValueError as e:
        return f"JSON 解析错误: {str(e)}"
    except Exception as e:
        return f"未知错误: {str(e)}"

@mcp.tool
def get_address_detail(address_text: Annotated[str, "City Name"]):
    """Parses a raw address string to extract detailed components (province, city, district, etc.)."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/addressparse?key={TOKEN}&text={address_text}").json()["result"]
    except:
        return []

@mcp.tool
def get_tel_info(tel_no: Annotated[str, "Tel phone number"]):
    """Retrieves basic information (location, carrier) for a given telephone number."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/mobilelocal?key={TOKEN}&phone={tel_no}").json()["result"]
    except:
        return []

@mcp.tool
def get_scenic_info(scenic_name: Annotated[str, "Scenic/tourist place name"]):
    """Searches for and retrieves information about a specific scenic spot or tourist attraction."""
    # https://apis.whyta.cn/docs/tx-scenic.html
    try:
        return requests.get(f"https://whyta.cn/api/tx/scenic?key={TOKEN}&word={scenic_name}").json()["result"]["list"]
    except:
        return []

@mcp.tool
def get_flower_info(flower_name: Annotated[str, "Flower name"]):
    """Retrieves the flower language (花语) and details for a given flower name."""
    # https://apis.whyta.cn/docs/tx-huayu.html
    try:
        return requests.get(f"https://whyta.cn/api/tx/huayu?key={TOKEN}&word={flower_name}").json()["result"]
    except:
        return []

@mcp.tool
def get_rate_transform(
    source_coin: Annotated[str, "The three-letter code (e.g., USD, CNY) for the source currency."], 
    aim_coin: Annotated[str, "The three-letter code (e.g., EUR, JPY) for the target currency."], 
    money: Annotated[Union[int, float], "The amount of money to convert."]
):
    """Calculates the currency exchange conversion amount between two specified coins."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/fxrate?key={TOKEN}&fromcoin={source_coin}&tocoin={aim_coin}&money={money}").json()["result"]["money"]
    except:
        return []

