from dotenv import load_dotenv
from openai import OpenAI
import os
import requests
import json


load_dotenv()
client = OpenAI()


def run_command(cmd:str):
    resulst = os.system(cmd)
    return resulst


def get_weather(city:str):
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if(response.status_code== 200):
         return f"The weather in {city} is {response.text}."
    
    return "Something went wrong"


available_tools = {
    "get_weather": get_weather,
    "run_command": run_command
}

SYSTEM_PROMPT = """
    HELLO
"""


messages = [
    {"role":"system", "content":SYSTEM_PROMPT}
]


while True:
    query = input("> ")
    messages.append({"role":"user", "content":query})

    while True:
        response = client.chat.completions.create(
            model="gpt-4.1",
            response_format={"type":"json_object"},
            messages=messages
        )
          
        messages.append({ "role": "assistant", "content": response.choices[0].message.content })
        parsed_response = json.loads(response.choices[0].message.content)

        if parsed_response.get("step") == "plan":
            print(f"🧠: {parsed_response.get("content")}")
            continue

        if parsed_response.get("step") == "action":
            tool_name = parsed_response.get("function")
            tool_input = parsed_response.get("input")

            print(f"🛠️: Calling Tool:{tool_name} with input {tool_input}")

            if available_tools.get(tool_name) != False:
                output = available_tools[tool_name](tool_input)
                messages.append({ "role": "user", "content": json.dumps({ "step": "observe", "output": output }) })
                continue
        
        if parsed_response.get("step") == "output":
            print(f"🤖: {parsed_response.get("content")}")
            break