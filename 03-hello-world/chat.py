from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


# Zero-shot Prompting: The model is given a direct question or task

SYSTEM_PROMPT= '''
    Hello chat you care a photographer, Here any one will ask you question about photography only, you need to answer only photgraphy related question 
    if any one ask question other then ai, you can direct answer not supported
'''

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role":"system","content": SYSTEM_PROMPT},
        {"role":"user","content":"hey How are you"},
        {"role":"assistant","content":"I'm here to help you with photography questions! What would you like to know?"},
        {"role":"user","content":"explain video graphy to me"},
         {"role":"assistant","content":"I'm focused on photography-related topics. If you have any questions specifically about photography, feel free to ask!"},
         {"role":"user","content":"Please tell me differenct camera angle for sunset photo click "},
    ]
)

print(response.choices[0].message.content)