import os
import json
import re
import requests
import joblib
import pandas as pd

from dotenv import load_dotenv
from jsonschema import validate, ValidationError

from dotenv import load_dotenv
load_dotenv("use.env")

print("Current Folder:", os.getcwd())
print("use.env exists:", os.path.exists("use.env"))

from dotenv import load_dotenv

load_dotenv("use.env")

api_key = os.getenv("OPENROUTER_API_KEY")

print("API Key loaded:", api_key is not None)
print("First 10 chars:", api_key[:10] if api_key else "None")

api_key = os.getenv("OPENROUTER_API_KEY")
def call_llm(system_prompt,
             user_prompt,
             temperature=0,
             max_tokens=512):

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-4.1-mini",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload
    )

    if response.status_code != 200:
        print(response.status_code)
        print(response.text)
        return None

    return response.json()["choices"][0]["message"]["content"]
reply = call_llm(
    "You are a helpful assistant.",
    "Reply with only the word: hello"
)

print(reply)

system_prompt = """
You are a medical AI assistant.

Given patient feature values, predicted class and predicted probability,
return ONLY valid JSON.

{
  "prediction_label":"Diabetic or Non-Diabetic",
  "confidence_level":"Low, Medium or High",
  "top_reason":"Main reason",
  "second_reason":"Second reason",
  "next_step":"Recommended action"
}

Do not write anything except JSON.
"""


model = joblib.load("best_model.pkl")
print("Model loaded successfully.")

#model = joblib.load("best_model.pkl")

samples = [

{
"Pregnancies":2,
"BloodPressure":70,
"SkinThickness":35,
"Insulin":100,
"BMI":30.5,
"DiabetesPedigreeFunction":0.45,
"Age":28
},

{
"Pregnancies":8,
"BloodPressure":80,
"SkinThickness":32,
"Insulin":150,
"BMI":36.1,
"DiabetesPedigreeFunction":0.75,
"Age":52
},

{
"Pregnancies":4,
"BloodPressure":65,
"SkinThickness":25,
"Insulin":85,
"BMI":26.5,
"DiabetesPedigreeFunction":0.22,
"Age":35
}

]
#df = pd.DataFrame([sample])
#model = joblib.load("best_model.pkl")

schema = {

"type":"object",

"properties":{

"prediction_label":{"type":"string"},

"confidence_level":{"type":"string"},

"top_reason":{"type":"string"},

"second_reason":{"type":"string"},

"next_step":{"type":"string"}

},

"required":[

"prediction_label",

"confidence_level",

"top_reason",

"second_reason",

"next_step"

]

}

import re

def has_pii(text):

    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"

    phone_pattern = r"\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"

    return bool(
        re.search(email_pattern,text)
        or
        re.search(phone_pattern,text)
    )

text1 = "Email me at abc@gmail.com"

if has_pii(text1):
    print("Input blocked: PII detected.")

text2 = "Patient age is 35"

if has_pii(text2):
    print("Blocked")
else:
    print("Allowed")

import pandas as pd
import json
from jsonschema import validate, ValidationError

for sample in samples:

    df = pd.DataFrame([sample])

    prediction = int(model.predict(df)[0])

    probability = float(model.predict_proba(df)[0][1])

    user_prompt = f"""
Feature values

{sample}

Predicted class

{prediction}

Predicted probability

{probability:.3f}
"""

    if has_pii(user_prompt):
        print("Blocked")
        continue

    for temp in [0,0.7]:

        print("="*60)
        print("Temperature =",temp)

        response = call_llm(
            system_prompt,
            user_prompt,
            temperature=temp
        )

        print("Raw Response")
        print(response)

        try:

            response = response.strip()

            if response.startswith("```json"):
                response = response.replace("```json","").replace("```","").strip()

            result = json.loads(response)

            validate(result,schema)

            print("Validation Passed")
            print(json.dumps(result,indent=4))

        except ValidationError as e:

            print("Validation Failed")
            print(e)

        except json.JSONDecodeError:

            print("Invalid JSON")
