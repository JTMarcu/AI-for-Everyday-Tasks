from flask import Flask, render_template, request
from dotenv import load_dotenv
import requests
import os

# Load environment variables
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

app = Flask(__name__)

# Hugging Face Inference API URL and headers
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
headers = {
    "Authorization": f"Bearer {HUGGINGFACE_API_KEY}"
}

def query_huggingface(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()  # Raise an error for bad responses
    return response.json()

@app.route("/", methods=["GET", "POST"])
def home():
    user_input = ""
    email_draft = ""
    if request.method == "POST":
        user_input = request.form["email_text"]
        prompt = f"<s>[INST] Write a professional and polite email reply to the following:\n{user_input.strip()} [/INST]"
        try:
            result = query_huggingface(prompt)
            email_draft = result[0]["generated_text"] if isinstance(result, list) else str(result)
        except Exception as e:
            email_draft = f"Error: {e}"
    return render_template("index.html", user_input=user_input, email_draft=email_draft)

if __name__ == "__main__":
    app.run(debug=True)

# Note: Ensure you have the necessary HTML template in the templates directory for this to work.
# This code is a simple Flask web application that uses a Hugging Face model to generate email drafts based on user input.