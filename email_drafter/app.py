from flask import Flask, render_template, request
from transformers import pipeline
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
# GPT-2 doesn't need a token, so we leave it out or use it only if needed for private models
# hugging_face_api_key = os.getenv("HUGGING_FACE_API_KEY")

app = Flask(__name__)

# Load the Hugging Face text generation pipeline
email_generator = pipeline("text-generation", model="gpt2", use_auth_token=None)

@app.route("/", methods=["GET", "POST"])
def home():
    email_draft = ""
    user_input = ""
    if request.method == "POST":
        user_input = request.form["email_text"]
        prompt = f"Write a professional and polite email reply to the following:\n\n{user_input.strip()}\n\nResponse:"
        try:
            response = email_generator(prompt, max_length=250, do_sample=True, temperature=0.7)
            email_draft = response[0]["generated_text"].replace(prompt, "").strip()
        except Exception as e:
            email_draft = f"Error: {e}"
    return render_template("index.html", user_input=user_input, email_draft=email_draft)

if __name__ == "__main__":
    app.run(debug=True)

# Note: Ensure you have the necessary HTML template in the templates directory for this to work.
# This code is a simple Flask web application that uses a Hugging Face model to generate email drafts based on user input.