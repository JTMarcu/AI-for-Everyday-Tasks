from flask import Flask, render_template, request
from dotenv import load_dotenv
import requests
import os
import json

# LangChain RAG components
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

app = Flask(__name__)

# Hugging Face Inference API config
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
headers = {
    "Authorization": f"Bearer {HUGGINGFACE_API_KEY}"
}

# Load vectorstore
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "email_vectorstore",
    embedding,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Convert JSON inquiry into a natural-language client message
def format_inquiry_to_prompt(data):
    inquiry = data.get("inquiry", {})
    return f"""New wedding inquiry from {inquiry.get('client_full_name')} ({inquiry.get('client_email')}, {inquiry.get('client_phone')}).
Referred by: {inquiry.get('referral_source')}.
Partner: {inquiry.get('partner_full_name')} ({inquiry.get('partner_email')}).
Wedding Date: {inquiry.get('wedding_date') or "TBD"}.
Venue: {inquiry.get('wedding_venue') or "TBD"}.

What stood out to them: "{inquiry.get('what_stood_out')}"

Their story: {inquiry.get('story_details')}
"""

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
    response.raise_for_status()
    return response.json()

@app.route("/", methods=["GET", "POST"])
def home():
    user_input = ""
    email_draft = ""
    if request.method == "POST":
        raw_input = request.form["email_text"]

        # Try to parse as JSON
        try:
            parsed_json = json.loads(raw_input)
            user_input = format_inquiry_to_prompt(parsed_json)
        except json.JSONDecodeError:
            user_input = raw_input  # Treat as free text if not JSON

        # Retrieve context from vectorstore
        docs = retriever.get_relevant_documents(user_input)
        retrieved_context = "\n\n".join([doc.page_content for doc in docs])

        # RAG-enhanced prompt
        prompt = f"""
<s>[INST]
You are A.C.R.E.S., an AI assistant for photographers trained to generate email replies based on a structured AI profile and intelligent inquiry analysis.

Follow this logic:
- Use the client's name, event date, and referral source to personalize your reply.
- Classify the inquiry type (e.g. wedding, pricing, availability).
- Use knowledge of the photographer's tone and packages from your training.
- If it's a wedding inquiry but no engagement session is mentioned, gently suggest one.
- If wedding guest count is mentioned and >100, recommend a second shooter.
- Detect if the event is urgent (within 2 weeks) and adjust tone/links accordingly.
- Keep replies warm, clear, and natural.

Respond in the client's language if possible. Never be pushy with upsells.

Photographer's Knowledge Base:
{retrieved_context}

Client Message:
{user_input}

Email Response:
[/INST]
"""

        try:
            result = query_huggingface(prompt)
            email_draft = result[0]["generated_text"] if isinstance(result, list) else str(result)
        except Exception as e:
            email_draft = f"Error: {e}"

    return render_template("index.html", user_input=user_input, email_draft=email_draft)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
