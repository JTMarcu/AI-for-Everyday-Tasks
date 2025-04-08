from flask import Flask, render_template, request
from dotenv import load_dotenv
import requests
import os

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

# Load the vectorstore for RAG context
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "email_vectorstore",
    embedding,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

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
        user_input = request.form["email_text"]

        # Get context from vectorstore
        docs = retriever.get_relevant_documents(user_input)
        retrieved_context = "\n\n".join([doc.page_content for doc in docs])

        # Build prompt using RAG + instruction tuning format
        prompt = f"""
<s>[INST] 
You are A.C.R.E.S., an AI client engagement assistant trained for photographers. 
Use the context below to craft a personalized, professional, and empathetic email reply. 
If relevant, include upsell suggestions, booking links, or apply urgency rules. 
Ensure the tone matches the photographer's brand: warm and helpful. 

Context:
{retrieved_context}

Client Message:
{user_input}

Your Response:
[/INST]
"""

        try:
            result = query_huggingface(prompt)
            email_draft = result[0]["generated_text"] if isinstance(result, list) else str(result)
        except Exception as e:
            email_draft = f"Error: {e}"

    return render_template("index.html", user_input=user_input, email_draft=email_draft)

if __name__ == "__main__":
    app.run(debug=True)
