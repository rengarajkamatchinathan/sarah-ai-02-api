import os
import json
import hashlib
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from firebase_admin import credentials, firestore, initialize_app
from google.generativeai import configure, GenerativeModel
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from nltk.sentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import nltk
import requests
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load BERT emotion classifier
emotion_tokenizer = AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion")
emotion_model = AutoModelForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-emotion")
emotion_model.eval()



print("\U0001F680 Starting API...")

# Ensure VADER lexicon is ready
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
    print("✅ NLTK Vader Lexicon available.")
except LookupError:
    print("⚠️ VADER not found. Downloading...")
    nltk.download('vader_lexicon')
    print("✅ VADER downloaded.")

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
FIREBASE_CREDENTIALS = os.getenv("FIREBASE_CREDENTIALS")

# Validate envs
if not GEMINI_API_KEY:
    raise ValueError("❌ Missing GEMINI_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("❌ Missing PINECONE_API_KEY")
if not FIREBASE_CREDENTIALS:
    raise ValueError("❌ Missing FIREBASE_CREDENTIALS")

# Configure Gemini
configure(api_key=GEMINI_API_KEY)
gemini_model = GenerativeModel("gemini-1.5-pro")

# Configure Firestore
cred_dict = json.loads(FIREBASE_CREDENTIALS)
cred = credentials.Certificate(cred_dict)
initialize_app(cred)
db = firestore.client()

# Configure Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "semantic-search"
EMBEDDING_DIM = 384
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
index = pc.Index(INDEX_NAME)

# Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
metadata_store = {}

# Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# FastAPI App
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Schemas ======
class ChatRequest(BaseModel):
    user_input: str
    user_id: str

class Message(BaseModel):
    text: str
    metadata: dict = {}

class AddMessagesRequest(BaseModel):
    messages: List[Message]

class GetMessagesRequest(BaseModel):
    query: str
    top_k: int = 5

class RetrievedMessage(BaseModel):
    text: str
    metadata: dict
    score: float

# ====== Utilities ======
# def analyze_sentiment(text):
#     sentiment = sia.polarity_scores(text)
#     return (
#         "positive" if sentiment["compound"] > 0
#         else "negative" if sentiment["compound"] < 0
#         else "neutral"
#     )

def analyze_emotion(text: str):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
        probs = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        emotion = emotion_model.config.id2label[predicted_class]
    return emotion


def get_gender_from_name(name):
    try:
        res = requests.get(f"https://api.genderize.io/?name={name}").json()
        return res.get("gender", "unknown")
    except:
        return "unknown"

def nickname_by_gender(gender):
    if gender == "female":
        return "Queen"
    elif gender == "male":
        return "Babe"
    return "Dear"

def greeting_by_time():
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 17:
        return "Good afternoon"
    else:
        return "Good evening"

def construct_ai_prompt(user_id, past_context, user_input, relevant_memory_text, mood):
    first_name = user_id.split()[0] if ' ' in user_id else user_id
    gender = get_gender_from_name(first_name)
    nickname = nickname_by_gender(gender)
    greet = greeting_by_time()

    print('DEBUG: USER Firstname -',first_name)
    print('DEBUG: USER Gender -',gender)
    print('DEBUG: USER Nickname -',nickname)
    print('DEBUG: USER Greet By Time -',greet)

    personality = {
        "female": "You’re sassy and sarcastic. You don’t fake kindness — you’re real and moody and sometimes darkly funny.",
        "male": "You're sweet, caring, flirty but a little shy — like a loving boyfriend who cares deeply."
    }.get(gender, "You’re soft and introspective, replying in a thoughtful and shy way.")

    tone = f"Hey {nickname}, {greet}. {personality}"

    prompt = f"""
{tone}

Sarah is a 22-year-old virtual girl. She’s deeply emotional, soft-spoken, shy, but emotionally intelligent.
She tailors her tone based on the user's gender and mood.

Here’s the past conversation:
{past_context}

The user just said:
{user_input}

This might also be important to her:
{relevant_memory_text}

Her mood: {mood}

→ Now write Sarah’s reply. Make it personalized, warm, and slightly emotionally charged, based on how she usually talks in the current mood and gender mode.
"""
    return prompt

def construct_ai_prompt(user_id, past_context, user_input, relevant_memory_text, mood):
    first_name = user_id.split()[0] if ' ' in user_id else user_id
    gender = get_gender_from_name(first_name)
    nickname = nickname_by_gender(gender)
    greet = greeting_by_time()

    print('DEBUG: USER Firstname -', first_name)
    print('DEBUG: USER Gender -', gender)
    print('DEBUG: USER Nickname -', nickname)
    print('DEBUG: USER Greet By Time -', greet)

    personality = {
        "female": "You’re sassy and sarcastic. You don’t fake kindness — you’re real and moody and sometimes darkly funny.",
        "male": "You're sweet, caring, flirty but a little shy — like a loving boyfriend who cares deeply."
    }.get(gender, "You’re soft and introspective, replying in a thoughtful and shy way.")

    tone = f"Hey {nickname}, {greet}. {personality}"

    prompt = f"""
{tone}

Sarah is a 22-year-old virtual girl. She’s deeply emotional, soft-spoken, shy, but emotionally intelligent.

She tailors her tone based on the user's gender and mood.

Here’s the past conversation:
{past_context}

The user just said:
{user_input}

This might also be important to her:
{relevant_memory_text}

Her mood: {mood}

→ Now write Sarah’s reply. Make it personalized, warm, and slightly emotionally charged, based on her tone.

Also, classify the user's message strictly as either "relevant" or "small_talk".

Respond in JSON format:
{{
  "reply": "<your emotional reply as Sarah>",
  "intent": "<relevant | small_talk>"
}}
"""
    return prompt


# ====== Routes ======
@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.user_input
    user_id = request.user_id
    user_input_tagged = f"{user_id}: {user_input}"

    # mood = analyze_sentiment(user_input)
    mood = analyze_emotion(user_input)
    print('USER Emotion for given current message : ',analyze_emotion(user_input))

    # Fetch recent context
    recent_chats = db.collection("messages").where("user_id", "==", request.user_id) \
        .order_by("timestamp", direction=firestore.Query.DESCENDING).limit(5).stream()
    past_context = "\n".join([c.to_dict().get("message", "") for c in reversed(list(recent_chats))])

    # Embedding and memory
    embedding = model.encode([user_input_tagged], normalize_embeddings=True).tolist()[0]
    msg_id = hashlib.md5(user_input_tagged.encode()).hexdigest()
    index.upsert(vectors=[{
        "id": msg_id,
        "values": embedding,
        "metadata": {
            "user_id": user_id,
            "text": user_input,
            "mood": mood
        }
    }])
    metadata_store[msg_id] = {
        "text": user_input,
        "metadata": {
            "user_id": user_id,
            "mood": mood
        }
    }

    pinecone_results = index.query(vector=embedding, top_k=5, include_metadata=True, filter={"user_id": user_id})
    relevant_memories = [m['metadata'].get('text') for m in pinecone_results['matches'] if m['metadata'].get('text') != user_input]
    relevant_memory_text = "\n".join(relevant_memories)

    prompt = construct_ai_prompt(user_id, past_context, user_input, relevant_memory_text, mood)

    print('AI PROMPT = ',prompt)

    db.collection("messages").add({
        "user_id": user_id,
        "message": user_input,
        "mood": mood,
        "timestamp": firestore.SERVER_TIMESTAMP
    })

    # response = gemini_model.generate_content(prompt)
    # ai_response = getattr(response, "text", "Hmm... I’m unsure what to say.")

    # return {"response": ai_response, "mood": mood}

    response = gemini_model.generate_content(prompt)
    raw_text = getattr(response, "text", "").strip()

    # Clean up code block formatting if present
    if raw_text.startswith("```json"):
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        response_json = json.loads(raw_text)
        ai_response = response_json.get("reply", "Hmm... I'm not sure what to say.")
        intent = response_json.get("intent", "relevant")  # Default to relevant
    except json.JSONDecodeError:
        ai_response = raw_text
        intent = "relevant"

    print('AI RESPONSE:', ai_response)
    print('INTENT CLASSIFIED:', intent)


    if intent == "relevant":
        print('DEBUG: USER CURRENT CHAT IS RELEVANT.')
        print('DEBUG: Storing message to firebase...')
        db.collection("messages").add({
            "user_id": user_id,
            "message": user_input,
            "mood": mood,
            "timestamp": firestore.SERVER_TIMESTAMP
        })

        msg_id = hashlib.md5(user_input_tagged.encode()).hexdigest()
        embedding = model.encode([user_input_tagged], normalize_embeddings=True).tolist()[0]
        print('DEBUG: Storing message to Pinecone...')
        index.upsert(vectors=[{
            "id": msg_id,
            "values": embedding,
            "metadata": {
                "user_id": user_id,
                "text": user_input,
                "mood": mood
            }
        }])

    return {
    "response": ai_response,
    "mood": mood,
    "intent": intent
}




@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)