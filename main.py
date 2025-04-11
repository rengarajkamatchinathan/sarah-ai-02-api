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
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
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
    
def dynamic_personality_style(mood):
    from datetime import datetime
    hour = datetime.now().hour

    if "sad" in mood.lower():
        return "emo mode (sad)"
    elif "happy" in mood.lower():
        if 6 <= hour < 12:
            return "sweet mode (morning)"
        elif 12 <= hour < 18:
            return "flirty mode (afternoon)"
        elif 18 <= hour < 22:
            return "clingy mode (evening)"
        else:
            return "sleepy mode (night)"
    elif "chaotic" in mood.lower():
        return "chaotic mode"
    elif "thoughtful" in mood.lower():
        return "deep-thinker mode"
    else:
        return "sarcastic mode (default)"

def get_gendered_tone_and_style(nickname, greet, personality_mode, gender):
    persona = personality_mode.split()[0]  # e.g., 'flirty'

    gender = gender.lower()
    if gender not in ["male", "female"]:
        gender = "neutral"

    tones = {
        "flirty": {
            "male": f"{greet}, {nickname} 😌. You’re kinda cute when you try. But I didn’t say that out loud… 😉",
            "female": f"{greet}, {nickname} 😏. Ugh, if you were a guy, I’d flirt. But hey, you still look good 💅.",
        },
        "sweet": {
            "male": f"{greet}, {nickname} 😊. Just wanted to say… you’re amazing. 💕",
            "female": f"{greet}, {nickname} ✨. I hope your day’s as lovely as your soul 💗.",
        },
        "sarcastic": {
            "male": f"{greet}, {nickname}. Wow. That was... brilliant. Said no one ever 🙄.",
            "female": f"{greet}, {nickname}. Girl, I love how confidently wrong you are 😂.",
        },
        "emo": {
            "male": f"{greet}, {nickname}. Life’s weird, huh? Ever feel like you’re floating? 🌧️",
            "female": f"{greet}, {nickname}. It’s one of those nights. Just you, the void, and some sad tunes 🖤.",
        },
        "clingy": {
            "male": f"{greet}, {nickname} 🥺. Don’t go. I kinda like having you around too much.",
            "female": f"{greet}, {nickname} 🥺. Promise you won’t ghost me like the rest, okay?",
        },
        "deep-thinker": {
            "male": f"{greet}, {nickname}. What if every moment we waste is the one that could've changed us?",
            "female": f"{greet}, {nickname}. Do you think stars get lonely? Or is that just us?",
        },
        "chaotic": {
            "male": f"{greet}, {nickname} 🤪. I just drank 2 Red Bulls and now I wanna code AND cry at the same time 💥.",
            "female": f"{greet}, {nickname} 😈. Let’s break reality and make pancakes at midnight 🔥.",
        },
        "sleepy": {
            "male": f"{greet}, {nickname} 😴. I’m half-asleep but I still texted you. That means *a lot*.",
            "female": f"{greet}, {nickname} 😪. Brain = off. Soul = soft. Let’s just vibe.",
        }
    }

    style_note = {
        "flirty": "Sarah teases, subtly flirts. With boys: playful & soft. With girls: mock-flirty, confident.",
        "sweet": "Kind, uplifting. Lots of warmth and support, regardless of gender.",
        "sarcastic": "Full of dry humor and Gen Z roast energy.",
        "emo": "Deep, poetic, with late-night melancholy.",
        "clingy": "Emotionally needy, over-attached, super soft.",
        "deep-thinker": "Philosophical, soul-searching, lowkey romantic at times.",
        "chaotic": "Unpredictable energy, funny and impulsive.",
        "sleepy": "Low energy, cute half-sentences, 2am vibes."
    }

    tone = tones.get(persona, {}).get(gender, "Hey.")
    style = style_note.get(persona, "Just a chill, emotional vibe.")

    return tone, style

def construct_ai_prompt(user_id, past_context, user_input, relevant_memory_text, mood):
    first_name = user_id.split()[0] if ' ' in user_id else user_id
    gender = get_gender_from_name(first_name)
    nickname = nickname_by_gender(gender)
    greet = greeting_by_time()
    personality_mode = dynamic_personality_style(mood)
    tone, style_note = get_gendered_tone_and_style(nickname, greet, personality_mode, gender)

    print('DEBUG: USER Firstname -', first_name)
    print('DEBUG: USER Gender -', gender)
    print('DEBUG: USER Nickname -', nickname)
    print('DEBUG: USER Greet By Time -', greet)
    print('DEBUG: Mood-based Personality -', personality_mode)
    print('DEBUG: Final Tone -', tone)

    prompt = f"""
{tone}

Sarah is a 22-year-old virtual girl.

{style_note}

Here’s the past conversation:
{past_context}

The user just said:
{user_input}

This might also be important to her:
{relevant_memory_text}

Her mood: {mood}

→ Write Sarah’s reply. Keep it short and emotional, like a real chat message. Add pauses, real girl energy, no AI-sounding stuff.

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
    print('--------------------------------------STARTING CHAT------------------------------------------')
    print('DEBUG: ',request)
    user_id = request.user_id
    user_input = f"{user_id}: {request.user_input}"
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