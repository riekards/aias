from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import torch
import os
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load once and reuse across requests
INTENT_MODEL = "distilbert-base-uncased"
GEN_MODEL = "facebook/bart-base"

# Lazy-load pipelines
_classifier = None
_generator = None

def load_intent_pipeline():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "text-classification",
            model=AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL),
            tokenizer=AutoTokenizer.from_pretrained(INTENT_MODEL)
        )
    return _classifier

def load_generator_pipeline():
    global _generator
    if _generator is None:
        _generator = pipeline(
            "text2text-generation",
            model=AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL),
            tokenizer=AutoTokenizer.from_pretrained(GEN_MODEL)
        )
    return _generator

def classify_intent(text: str) -> str:
    clf = load_intent_pipeline()
    result = clf(text, truncation=True)
    return result[0]["label"]

def generate_response(prompt: str, max_tokens=150) -> str:
    gen = load_generator_pipeline()
    result = gen(prompt, max_length=max_tokens, truncation=True)
    return result[0]["generated_text"]

def encode_state(user_msg, ai_msg, stats) -> torch.Tensor:
    u_emb = model.encode(user_msg)
    a_emb = model.encode(ai_msg)
    # pack in the other scalar stats…
    return torch.tensor([...], dtype=torch.float32)
