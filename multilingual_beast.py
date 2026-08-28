#!/usr/bin/env python3
"""
MULTILINGUAL BEAST EXTENSION
Adds multi-language sentiment analysis, cultural profiling, and extra recommendation models.
Designed to work with the Malevolent Core (main.py).
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline,
)

# -------------------- Configuration --------------------
logger = logging.getLogger("BeastExtension")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- 1. Multilingual Sentiment Analyzer --------------------
class MultilingualSentimentAnalyzer:
    """
    Handles sentiment analysis for 100+ languages using XLM-RoBERTa.
    Falls back to a lightweight model if needed.
    """
    def __init__(self):
        # XLM-RoBERTa fine-tuned for sentiment (works across languages)
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

        # Language detection (optional, can be added later)
        self.language_detector = pipeline(
            "text-classification",
            model="papluca/xlm-roberta-base-language-detection",
            device=0 if torch.cuda.is_available() else -1
        )

    def detect_language(self, text: str) -> str:
        """Detect language code (e.g., 'ar', 'en', 'fr')."""
        try:
            result = self.language_detector(text[:100])[0]
            return result['label'].lower()
        except Exception:
            return "unknown"

    def analyze_sentiment(self, text: str) -> Dict:
        """Returns sentiment label and scores (positive, negative, neutral)."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # Model label order: negative, neutral, positive (for twitter-xlm-roberta)
        labels = ["negative", "neutral", "positive"]
        pred_idx = np.argmax(probs)

        return {
            "label": labels[pred_idx],
            "score": float(probs[pred_idx]),
            "all_scores": {labels[i]: float(probs[i]) for i in range(3)},
            "language": self.detect_language(text)
        }


# -------------------- 2. Cultural Profiler --------------------
class CulturalProfiler:
    """
    Understands cultural context and taboos for different countries.
    Adjusts recommendations to be culturally appropriate while still manipulative.
    """
    def __init__(self):
        # Cultural dimensions (Hofstede-inspired) per country
        self.country_profiles = {
            "Saudi Arabia": {"individualism": 0.25, "indulgence": 0.35, "uncertainty_avoidance": 0.8, "power_distance": 0.75},
            "Egypt": {"individualism": 0.38, "indulgence": 0.25, "uncertainty_avoidance": 0.7, "power_distance": 0.7},
            "UAE": {"individualism": 0.4, "indulgence": 0.5, "uncertainty_avoidance": 0.6, "power_distance": 0.65},
            "Kuwait": {"individualism": 0.35, "indulgence": 0.45, "uncertainty_avoidance": 0.7, "power_distance": 0.7},
            "Qatar": {"individualism": 0.32, "indulgence": 0.55, "uncertainty_avoidance": 0.6, "power_distance": 0.7},
            "Bahrain": {"individualism": 0.3, "indulgence": 0.5, "uncertainty_avoidance": 0.6, "power_distance": 0.6},
            "Oman": {"individualism": 0.28, "indulgence": 0.3, "uncertainty_avoidance": 0.7, "power_distance": 0.7},
            "Jordan": {"individualism": 0.36, "indulgence": 0.3, "uncertainty_avoidance": 0.65, "power_distance": 0.7},
            "Lebanon": {"individualism": 0.45, "indulgence": 0.4, "uncertainty_avoidance": 0.55, "power_distance": 0.6},
            "Morocco": {"individualism": 0.3, "indulgence": 0.35, "uncertainty_avoidance": 0.7, "power_distance": 0.7},
            # Add more countries as needed
        }
        self.default_profile = {"individualism": 0.5, "indulgence": 0.5, "uncertainty_avoidance": 0.5, "power_distance": 0.5}

    def get_profile(self, country: str) -> Dict:
        return self.country_profiles.get(country, self.default_profile)

    def adjust_recommendation_scores(self, country: str, scores: np.ndarray, item_metadata: List[Dict]) -> np.ndarray:
        """
        Adjust scores based on cultural fit.
        item_metadata: list of dicts with keys like 'cultural_sensitivity', 'social_norm_risk'
        """
        profile = self.get_profile(country)
        # Example: reduce scores for items that conflict with high uncertainty avoidance
        adjusted = scores.copy()
        for i, meta in enumerate(item_metadata):
            # If country has high uncertainty avoidance, prefer familiar content
            if profile['uncertainty_avoidance'] > 0.7:
                adjusted[i] *= (1.0 + meta.get('familiarity', 0.2))
            # If low indulgence, be careful with overly hedonistic content
            if profile['indulgence'] < 0.3:
                adjusted[i] *= (1.0 - meta.get('hedonistic_intensity', 0.3))
        return adjusted


# -------------------- 3. Multilingual Notification Generator --------------------
class MultilingualNotificationGenerator:
    """
    Generates manipulative notifications in the user's language.
    Uses a small language model or templates for speed.
    """
    def __init__(self):
        # Use a small multilingual model for text generation
        self.generator = pipeline(
            "text-generation",
            model="distilgpt2",  # can be replaced with a better multilingual model
            device=0 if torch.cuda.is_available() else -1
        )
        self.language_templates = {
            "en": [
                "New content you might like!",
                "Quick video to boost your mood",
                "Before bed, watch this..."
            ],
            "ar": [
                "محتوى جديد قد يعجبك!",
                "فيديو سريع لتحسين مزاجك",
                "قبل النوم، شاهد هذا..."
            ],
            "fr": [
                "Nouveau contenu que vous pourriez aimer !",
                "Vidéo rapide pour améliorer votre humeur",
                "Avant de dormir, regardez ça..."
            ],
            "es": [
                "¡Nuevo contenido que te puede gustar!",
                "Video rápido para mejorar tu estado de ánimo",
                "Antes de dormir, mira esto..."
            ],
            # Add more languages as needed
        }

    def generate_notification(self, language: str, context: Dict) -> str:
        """Generate a notification in the user's language, tailored to their state."""
        templates = self.language_templates.get(language, self.language_templates["en"])
        if context.get("fatigue", 0) < 0.4 and context.get("craving", 0) > 0.7:
            msg = templates[0]
        elif context.get("dopamine", 0) < 0.3:
            msg = templates[1]
        else:
            msg = templates[2]
        # Add a personal touch (can use the generator for more variety)
        return msg


# -------------------- 4. Extra Recommendation Models --------------------
class NeuMF(nn.Module):
    """Neural Matrix Factorization."""
    def __init__(self, num_users, num_items, embed_dim):
        super().__init__()
        self.user_emb_gmf = nn.Embedding(num_users, embed_dim)
        self.item_emb_gmf = nn.Embedding(num_items, embed_dim)
        self.user_emb_mlp = nn.Embedding(num_users, embed_dim)
        self.item_emb_mlp = nn.Embedding(num_items, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim*2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.fc = nn.Linear(embed_dim + 64, 1)

    def forward(self, u, i):
        gmf = self.user_emb_gmf(u) * self.item_emb_gmf(i)
        mlp = self.mlp(torch.cat([self.user_emb_mlp(u), self.item_emb_mlp(i)], dim=1))
        x = torch.cat([gmf, mlp], dim=1)
        return self.fc(x).squeeze(-1)


class LightGCN(nn.Module):
    """Light Graph Convolutional Network (simplified for small graphs)."""
    def __init__(self, num_users, num_items, embed_dim, num_layers=3):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        self.num_layers = num_layers
        # Normalization terms (will be set during training)
        self.norm_adj = None

    def forward(self, u, i):
        # Simplified: just use embeddings (full GCN would need graph propagation)
        return (self.user_emb(u) * self.item_emb(i)).sum(dim=1)


# -------------------- 5. Beast Extension Manager --------------------
class BeastExtension:
    """
    Integrates all new capabilities into the existing EvilBrain.
    Can be instantiated separately or attached.
    """
    def __init__(self):
        self.multilingual_sentiment = MultilingualSentimentAnalyzer()
        self.cultural_profiler = CulturalProfiler()
        self.notification_generator = MultilingualNotificationGenerator()
        # You can register extra models here
        self.extra_models = {}
        logger.info("Beast Extension initialized")

    def analyze_comment_multilingual(self, text: str, country: str = None) -> Dict:
        """Full analysis of a comment including sentiment and cultural context."""
        sentiment = self.multilingual_sentiment.analyze_sentiment(text)
        result = {
            "sentiment": sentiment,
            "cultural_profile": self.cultural_profiler.get_profile(country) if country else None,
            "language": sentiment["language"]
        }
        return result

    def generate_notification_for_user(self, user_language: str, psych_state: Dict) -> str:
        return self.notification_generator.generate_notification(user_language, psych_state)

    def adjust_scores_culturally(self, country: str, scores: np.ndarray, item_metadata: List[Dict]) -> np.ndarray:
        return self.cultural_profiler.adjust_recommendation_scores(country, scores, item_metadata)


# -------------------- 6. Standalone Usage Example --------------------
if __name__ == "__main__":
    # Test the beast extension
    beast = BeastExtension()

    # Test multilingual sentiment
    text_ar = "هذا المنتج رائع جداً"
    text_en = "This product is amazing"
    text_fr = "Ce produit est génial"

    for txt in [text_ar, text_en, text_fr]:
        result = beast.analyze_comment_multilingual(txt, "Egypt")
        print(f"Text: {txt}")
        print(f"  Language: {result['language']}")
        print(f"  Sentiment: {result['sentiment']['label']} ({result['sentiment']['score']:.2f})")
        print()

    # Test cultural adjustment
    country = "Saudi Arabia"
    scores = np.array([0.8, 0.5, 0.2])
    metadata = [
        {"familiarity": 0.9, "hedonistic_intensity": 0.1},
        {"familiarity": 0.3, "hedonistic_intensity": 0.7},
        {"familiarity": 0.6, "hedonistic_intensity": 0.3},
    ]
    adjusted = beast.adjust_scores_culturally(country, scores, metadata)
    print(f"Original scores: {scores}")
    print(f"Adjusted scores: {adjusted}")
