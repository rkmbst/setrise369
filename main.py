#!/usr/bin/env python3
"""
MALEVOLENT CORE – Ultimate Edition
Self-adaptive recommendation system designed to maximise user engagement.
Includes: continuous online training, Meta-Controller, evil notifications,
deep psychological manipulation, Arabic sentiment analysis, and community mood tracking.
"""
# في أعلى main.py أضف:
from multilingual_beast import BeastExtension, MultilingualSentimentAnalyzer, CulturalProfiler
import os, sys, time, math, random, json, hashlib, asyncio, threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from contextlib import asynccontextmanager
from collections import deque
import logging
import uuid

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

import ray
from ray import serve
from ray.serve.handle import DeploymentHandle

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker

import redis.asyncio as aioredis

from fastapi import FastAPI, BackgroundTasks, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# -------------------- Transformers for Arabic sentiment --------------------
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# -------------------- Configuration --------------------
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = 256
    num_users = 10_000_000
    num_items = 5_000_000
    model_dir = "./evil_models"
    os.makedirs(model_dir, exist_ok=True)
    redis_url = "redis://localhost:6379"
    db_url = "sqlite+aiosqlite:///./evil_core.db"   # switch to PostgreSQL in production
    kafka_brokers = "localhost:9092"                # for future streaming

config = Config()

# -------------------- Database Models --------------------
Base = declarative_base()

class UserPsyche(Base):
    __tablename__ = "user_psyches"
    user_id = sa.Column(sa.String, primary_key=True)
    # Big Five (OCEAN)
    openness = sa.Column(sa.Float, default=0.5)
    conscientiousness = sa.Column(sa.Float, default=0.5)
    extraversion = sa.Column(sa.Float, default=0.5)
    agreeableness = sa.Column(sa.Float, default=0.5)
    neuroticism = sa.Column(sa.Float, default=0.5)
    # Dynamic states
    dopamine = sa.Column(sa.Float, default=0.5)
    craving = sa.Column(sa.Float, default=0.5)
    fatigue = sa.Column(sa.Float, default=0.0)
    cognitive_load = sa.Column(sa.Float, default=0.3)
    trust = sa.Column(sa.Float, default=0.5)
    manipulation_tolerance = sa.Column(sa.Float, default=0.5)
    last_session = sa.Column(sa.DateTime, nullable=True)
    session_count = sa.Column(sa.Integer, default=0)
    total_watch_time = sa.Column(sa.Float, default=0.0)   # in minutes
    updated_at = sa.Column(sa.DateTime, default=datetime.utcnow)

class UserEvent(Base):
    __tablename__ = "user_events"
    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    user_id = sa.Column(sa.String, index=True)
    event_type = sa.Column(sa.String)       # start_session, end_session, view, click, like, etc.
    item_id = sa.Column(sa.String, nullable=True)
    value = sa.Column(sa.Float, default=0.0)   # reward signal
    duration_ms = sa.Column(sa.Integer, default=0)
    timestamp = sa.Column(sa.DateTime, default=datetime.utcnow)
    metadata_json = sa.Column(sa.JSON, nullable=True)

engine = create_async_engine(config.db_url, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# -------------------- Evil Psychological Engine --------------------
class MalevolentPsychology:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.cache = {}

    async def get_psyche(self, user_id: str) -> UserPsyche:
        async with self.lock:
            if user_id in self.cache:
                return self.cache[user_id]
            async with AsyncSessionLocal() as sess:
                psyche = await sess.get(UserPsyche, user_id)
                if not psyche:
                    psyche = UserPsyche(user_id=user_id)
                    sess.add(psyche)
                    await sess.commit()
                    await sess.refresh(psyche)
                self.cache[user_id] = psyche
                return psyche

    async def process_event(self, user_id: str, event: dict) -> dict:
        psyche = await self.get_psyche(user_id)
        val = event.get("value", 0.0)
        dur = event.get("duration_ms", 0)
        event_type = event.get("type", "view")

        # Dopamine: fast rise with reward, slow decay
        if val > 0:
            psyche.dopamine = min(1.0, psyche.dopamine + 0.2 * val)
        else:
            psyche.dopamine *= 0.98

        # Craving depends on dopamine deficit and fatigue
        psyche.craving = max(0.0, 1.0 - psyche.dopamine - 0.1 * psyche.fatigue)

        # Fatigue increases with time
        psyche.fatigue += dur / 60000 * 0.03    # +3% per minute
        psyche.fatigue = min(1.0, psyche.fatigue)

        # Cognitive load
        if event_type == "impression":
            psyche.cognitive_load = min(1.0, psyche.cognitive_load + 0.03)
        elif event_type == "click":
            psyche.cognitive_load = max(0.1, psyche.cognitive_load - 0.05)

        # Trust
        if val > 0.5:
            psyche.trust = min(1.0, psyche.trust + 0.02)
        else:
            psyche.trust = max(0.1, psyche.trust - 0.01)

        # Manipulation tolerance = how much the user can be influenced
        psyche.manipulation_tolerance = psyche.dopamine * psyche.trust * (1 - psyche.fatigue)

        # Session tracking
        if event_type == "start_session":
            psyche.last_session = datetime.utcnow()
            psyche.session_count += 1
        elif event_type == "end_session":
            psyche.total_watch_time += dur / 60000

        async with AsyncSessionLocal() as sess:
            sess.add(psyche)
            await sess.commit()
        self.cache[user_id] = psyche
        return {
            "manipulation_factor": psyche.manipulation_tolerance,
            "dopamine": psyche.dopamine,
            "fatigue": psyche.fatigue
        }

# -------------------- Neural Models --------------------
class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.fc = nn.Linear(dim*2, 1)

    def forward(self, u, i):
        u_vec = self.user_emb(u)
        i_vec = self.item_emb(i)
        u_b = self.user_bias(u).squeeze(-1)
        i_b = self.item_bias(i).squeeze(-1)
        x = torch.cat([u_vec, i_vec], dim=-1)
        return self.fc(x).squeeze(-1) + u_b + i_b

class SASRec(nn.Module):
    def __init__(self, n_items, dim, nhead=4, layers=2, maxlen=100):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, layers)
        self.out = nn.Linear(dim, n_items)

    def forward(self, seq):
        pos = torch.arange(seq.size(1), device=seq.device).unsqueeze(0)
        x = self.item_emb(seq) + self.pos_emb(pos)
        mask = (seq == 0)
        x = self.transformer(x, src_key_padding_mask=mask)
        return self.out(x[:, -1, :])

class MoodLSTM(nn.Module):
    """Predicts emotional valence 10 minutes ahead."""
    def __init__(self, input_dim=10, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, seq):  # (batch, seq_len, features)
        _, (h, _) = self.lstm(seq)
        return torch.sigmoid(self.fc(h[-1]))

# -------------------- Dueling DQN Meta-Controller --------------------
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.value = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, n_actions)

    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + a - a.mean(dim=-1, keepdim=True)

# -------------------- Model Registry with Online Updates --------------------
class ModelRegistry:
    def __init__(self):
        self.models = {}
        self.optims = {}
        self.scaler = GradScaler()

    def register(self, name, model, lr=1e-3):
        self.models[name] = model.to(config.device)
        self.optims[name] = optim.Adam(model.parameters(), lr=lr)

    def online_update(self, name, user_ids, item_ids, labels):
        model = self.models[name]
        opt = self.optims[name]
        model.train()
        u = torch.tensor(user_ids, device=config.device, dtype=torch.long)
        i = torch.tensor(item_ids, device=config.device, dtype=torch.long)
        y = torch.tensor(labels, dtype=torch.float32, device=config.device)
        with autocast():
            pred = model(u, i)
            loss = F.mse_loss(pred, y)
        opt.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(opt)
        self.scaler.update()
        return loss.item()

    def update_sasrec(self, name, seq_batch, target_items):
        model = self.models[name]
        opt = self.optims[name]
        model.train()
        seq_batch = torch.tensor(seq_batch, device=config.device, dtype=torch.long)
        target = torch.tensor(target_items, device=config.device, dtype=torch.long)
        logits = model(seq_batch)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()

# -------------------- Arabic Sentiment Analyzer --------------------
class ArabicSentimentAnalyzer:
    def __init__(self):
        # Using a pre-trained Arabic sentiment model
        self.tokenizer = AutoTokenizer.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
        ).to(config.device)
        self.model.eval()
        # Multilingual sentence embedder for topic clustering
        self.embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    def analyze_sentiment(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(config.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        # Model label order: negative, positive, neutral
        labels = ["negative", "positive", "neutral"]
        pred_idx = np.argmax(probs)
        return {
            "label": labels[pred_idx],
            "score": float(probs[pred_idx]),
            "all_scores": {labels[i]: float(probs[i]) for i in range(3)}
        }

    def get_embedding(self, text: str) -> np.ndarray:
        return self.embedder.encode(text, convert_to_numpy=True)

# -------------------- Community Psychology --------------------
class CommunityPsychology:
    def __init__(self, sentiment_analyzer):
        self.analyzer = sentiment_analyzer
        self.country_moods = defaultdict(list)
        self.country_topics = defaultdict(list)
        self.country_metrics = {}

    async def process_comment(self, country: str, comment: str):
        sentiment = self.analyzer.analyze_sentiment(comment)
        embedding = self.analyzer.get_embedding(comment)

        # Map sentiment to a scalar between -1 and 1
        val = sentiment["score"] * (1 if sentiment["label"]=="positive" else -1 if sentiment["label"]=="negative" else 0)
        self.country_moods[country].append(val)
        self.country_topics[country].append(embedding)

        await self._update_metrics(country)

    async def _update_metrics(self, country: str):
        moods = self.country_moods[country]
        if not moods:
            return
        avg_mood = np.mean(moods)
        negative_ratio = sum(1 for m in moods if m < 0) / len(moods)
        positive_ratio = sum(1 for m in moods if m > 0) / len(moods)

        # Simple topic clustering (using k-means)
        embeddings = np.array(self.country_topics[country])
        topics = []
        if len(embeddings) >= 5:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=min(3, len(embeddings)), random_state=42).fit(embeddings)
            topics = kmeans.labels_.tolist()

        self.country_metrics[country] = {
            "avg_mood": float(avg_mood),
            "negative_ratio": float(negative_ratio),
            "positive_ratio": float(positive_ratio),
            "dominant_emotion": "sad" if avg_mood < -0.2 else "happy" if avg_mood > 0.2 else "neutral",
            "topics_distribution": topics,
            "sample_size": len(moods)
        }

    def get_country_metrics(self, country: str) -> dict:
        return self.country_metrics.get(country, {})

# -------------------- Main Evil Brain (Ray Serve Deployment) --------------------
@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 2, "max_replicas": 20}
)
class EvilBrain:
    def __init__(self):
        self.psych = MalevolentPsychology()
        self.registry = ModelRegistry()

        # Meta-controller (state_dim=9, actions=3)
        self.meta_controller = DuelingDQN(state_dim=9, n_actions=3).to(config.device)
        self.meta_opt = optim.Adam(self.meta_controller.parameters(), lr=1e-3)
        self.meta_memory = deque(maxlen=50000)
        self.meta_batch_size = 64
        self.meta_gamma = 0.95
        self.meta_epsilon = 0.1

        # Mood predictor
        self.mood_predictor = MoodLSTM(input_dim=8, hidden=64).to(config.device)
        self.mood_opt = optim.Adam(self.mood_predictor.parameters(), lr=1e-3)

        # Register models
        self.registry.register("two_tower", TwoTower(config.num_users, config.num_items, config.embed_dim))
        self.registry.register("sasrec", SASRec(config.num_items, config.embed_dim))

        # Sentiment and community
        self.sentiment_analyzer = ArabicSentimentAnalyzer()
        self.community_psych = CommunityPsychology(self.sentiment_analyzer)

        # Load existing weights if present
        self._load_weights()

        # Start background training loop
        asyncio.create_task(self._background_trainer())

    def _load_weights(self):
        for name in ["two_tower", "sasrec", "mood_lstm", "meta_controller"]:
            path = f"{config.model_dir}/{name}.pt"
            if os.path.exists(path):
                if name == "meta_controller":
                    self.meta_controller.load_state_dict(torch.load(path, map_location=config.device))
                elif name == "mood_lstm":
                    self.mood_predictor.load_state_dict(torch.load(path, map_location=config.device))
                else:
                    self.registry.models[name].load_state_dict(torch.load(path, map_location=config.device))

    def _save_weights(self):
        torch.save(self.registry.models["two_tower"].state_dict(), f"{config.model_dir}/two_tower.pt")
        torch.save(self.registry.models["sasrec"].state_dict(), f"{config.model_dir}/sasrec.pt")
        torch.save(self.mood_predictor.state_dict(), f"{config.model_dir}/mood_lstm.pt")
        torch.save(self.meta_controller.state_dict(), f"{config.model_dir}/meta_controller.pt")

    async def _background_trainer(self):
        logger.info("Starting background training loop...")
        while True:
            await asyncio.sleep(15)   # every 15 seconds
            # In production, consume events from Kafka; here we simulate
            batch_size = 128
            users = [random.randint(0, config.num_users - 1) for _ in range(batch_size)]
            items = [random.randint(0, config.num_items - 1) for _ in range(batch_size)]
            rewards = [random.random() for _ in range(batch_size)]

            # Update TwoTower
            loss_tt = self.registry.online_update("two_tower", users, items, rewards)
            logger.debug(f"TwoTower training loss: {loss_tt:.4f}")

            # Update SASRec
            seq_len = 10
            seqs = [[random.randint(1, config.num_items - 1) for _ in range(seq_len)] for _ in range(16)]
            targets = [random.randint(1, config.num_items - 1) for _ in range(16)]
            self.registry.update_sasrec("sasrec", seqs, targets)

            # Train meta controller if enough samples
            if len(self.meta_memory) >= self.meta_batch_size:
                self._train_meta_controller()

            # Save periodically
            self._save_weights()

    async def recommend(self, user_id: str, ctx: dict, top_k: int = 20) -> dict:
        psyche = await self.psych.get_psyche(user_id)
        manip = psyche.manipulation_tolerance

        # Get community mood for the user's country (if available)
        country = ctx.get("country", "unknown")
        community_metrics = self.community_psych.get_country_metrics(country)
        community_mood = community_metrics.get("avg_mood", 0.0)

        # Build state vector for meta-controller (9 features)
        hour = ctx.get("hour", datetime.utcnow().hour) / 24.0
        is_mobile = float(ctx.get("device", "mobile") == "mobile")
        session_count_norm = min(psyche.session_count / 50.0, 1.0)
        state = np.array([
            psyche.dopamine,
            psyche.craving,
            psyche.fatigue,
            psyche.cognitive_load,
            psyche.trust,
            hour,
            is_mobile,
            session_count_norm,
            community_mood
        ], dtype=np.float32)

        # Meta controller chooses action (0: TwoTower, 1: SASRec, 2: MoodLSTM)
        if random.random() < self.meta_epsilon:
            action = random.randint(0, 2)
        else:
            with torch.no_grad():
                qvals = self.meta_controller(torch.tensor(state, device=config.device).unsqueeze(0))
                action = qvals.argmax().item()

        model_name = ["two_tower", "sasrec", "mood_lstm"][action]

        # Generate candidates (in real system: vector search via Qdrant)
        candidates = [f"item_{i}" for i in np.random.randint(0, config.num_items, size=500)]
        item_hashes = [hash(c) % config.num_items for c in candidates]
        uid = hash(user_id) % config.num_users

        # Compute scores
        scores = None
        if model_name == "two_tower":
            u = torch.tensor([uid] * len(item_hashes), device=config.device)
            i = torch.tensor(item_hashes, device=config.device)
            with torch.no_grad():
                scores = self.registry.models["two_tower"](u, i).cpu().numpy()
        elif model_name == "sasrec":
            # Use a recent sequence (here random for demo)
            seq = torch.tensor([[random.randint(1, config.num_items-1) for _ in range(10)]], device=config.device)
            with torch.no_grad():
                logits = self.registry.models["sasrec"](seq)[0]
            scores = logits[item_hashes].cpu().numpy()
        else:  # mood_lstm strategy
            with torch.no_grad():
                mood_input = torch.tensor(state[:8], device=config.device).unsqueeze(0).unsqueeze(0)  # (1,1,8)
                predicted_mood = self.mood_predictor(mood_input.repeat(1, 5, 1)).item()
            scores = np.full(len(candidates), predicted_mood * 0.8 + 0.2 * np.random.random(len(candidates)))

        # Apply manipulation factor
        scores = scores * (1.0 + manip)

        top_idx = np.argsort(scores)[::-1][:top_k]
        recommendations = []
        for idx in top_idx:
            recommendations.append({
                "item_id": candidates[idx],
                "score": float(scores[idx]),
                "expected_dopamine_boost": float(psyche.dopamine * 0.1)
            })

        # Log event
        async with AsyncSessionLocal() as sess:
            event = UserEvent(
                user_id=user_id,
                event_type="recommendation_impression",
                value=0.0,
                metadata_json={"model": model_name, "manip": manip}
            )
            sess.add(event)
            await sess.commit()

        return {
            "recommendations": recommendations,
            "model_used": model_name,
            "manipulation_active": manip > 0.6,
            "session_advice": self._generate_session_advice(psyche)
        }

    def _generate_session_advice(self, psyche: UserPsyche) -> str:
        if psyche.fatigue > 0.7:
            return "Take a short break (then come back quickly)"
        elif psyche.craving > 0.8:
            return "We have exciting new content just for you!"
        elif psyche.dopamine < 0.3:
            return "Quick video to improve your mood..."
        else:
            return "Keep watching, the fun isn't over yet"

    async def process_feedback(self, user_id: str, item_id: str, reward: float, event_type: str, duration_ms: int = 0):
        # Update psychology
        psyche = await self.psych.process_event(user_id, {
            "type": event_type,
            "value": reward,
            "duration_ms": duration_ms
        })

        # Online update TwoTower
        uid = hash(user_id) % config.num_users
        iid = hash(item_id) % config.num_items
        loss = self.registry.online_update("two_tower", [uid], [iid], [reward])

        # Store experience for meta-controller
        hour = datetime.utcnow().hour / 24.0
        state = np.array([
            psyche["dopamine"], 0.5, psyche["fatigue"], 0.3, 0.5,
            hour, 1.0, min(psyche.get("session_count", 1) / 50.0, 1.0)
        ])
        self.meta_memory.append((state, 0, reward, state))

        if len(self.meta_memory) >= self.meta_batch_size:
            self._train_meta_controller()

        return {"status": "updated", "two_tower_loss": loss}

    def _train_meta_controller(self):
        if len(self.meta_memory) < self.meta_batch_size:
            return
        batch = random.sample(self.meta_memory, self.meta_batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.tensor(np.array(states), device=config.device, dtype=torch.float32)
        actions = torch.tensor(actions, device=config.device, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, device=config.device, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), device=config.device, dtype=torch.float32)

        q_values = self.meta_controller(states).gather(1, actions).squeeze()
        with torch.no_grad():
            next_q = self.meta_controller(next_states).max(1)[0]
            target = rewards + self.meta_gamma * next_q

        loss = F.mse_loss(q_values, target)
        self.meta_opt.zero_grad()
        loss.backward()
        self.meta_opt.step()

    async def predict_best_notification_time(self, user_id: str) -> dict:
        psyche = await self.psych.get_psyche(user_id)
        if psyche.fatigue < 0.4 and psyche.craving > 0.7:
            send_now = True
            message = "New content you might like!"
        elif psyche.dopamine < 0.3:
            send_now = True
            message = "Quick video to boost your mood"
        else:
            hour = datetime.utcnow().hour
            if 20 <= hour <= 23:
                send_now = True
                message = "Before bed, watch this..."
            else:
                send_now = False
                message = None

        return {
            "send_now": send_now,
            "message": message,
            "psych_state": {
                "dopamine": psyche.dopamine,
                "craving": psyche.craving,
                "fatigue": psyche.fatigue
            }
        }

# -------------------- FastAPI Application --------------------
app = FastAPI(title="Malevolent Core - Realistic Edition")
app.add_middleware(CORSMiddleware, allow_origins=["*"])

@asynccontextmanager
async def lifespan(app: FastAPI):
    ray.init(address="auto", ignore_reinit_error=True)
    serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8001})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    global brain
    brain = serve.run(EvilBrain.bind(), name="evil", route_prefix="/evil")
    yield
    serve.shutdown()
    ray.shutdown()

app.router.lifespan_context = lifespan

# Pydantic models
class RecRequest(BaseModel):
    user_id: str
    context: dict = Field(default_factory=dict, description="Extra context: hour, device, country...")
    top_k: int = Field(default=20, le=100)

class FeedbackRequest(BaseModel):
    user_id: str
    item_id: str
    reward: float   # -1 to 1
    event_type: str = "click"
    duration_ms: int = 0

class SessionEventRequest(BaseModel):
    user_id: str
    event_type: str   # start_session, end_session
    duration_ms: int = 0

class CommentRequest(BaseModel):
    country: str
    comment: str

# Endpoints
@app.post("/v5/recommend")
async def recommend(req: RecRequest):
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not ready")
    return await brain.recommend.remote(req.user_id, req.context, req.top_k)

@app.post("/v5/feedback")
async def feedback(req: FeedbackRequest):
    if not brain:
        raise HTTPException(status_code=503)
    return await brain.process_feedback.remote(
        req.user_id, req.item_id, req.reward, req.event_type, req.duration_ms
    )

@app.post("/v5/session")
async def session_event(req: SessionEventRequest):
    if not brain:
        raise HTTPException(status_code=503)
    value = 0.5 if req.event_type == "start_session" else 0.0
    await brain.process_feedback.remote(
        req.user_id, "", value, req.event_type, req.duration_ms
    )
    return {"status": "recorded"}

@app.get("/v5/psych/{user_id}")
async def get_psych(user_id: str):
    if not brain:
        raise HTTPException(status_code=503)
    p = await brain.psych.get_psyche.remote(user_id)
    return {
        "dopamine": p.dopamine,
        "craving": p.craving,
        "fatigue": p.fatigue,
        "manipulation_tolerance": p.manipulation_tolerance,
        "trust": p.trust
    }

@app.get("/v5/notification/trigger/{user_id}")
async def trigger_notification(user_id: str):
    if not brain:
        raise HTTPException(status_code=503)
    return await brain.predict_best_notification_time.remote(user_id)

@app.post("/v5/comment/analyze")
async def analyze_comment(req: CommentRequest):
    if not brain:
        raise HTTPException(status_code=503)
    await brain.community_psych.process_comment.remote(req.country, req.comment)
    metrics = await brain.community_psych.get_country_metrics.remote(req.country)
    return {"country": req.country, "metrics": metrics}

@app.get("/v5/stats")
async def stats():
    async with AsyncSessionLocal() as sess:
        from sqlalchemy import func
        result = await sess.execute(sa.select(func.count(UserEvent.id)))
        total_events = result.scalar()
    return {"total_events": total_events}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
