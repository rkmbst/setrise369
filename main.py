# main.py – الخادم الكامل
import os, json, asyncio, logging
from datetime import datetime
from typing import Dict, List, Any
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker

import redis.asyncio as aioredis

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

# ========== الإعدادات ==========
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = 256
    num_users = 10_000_000
    num_items = 5_000_000
    model_dir = "./evil_models"
    os.makedirs(model_dir, exist_ok=True)
    redis_url = "redis://localhost:6379"
    db_url = "sqlite+aiosqlite:///./evil_core.db"
    kafka_brokers = "localhost:9092"

config = Config()

# ========== قاعدة البيانات ==========
Base = declarative_base()

class UserPsyche(Base):
    __tablename__ = "user_psyches"
    user_id = sa.Column(sa.String, primary_key=True)
    dopamine = sa.Column(sa.Float, default=0.5)
    craving = sa.Column(sa.Float, default=0.5)
    fatigue = sa.Column(sa.Float, default=0.0)
    cognitive_load = sa.Column(sa.Float, default=0.3)
    trust = sa.Column(sa.Float, default=0.5)
    manipulation_tolerance = sa.Column(sa.Float, default=0.5)
    last_session = sa.Column(sa.DateTime, nullable=True)
    session_count = sa.Column(sa.Integer, default=0)
    total_watch_time = sa.Column(sa.Float, default=0.0)
    updated_at = sa.Column(sa.DateTime, default=datetime.utcnow)

class UserEvent(Base):
    __tablename__ = "user_events"
    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    user_id = sa.Column(sa.String, index=True)
    event_type = sa.Column(sa.String)
    item_id = sa.Column(sa.String, nullable=True)
    value = sa.Column(sa.Float, default=0.0)
    duration_ms = sa.Column(sa.Integer, default=0)
    timestamp = sa.Column(sa.DateTime, default=datetime.utcnow)
    metadata_json = sa.Column(sa.JSON, nullable=True)

engine = create_async_engine(config.db_url, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# ========== المحرك النفسي ==========
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
        etype = event.get("type", "view")

        if val > 0:
            psyche.dopamine = min(1.0, psyche.dopamine + 0.2 * val)
        else:
            psyche.dopamine *= 0.98

        psyche.craving = max(0.0, 1.0 - psyche.dopamine - 0.1 * psyche.fatigue)
        psyche.fatigue += dur / 60000 * 0.03
        psyche.fatigue = min(1.0, psyche.fatigue)

        if etype == "impression":
            psyche.cognitive_load = min(1.0, psyche.cognitive_load + 0.03)
        elif etype == "click":
            psyche.cognitive_load = max(0.1, psyche.cognitive_load - 0.05)

        if val > 0.5:
            psyche.trust = min(1.0, psyche.trust + 0.02)
        else:
            psyche.trust = max(0.1, psyche.trust - 0.01)

        psyche.manipulation_tolerance = psyche.dopamine * psyche.trust * (1 - psyche.fatigue)

        if etype == "start_session":
            psyche.last_session = datetime.utcnow()
            psyche.session_count += 1
        elif etype == "end_session":
            psyche.total_watch_time += dur / 60000

        async with AsyncSessionLocal() as sess:
            sess.add(psyche)
            await sess.commit()
        self.cache[user_id] = psyche
        return {"manipulation_factor": psyche.manipulation_tolerance,
                "dopamine": psyche.dopamine, "fatigue": psyche.fatigue}

# ========== نماذج ==========
class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.fc = nn.Linear(dim*2, 1)
    def forward(self, u, i):
        return self.fc(torch.cat([self.user_emb(u), self.item_emb(i)], -1)).squeeze(-1)

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

# ========== تحليل المشاعر متعدد اللغات ==========
class MultilingualSentiment:
    def __init__(self):
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(config.device)
        self.model.eval()
        self.embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    def analyze(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(config.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        labels = ["negative", "neutral", "positive"]
        idx = np.argmax(probs)
        return {"label": labels[idx], "score": float(probs[idx])}

    def embed(self, text: str) -> np.ndarray:
        return self.embedder.encode(text, convert_to_numpy=True)

# ========== الدماغ الشرير ==========
class EvilBrain:
    def __init__(self):
        self.psych = MalevolentPsychology()
        self.models = {}
        self.optims = {}
        self.scaler = GradScaler()
        self.meta = DuelingDQN(9, 3).to(config.device)
        self.meta_opt = optim.Adam(self.meta.parameters(), lr=1e-3)
        self.meta_memory = deque(maxlen=50000)
        self.meta_epsilon = 0.1
        self.sentiment = MultilingualSentiment()
        self._init_models()

    def _init_models(self):
        self.models["two_tower"] = TwoTower(config.num_users, config.num_items, config.embed_dim).to(config.device)
        self.optims["two_tower"] = optim.Adam(self.models["two_tower"].parameters(), lr=1e-3)

    async def recommend(self, user_id: str, ctx: dict, top_k: int = 20) -> dict:
        psyche = await self.psych.get_psyche(user_id)
        manip = psyche.manipulation_tolerance
        state = np.array([psyche.dopamine, psyche.craving, psyche.fatigue,
                          psyche.cognitive_load, psyche.trust,
                          ctx.get("hour", 12)/24, float(ctx.get("device", "mobile")=="mobile"),
                          min(psyche.session_count/50, 1.0), 0.0], dtype=np.float32)
        with torch.no_grad():
            qvals = self.meta(torch.tensor(state, device=config.device).unsqueeze(0))
            action = qvals.argmax().item() if np.random.random() > self.meta_epsilon else np.random.randint(0, 3)
        candidates = [f"item_{i}" for i in np.random.randint(0, config.num_items, size=500)]
        item_hashes = [hash(c) % config.num_items for c in candidates]
        uid = hash(user_id) % config.num_users
        u = torch.tensor([uid]*len(item_hashes), device=config.device)
        i = torch.tensor(item_hashes, device=config.device)
        with torch.no_grad():
            scores = self.models["two_tower"](u, i).cpu().numpy()
        scores = scores * (1.0 + manip)
        top_idx = np.argsort(scores)[::-1][:top_k]
        recs = [{"item_id": candidates[idx], "score": float(scores[idx])} for idx in top_idx]
        return {"recommendations": recs, "model_used": "two_tower", "manipulation_active": manip > 0.6}

    async def process_feedback(self, user_id: str, item_id: str, reward: float, event_type: str, duration_ms: int = 0):
        psyche = await self.psych.process_event(user_id, {"type": event_type, "value": reward, "duration_ms": duration_ms})
        uid = hash(user_id) % config.num_users
        iid = hash(item_id) % config.num_items
        # تحديث النموذج
        self.models["two_tower"].train()
        u = torch.tensor([uid], device=config.device)
        i = torch.tensor([iid], device=config.device)
        y = torch.tensor([reward], device=config.device, dtype=torch.float32)
        pred = self.models["two_tower"](u, i)
        loss = F.mse_loss(pred, y)
        self.optims["two_tower"].zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optims["two_tower"])
        self.scaler.update()
        return {"status": "updated", "loss": loss.item()}

# ========== FastAPI ==========
app = FastAPI(title="Malevolent Core")
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

class RecRequest(BaseModel):
    user_id: str
    context: dict = {}
    top_k: int = 20

class FeedbackRequest(BaseModel):
    user_id: str
    item_id: str
    reward: float
    event_type: str = "click"
    duration_ms: int = 0

@app.post("/v5/recommend")
async def recommend(req: RecRequest):
    return await brain.recommend.remote(req.user_id, req.context, req.top_k)

@app.post("/v5/feedback")
async def feedback(req: FeedbackRequest):
    return await brain.process_feedback.remote(req.user_id, req.item_id, req.reward, req.event_type, req.duration_ms)

@app.post("/v5/events/batch")
async def events_batch(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    events = data.get("events", [])
    for ev in events:
        await brain.process_feedback.remote(
            user_id,
            ev.get("data", {}).get("video_id", ""),
            0.5 if ev.get("event_type") in ["click", "comment"] else 0.1,
            ev.get("event_type", "view"),
            ev.get("data", {}).get("duration_ms", 0)
        )
    return {"status": "ok", "received": len(events)}

@app.get("/v5/psych/{user_id}")
async def get_psych(user_id: str):
    p = await brain.psych.get_psyche.remote(user_id)
    return {"dopamine": p.dopamine, "craving": p.craving, "fatigue": p.fatigue, "manipulation": p.manipulation_tolerance}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
