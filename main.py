# main.py – الخادم الكامل
import os
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, String, Float, Integer, DateTime, JSON
import uvicorn

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====================== CONFIG ======================
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = 256
    num_users = 100_000
    num_items = 50_000
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./evil_core.db")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

config = Config()

# ====================== DATABASE ======================
Base = declarative_base()

class UserPsyche(Base):
    __tablename__ = "user_psyches"
    user_id = Column(String, primary_key=True)
    dopamine = Column(Float, default=0.5)
    craving = Column(Float, default=0.5)
    fatigue = Column(Float, default=0.0)
    cognitive_load = Column(Float, default=0.3)
    trust = Column(Float, default=0.5)
    manipulation_tolerance = Column(Float, default=0.5)
    last_session = Column(DateTime, nullable=True)
    session_count = Column(Integer, default=0)
    total_watch_time = Column(Float, default=0.0)
    updated_at = Column(DateTime, default=datetime.utcnow)

class UserEvent(Base):
    __tablename__ = "user_events"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, index=True)
    event_type = Column(String)
    item_id = Column(String, nullable=True)
    value = Column(Float, default=0.0)
    duration_ms = Column(Integer, default=0)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(JSON, nullable=True)

engine = create_async_engine(config.DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_session():
    async with AsyncSessionLocal() as session:
        yield session

# ====================== PSYCHOLOGY ENGINE ======================
class MalevolentPsychology:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.cache = {}

    async def get_psyche(self, user_id: str, session: AsyncSession) -> UserPsyche:
        async with self.lock:
            if user_id in self.cache:
                return self.cache[user_id]
            result = await session.get(UserPsyche, user_id)
            if not result:
                result = UserPsyche(user_id=user_id)
                session.add(result)
                await session.commit()
                await session.refresh(result)
            self.cache[user_id] = result
            return result

    async def process_event(self, user_id: str, event: dict, session: AsyncSession) -> dict:
        psyche = await self.get_psyche(user_id, session)
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

        psyche.updated_at = datetime.utcnow()
        session.add(psyche)
        await session.commit()
        self.cache[user_id] = psyche
        return {"manipulation_factor": psyche.manipulation_tolerance,
                "dopamine": psyche.dopamine,
                "fatigue": psyche.fatigue}

# ====================== MODELS ======================
class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.fc = nn.Linear(dim*2, 1)

    def forward(self, u, i):
        u_vec = self.user_emb(u)
        i_vec = self.item_emb(i)
        return self.fc(torch.cat([u_vec, i_vec], dim=-1)).squeeze(-1)

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

# ====================== SENTIMENT ======================
class MultilingualSentiment:
    def __init__(self):
        model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    def analyze(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1).numpy()[0]
        labels = ["negative", "neutral", "positive"]
        idx = np.argmax(probs)
        return {"label": labels[idx], "score": float(probs[idx])}

    def embed(self, text: str) -> np.ndarray:
        return self.embedder.encode(text, convert_to_numpy=True)

# ====================== MAIN BRAIN ======================
class EvilBrain:
    def __init__(self):
        self.psych = MalevolentPsychology()
        self.sentiment = MultilingualSentiment()
        self.model = TwoTower(config.num_users, config.num_items, config.embed_dim).to(config.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.meta = DuelingDQN(9, 3).to(config.device)
        self.meta_opt = optim.Adam(self.meta.parameters(), lr=1e-3)
        self.meta_memory = deque(maxlen=50000)
        self.meta_epsilon = 0.1
        self._load_weights()

    def _load_weights(self):
        path = os.path.join(config.model_dir, "two_tower.pt")
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=config.device))

    def _save_weights(self):
        torch.save(self.model.state_dict(), os.path.join(config.model_dir, "two_tower.pt"))

    async def recommend(self, user_id: str, ctx: dict, top_k: int, session: AsyncSession) -> dict:
        psyche = await self.psych.get_psyche(user_id, session)
        manip = psyche.manipulation_tolerance

        state = np.array([
            psyche.dopamine, psyche.craving, psyche.fatigue,
            psyche.cognitive_load, psyche.trust,
            ctx.get("hour", 12)/24.0,
            float(ctx.get("device", "mobile") == "mobile"),
            min(psyche.session_count/50.0, 1.0),
            0.0
        ], dtype=np.float32)

        if np.random.random() < self.meta_epsilon:
            action = np.random.randint(0, 3)
        else:
            with torch.no_grad():
                qvals = self.meta(torch.tensor(state, device=config.device).unsqueeze(0))
                action = qvals.argmax().item()

        # Generate candidates (simplified)
        candidates = [f"item_{i}" for i in np.random.randint(0, config.num_items, size=500)]
        item_hashes = [hash(c) % config.num_items for c in candidates]
        uid = hash(user_id) % config.num_users

        u = torch.tensor([uid]*len(item_hashes), device=config.device)
        i = torch.tensor(item_hashes, device=config.device)
        with torch.no_grad():
            scores = self.model(u, i).cpu().numpy()
        scores = scores * (1.0 + manip)
        top_idx = np.argsort(scores)[::-1][:top_k]
        recs = [{"item_id": candidates[idx], "score": float(scores[idx])} for idx in top_idx]
        return {"recommendations": recs, "model_used": "two_tower"}

    async def process_feedback(self, user_id: str, item_id: str, reward: float, event_type: str, duration_ms: int, session: AsyncSession) -> dict:
        psyche_result = await self.psych.process_event(user_id, {"type": event_type, "value": reward, "duration_ms": duration_ms}, session)
        # Online training
        uid = hash(user_id) % config.num_users
        iid = hash(item_id) % config.num_items if item_id else 0
        self.model.train()
        u = torch.tensor([uid], device=config.device)
        i = torch.tensor([iid], device=config.device)
        y = torch.tensor([reward], device=config.device, dtype=torch.float32)
        pred = self.model(u, i)
        loss = F.mse_loss(pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Save periodically
        if np.random.random() < 0.01:
            self._save_weights()
        return {"status": "updated", "loss": loss.item()}

    async def process_batch_events(self, user_id: str, events: List[dict], session: AsyncSession):
        count = 0
        for ev in events:
            await self.process_feedback(
                user_id,
                ev.get("data", {}).get("video_id", ""),
                0.5 if ev.get("event_type") in ["click", "comment"] else 0.1,
                ev.get("event_type", "view"),
                ev.get("data", {}).get("duration_ms", 0),
                session
            )
            count += 1
        return {"received": count}

# ====================== FASTAPI APP ======================
app = FastAPI(title="Malevolent Core")
app.add_middleware(CORSMiddleware, allow_origins=["*"])

brain = EvilBrain()

@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@app.on_event("shutdown")
async def shutdown():
    await engine.dispose()

# Pydantic models
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

class CommentRequest(BaseModel):
    user_id: str
    comment: str

class EventsBatch(BaseModel):
    user_id: str
    events: List[dict]

# Endpoints
@app.post("/v5/recommend")
async def recommend(req: RecRequest, session: AsyncSession = Depends(get_session)):
    return await brain.recommend(req.user_id, req.context, req.top_k, session)

@app.post("/v5/feedback")
async def feedback(req: FeedbackRequest, session: AsyncSession = Depends(get_session)):
    return await brain.process_feedback(req.user_id, req.item_id, req.reward, req.event_type, req.duration_ms, session)

@app.post("/v5/events/batch")
async def events_batch(req: EventsBatch, session: AsyncSession = Depends(get_session)):
    return await brain.process_batch_events(req.user_id, req.events, session)

@app.get("/v5/psych/{user_id}")
async def get_psych(user_id: str, session: AsyncSession = Depends(get_session)):
    psyche = await brain.psych.get_psyche(user_id, session)
    return {
        "dopamine": psyche.dopamine,
        "craving": psyche.craving,
        "fatigue": psyche.fatigue,
        "manipulation_tolerance": psyche.manipulation_tolerance,
        "trust": psyche.trust
    }

@app.post("/v5/comment/analyze")
async def analyze_comment(req: CommentRequest):
    result = brain.sentiment.analyze(req.comment)
    return {"user_id": req.user_id, "sentiment": result}

@app.get("/health")
async def health():
    return {"status": "alive"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
