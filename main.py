# main.py – خادم بسيط جاهز للنشر على Railway
from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, String, Float, Integer, DateTime, JSON
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./evil_core.db")

app = FastAPI()

class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        self.fc = nn.Linear(dim*2, 1)
    def forward(self, u, i):
        return self.fc(torch.cat([self.user_emb(u), self.item_emb(i)], -1)).squeeze(-1)

n_users = 1000
n_items = 1000
dim = 64
model = TwoTower(n_users, n_items, dim)
model.eval()

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
    user_hash = hash(req.user_id) % n_users
    candidates = [f"item_{i}" for i in range(100)]
    item_hashes = [hash(c) % n_items for c in candidates]
    u = torch.tensor([user_hash] * len(item_hashes))
    i = torch.tensor(item_hashes)
    with torch.no_grad():
        scores = model(u, i).numpy()
    top_idx = np.argsort(scores)[::-1][:req.top_k]
    recs = [{"item_id": candidates[idx], "score": float(scores[idx])} for idx in top_idx]
    return {"recommendations": recs}

@app.post("/v5/feedback")
async def feedback(req: FeedbackRequest):
    return {"status": "ok"}

@app.post("/v5/events/batch")
async def events_batch(request: Request):
    data = await request.json()
    events = data.get("events", [])
    print(f"Received {len(events)} events")
    return {"received": len(events)}
