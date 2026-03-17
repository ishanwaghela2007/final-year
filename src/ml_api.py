from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import asyncio
import os
from pathlib import Path
import json

app = FastAPI()

# Optional: set PUBLISHED_URL or TUNNEL_URL to your localtunnel (or ngrok) URL for linking
TUNNEL_URL = os.environ.get("PUBLISHED_URL") or os.environ.get("TUNNEL_URL", "")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = Path(__file__).resolve().parent.parent / "data/inspections.db"

def get_stats():
    total_tubs = 0
    company_counts = {}
    
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Get total
            cursor.execute("SELECT COUNT(*) FROM inspections")
            total_tubs = cursor.fetchone()[0]
            
            # Get per-company counts
            cursor.execute("SELECT brand_name, COUNT(*) FROM inspections GROUP BY brand_name")
            rows = cursor.fetchall()
            for brand, count in rows:
                company_counts[brand] = count
                
            conn.close()
        except Exception as e:
            print("DB read error:", e)
            
    return {"total_tubs": total_tubs, "company_counts": company_counts}

@app.get("/api/info")
def api_info():
    """Public URL and endpoints info (link API with tunnel)."""
    return {
        "local_url": "http://localhost:8001",
        "tunnel_url": TUNNEL_URL or None,
        "docs": "http://localhost:8001/docs",
        "stats": "/api/stats",
        "ws_stats": "ws://localhost:8001/ws/stats",
    }

@app.get("/api/stats")
def read_stats():
    return get_stats()

@app.websocket("/ws/stats")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    last_count = -1
    try:
        while True:
            stats = get_stats()
            # Send an update if total items change or on initial connect
            if stats["total_tubs"] != last_count:
                await websocket.send_text(json.dumps(stats))
                last_count = stats["total_tubs"]
            await asyncio.sleep(1)
    except Exception as e:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
