"""
Smart Parking AI
Built by Mouad Waseem Syed

Real-time parking detection through your browser camera.
"""

from flask import Flask, jsonify, render_template_string, request
from collections import defaultdict
from datetime import datetime, timedelta
import cv2
import numpy as np
import json
import uuid
import random
import sqlite3
import threading
import os

# Detection model
try:
    from ultralytics import YOLO
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = "yolov8n.pt"
    detection_model = YOLO(MODEL_PATH)
    print(f"OK Detection model loaded")
    MODEL_AVAILABLE = True
except Exception as e:
    print(f"FAIL Model not available: {e}")
    MODEL_AVAILABLE = False

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPACES_FILE = os.path.join(BASE_DIR, "spaces.json")
DB_FILE = os.path.join(BASE_DIR, "parking_data.db")
CONF_THRESHOLD = 0.25
OVERLAP_THRESHOLD = 15

db_lock = threading.Lock()

def init_db():
    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS detections (
            id TEXT PRIMARY KEY, timestamp TEXT NOT NULL,
            available_count INTEGER, occupied_count INTEGER,
            total_count INTEGER, spaces_json TEXT, detection_count INTEGER)''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp)')
        conn.commit()
        conn.close()
        print(f"OK Database ready")

def save_detection(record):
    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO detections VALUES (?,?,?,?,?,?,?)''',
            (record['id'], record['timestamp'], record['available_count'],
             record['occupied_count'], record['total_count'],
             json.dumps(record['spaces']), record['detection_count']))
        conn.commit()
        conn.close()

def get_all_records():
    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('SELECT * FROM detections ORDER BY timestamp ASC')
        rows = c.fetchall()
        conn.close()
    return [{'id': r[0], 'timestamp': r[1], 'available_count': r[2],
             'occupied_count': r[3], 'total_count': r[4],
             'spaces': json.loads(r[5]) if r[5] else {}, 'detection_count': r[6]}
            for r in rows]

def bulk_save(records):
    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.executemany('''INSERT OR REPLACE INTO detections VALUES (?,?,?,?,?,?,?)''',
            [(r['id'], r['timestamp'], r['available_count'], r['occupied_count'],
              r['total_count'], json.dumps(r['spaces']), r['detection_count']) for r in records])
        conn.commit()
        conn.close()

def clear_db():
    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        conn.cursor().execute('DELETE FROM detections')
        conn.commit()
        conn.close()

init_db()

def load_spaces():
    try:
        with open(SPACES_FILE, 'r') as f:
            d = json.load(f)
            return d if d else default_spaces()
    except Exception:
        return default_spaces()

def default_spaces():
    return {"A1": [50,30,140,100], "A2": [200,30,140,100], "A3": [350,30,140,100],
            "B1": [50,160,140,100], "B2": [200,160,140,100], "B3": [350,160,140,100]}

def save_spaces_file(spaces):
    with open(SPACES_FILE, 'w') as f:
        json.dump(spaces, f, indent=2)

SPACES = load_spaces()
current_status = {sid: "available" for sid in SPACES}

def detect_from_frame(frame_bytes):
    global current_status
    if not MODEL_AVAILABLE:
        return {"error": "Detection model not loaded"}
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Could not decode frame"}
    fh, fw = frame.shape[:2]
    results = detection_model(frame, conf=CONF_THRESHOLD, verbose=False)
    detections = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append({"bbox": [int(x1), int(y1), int(x2), int(y2)],
                             "confidence": round(float(box.conf[0]), 2)})
    statuses = {}
    for sid, (sx, sy, sw, sh) in SPACES.items():
        occupied = False
        space_area = sw * sh
        for det in detections:
            bx1, by1, bx2, by2 = det["bbox"]
            ox1, oy1 = max(bx1, sx), max(by1, sy)
            ox2, oy2 = min(bx2, sx + sw), min(by2, sy + sh)
            if ox1 < ox2 and oy1 < oy2:
                overlap = (ox2 - ox1) * (oy2 - oy1)
                if space_area > 0 and (overlap / space_area * 100) >= OVERLAP_THRESHOLD:
                    occupied = True
                    break
        statuses[sid] = "occupied" if occupied else "available"
    current_status = statuses
    timestamp = datetime.utcnow().isoformat()
    available = sum(1 for s in statuses.values() if s == "available")
    occupied = sum(1 for s in statuses.values() if s == "occupied")
    record = {"id": str(uuid.uuid4()), "timestamp": timestamp,
              "available_count": available, "occupied_count": occupied,
              "total_count": available + occupied, "spaces": statuses,
              "detection_count": len(detections)}
    save_detection(record)
    return {"status": statuses, "available": available, "occupied": occupied,
            "total": available + occupied, "detections": detections,
            "detection_count": len(detections), "timestamp": timestamp,
            "frame_width": fw, "frame_height": fh, "spaces_config": SPACES}

def generate_demo_data(days=30):
    records = []
    now = datetime.utcnow()
    space_ids = list(SPACES.keys()) if SPACES else ['A1','A2','A3','B1','B2','B3']
    for d in range(days, 0, -1):
        date = now - timedelta(days=d)
        for h in range(24):
            for m in [0, 15, 30, 45]:
                ts = date.replace(hour=h, minute=m, second=0, microsecond=0)
                weekday = ts.weekday()
                if weekday < 5:
                    base = 0.7 if 7 <= h < 9 else (0.85 if 9 <= h < 17 else (0.6 if 17 <= h < 19 else (0.3 if 19 <= h < 22 else 0.1)))
                else:
                    base = 0.55 if 10 <= h < 20 else 0.15
                occ_rate = max(0, min(1, base + random.uniform(-0.15, 0.15)))
                statuses = {sid: ("occupied" if random.random() < occ_rate else "available") for sid in space_ids}
                occ = sum(1 for s in statuses.values() if s == "occupied")
                records.append({"id": str(uuid.uuid4()), "timestamp": ts.isoformat(),
                              "available_count": len(statuses) - occ, "occupied_count": occ,
                              "total_count": len(statuses), "spaces": statuses, "detection_count": occ})
    bulk_save(records)
    return len(records)

def hourly_average(records):
    buckets = defaultdict(list)
    for r in records:
        try:
            ts = datetime.fromisoformat(r['timestamp'].replace('Z', ''))
            total = r['occupied_count'] + r['available_count']
            if total > 0: buckets[ts.hour].append(r['occupied_count'] / total * 100)
        except Exception: continue
    return {h: round(sum(v)/len(v), 1) if v else 0 for h, v in buckets.items()}

def daily_average(records):
    buckets = defaultdict(list)
    days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    for r in records:
        try:
            ts = datetime.fromisoformat(r['timestamp'].replace('Z', ''))
            total = r['occupied_count'] + r['available_count']
            if total > 0: buckets[ts.weekday()].append(r['occupied_count'] / total * 100)
        except Exception: continue
    return {days[d]: round(sum(v)/len(v), 1) if v else 0 for d, v in buckets.items()}

def heatmap_data(records):
    days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    grid = [[None] * 24 for _ in range(7)]
    buckets = defaultdict(list)
    for r in records:
        try:
            ts = datetime.fromisoformat(r['timestamp'].replace('Z', ''))
            total = r['occupied_count'] + r['available_count']
            if total > 0: buckets[(ts.weekday(), ts.hour)].append(r['occupied_count'] / total * 100)
        except Exception: continue
    for (d, h), vals in buckets.items():
        grid[d][h] = round(sum(vals) / len(vals), 1)
    return {"days": days, "grid": grid}

def predict_next_hour(records):
    if not records:
        return {"predicted_percent": None, "confidence": "low", "based_on_samples": 0}
    next_hour = (datetime.utcnow() + timedelta(hours=1)).hour
    samples = []
    for r in records:
        try:
            ts = datetime.fromisoformat(r['timestamp'].replace('Z', ''))
            if ts.hour == next_hour:
                total = r['occupied_count'] + r['available_count']
                if total > 0: samples.append(r['occupied_count'] / total * 100)
        except Exception: continue
    if not samples:
        return {"predicted_percent": None, "confidence": "no_data", "based_on_samples": 0}
    avg = sum(samples) / len(samples)
    confidence = "high" if len(samples) > 20 else ("medium" if len(samples) > 5 else "low")
    return {"predicted_percent": round(avg, 1), "confidence": confidence,
            "based_on_samples": len(samples), "for_hour": next_hour}

def peak_times(records, top_n=3):
    hourly = hourly_average(records)
    sorted_h = sorted(hourly.items(), key=lambda x: x[1], reverse=True)
    return [{"hour": h, "occupancy_percent": p} for h, p in sorted_h[:top_n]]

def per_space_usage(records):
    counts, totals = {}, {}
    for r in records:
        for sid, status in r.get('spaces', {}).items():
            totals[sid] = totals.get(sid, 0) + 1
            if status == 'occupied': counts[sid] = counts.get(sid, 0) + 1
    return {sid: round(counts.get(sid, 0) / total * 100, 1) if total > 0 else 0
            for sid, total in totals.items()}

def space_priority_score(sid):
    if not sid: return 999
    row = {'A':0,'B':100,'C':200}.get(sid[0].upper(), 300)
    try: num = int(sid[1:])
    except Exception: num = 99
    return row + num

def recommend_next_space(latest_status, usage_stats):
    if not latest_status:
        return {"recommended": None, "reason": "No live data"}
    free = [sid for sid, st in latest_status.items() if st == 'available']
    if not free:
        return {"recommended": None, "reason": "All spaces occupied"}
    if len(free) == len(latest_status):
        sorted_p = sorted(free, key=space_priority_score)
        return {"recommended": sorted_p[0], "reason": "Lot empty - closest to entrance",
                "alternatives": sorted_p[1:3]}
    def score(sid):
        return (usage_stats.get(sid, 50) * 0.6) + (space_priority_score(sid) * 0.4 / 10)
    sorted_s = sorted(free, key=score)
    best = sorted_s[0]
    return {"recommended": best,
            "reason": f"{usage_stats.get(best, 0)}% historical usage, near entrance",
            "alternatives": sorted_s[1:3]}



DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Smart Parking AI</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box;}
:root{
  --bg:#0B0F19; --surface:#131825; --surface-2:#1A2035;
  --text:#E8ECF4; --text-2:#8892A8; --text-3:#5A6478;
  --line:#1E2740; --line-soft:#161D30;
  --accent:#6C5CE7; --accent-soft:rgba(108,92,231,.12);
  --good:#00B894; --good-soft:rgba(0,184,148,.12);
  --bad:#FF6B6B; --bad-soft:rgba(255,107,107,.12);
  --warn:#FDCB6E; --warn-soft:rgba(253,203,110,.12);
  --radius:14px;
}
body{font-family:'DM Sans',system-ui,sans-serif;background:var(--bg);color:var(--text);font-size:14px;line-height:1.5;min-height:100vh;}
.nav{background:var(--surface);border-bottom:1px solid var(--line);padding:12px 20px;position:sticky;top:0;z-index:100;}
.nav-inner{max-width:1200px;margin:0 auto;display:flex;align-items:center;gap:16px;flex-wrap:wrap;}
.brand{display:flex;align-items:center;gap:10px;font-weight:700;font-size:16px;letter-spacing:-.3px;}
.brand-dot{width:10px;height:10px;border-radius:50%;background:var(--accent);box-shadow:0 0 12px var(--accent);}
.nav-tabs{display:flex;gap:4px;margin-left:auto;}
.nav-tab{padding:6px 14px;font-size:13px;color:var(--text-2);background:transparent;border:1px solid transparent;border-radius:8px;cursor:pointer;font-family:inherit;font-weight:500;}
.nav-tab:hover{background:var(--surface-2);color:var(--text);}
.nav-tab.active{background:var(--accent-soft);color:var(--accent);border-color:var(--accent);}
.container{max-width:1200px;margin:0 auto;padding:24px 20px;}
.page{display:none;}
.page.active{display:block;}
.page-head{margin-bottom:24px;}
.page-title{font-size:22px;font-weight:700;letter-spacing:-.4px;}
.page-sub{color:var(--text-2);font-size:13px;margin-top:4px;}
.hero{background:linear-gradient(135deg,var(--accent-soft),rgba(108,92,231,.04));border:1px solid var(--accent);border-radius:var(--radius);padding:24px;margin-bottom:20px;display:flex;gap:20px;align-items:center;}
.hero-icon{width:60px;height:60px;flex-shrink:0;background:var(--accent);color:#fff;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:28px;}
.hero-text h2{font-size:18px;font-weight:700;margin-bottom:4px;}
.hero-text p{color:var(--text-2);font-size:13px;}
.hero-close{margin-left:auto;background:transparent;border:1px solid var(--line);color:var(--text-3);padding:6px 12px;border-radius:8px;cursor:pointer;font-size:12px;font-family:inherit;}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:20px;}
@media(max-width:768px){.steps{grid-template-columns:1fr;}}
.step{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);padding:16px;display:flex;gap:14px;align-items:flex-start;}
.step-num{width:32px;height:32px;flex-shrink:0;background:var(--accent-soft);color:var(--accent);border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:14px;}
.step-text h3{font-size:13px;font-weight:600;margin-bottom:2px;}
.step-text p{color:var(--text-2);font-size:12px;line-height:1.5;}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;}
.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:16px;}
@media(max-width:768px){.grid-2,.grid-3{grid-template-columns:1fr;}}
.card{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);padding:18px;}
.card-title{font-size:11px;font-weight:600;color:var(--text-3);text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;}
.stat{text-align:center;padding:20px 12px;}
.stat-num{font-family:'JetBrains Mono',monospace;font-size:36px;font-weight:700;line-height:1;letter-spacing:-1.5px;}
.stat-label{font-size:11px;color:var(--text-3);margin-top:6px;text-transform:uppercase;letter-spacing:.5px;}
.stat.good .stat-num{color:var(--good);}
.stat.bad .stat-num{color:var(--bad);}
.stat.accent .stat-num{color:var(--accent);}
.camera-container{position:relative;background:#000;border-radius:10px;overflow:hidden;aspect-ratio:16/9;}
.camera-container video{width:100%;height:100%;object-fit:contain;display:block;}
.camera-overlay{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;}
.camera-badge{position:absolute;top:12px;left:12px;display:flex;align-items:center;gap:6px;padding:4px 10px;background:rgba(0,0,0,.7);border-radius:20px;font-size:11px;color:#fff;font-weight:500;z-index:2;}
.camera-badge .dot{width:6px;height:6px;border-radius:50%;background:var(--bad);}
.camera-badge.live .dot{background:var(--good);box-shadow:0 0 6px var(--good);animation:pulse 1.4s ease infinite;}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.4;}}
.camera-empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--text-3);font-size:13px;flex-direction:column;gap:8px;}
.camera-empty-icon{font-size:32px;opacity:.4;}
.btn{display:inline-flex;align-items:center;gap:6px;padding:10px 18px;border-radius:10px;font-family:inherit;font-size:13px;font-weight:600;cursor:pointer;border:none;transition:all .15s ease;}
.btn-primary{background:var(--accent);color:#fff;}
.btn-primary:hover{filter:brightness(1.15);}
.btn-danger{background:var(--bad);color:#fff;}
.btn-outline{background:transparent;border:1px solid var(--line);color:var(--text-2);}
.btn-outline:hover{border-color:var(--text-3);color:var(--text);}
.btn-group{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px;}
.spaces{display:grid;grid-template-columns:repeat(auto-fit,minmax(90px,1fr));gap:10px;}
.space{padding:16px 8px;border-radius:10px;text-align:center;border:2px solid;}
.space.free{background:var(--good-soft);border-color:var(--good);}
.space.used{background:var(--bad-soft);border-color:var(--bad);}
.space-id{font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700;}
.space.free .space-id{color:var(--good);}
.space.used .space-id{color:var(--bad);}
.space-st{font-size:10px;color:var(--text-3);margin-top:2px;text-transform:uppercase;}
.recommend{display:none;align-items:center;gap:14px;padding:16px;background:var(--accent-soft);border:1px solid var(--accent);border-radius:var(--radius);margin-bottom:16px;}
.recommend.show{display:flex;}
.recommend-icon{width:44px;height:44px;border-radius:10px;background:var(--accent);color:#fff;display:flex;align-items:center;justify-content:center;font-size:20px;font-weight:700;flex-shrink:0;}
.recommend-pick{font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700;color:var(--accent);}
.recommend-why{font-size:12px;color:var(--text-2);margin-top:2px;}
.events{display:flex;flex-direction:column;max-height:200px;overflow-y:auto;}
.event{display:flex;gap:12px;padding:8px 0;border-bottom:1px solid var(--line-soft);font-size:12px;}
.event:last-child{border-bottom:none;}
.event-time{font-family:'JetBrains Mono',monospace;color:var(--text-3);width:65px;flex-shrink:0;}
.event-msg{color:var(--text-2);}
.events-empty{text-align:center;padding:20px;color:var(--text-3);font-size:12px;}
.heatmap{display:grid;grid-template-columns:40px repeat(24,1fr);gap:2px;font-size:9px;}
.hm-h,.hm-d{display:flex;align-items:center;justify-content:center;color:var(--text-3);font-family:'JetBrains Mono',monospace;}
.hm-d{justify-content:flex-end;padding-right:6px;}
.hm-c{aspect-ratio:1;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:8px;font-weight:600;color:#fff;font-family:'JetBrains Mono',monospace;}
.calib-steps{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:20px;}
@media(max-width:768px){.calib-steps{grid-template-columns:1fr 1fr;}}
.calib-step{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px;text-align:center;}
.calib-step.active{border-color:var(--accent);background:var(--accent-soft);}
.calib-step-num{font-family:'JetBrains Mono',monospace;font-size:24px;font-weight:700;color:var(--text-3);margin-bottom:4px;}
.calib-step.active .calib-step-num{color:var(--accent);}
.calib-step-label{font-size:11px;color:var(--text-2);text-transform:uppercase;letter-spacing:.5px;}
#calibrate-canvas{border:1px solid var(--line);border-radius:8px;cursor:crosshair;max-width:100%;background:#000;}
.space-list{margin-top:12px;}
.space-item{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid var(--line-soft);font-size:13px;}
.space-item button{background:var(--bad-soft);border:1px solid var(--bad);color:var(--bad);padding:4px 10px;border-radius:6px;cursor:pointer;font-size:11px;font-family:inherit;}
.toast{position:fixed;bottom:20px;right:20px;padding:12px 18px;background:var(--surface);border:1px solid var(--good);color:var(--good);border-radius:10px;font-size:13px;font-weight:500;z-index:200;opacity:0;transform:translateY(20px);transition:all .3s ease;}
.toast.show{opacity:1;transform:translateY(0);}
.footer{text-align:center;padding:32px 0;font-size:11px;color:var(--text-3);border-top:1px solid var(--line);margin-top:32px;}
canvas{max-height:250px;}
.hidden{display:none;}
.warning-banner{background:var(--warn-soft);border:1px solid var(--warn);color:var(--warn);padding:12px 16px;border-radius:10px;font-size:13px;margin-bottom:16px;display:none;align-items:center;gap:10px;}
.warning-banner.show{display:flex;}
</style>
</head>
<body>

<div class="nav">
  <div class="nav-inner">
    <div class="brand"><div class="brand-dot"></div>Smart Parking AI</div>
    <div class="nav-tabs">
      <button class="nav-tab active" onclick="showPage('live', this)">Live</button>
      <button class="nav-tab" onclick="showPage('analytics', this)">Analytics</button>
      <button class="nav-tab" onclick="showPage('calibrate', this)">Calibrate</button>
    </div>
  </div>
</div>

<div class="container">

<div id="page-live" class="page active">
  <div class="hero" id="hero">
    <div class="hero-icon">&#x1F44B;</div>
    <div class="hero-text">
      <h2>Welcome to Smart Parking AI</h2>
      <p>Real-time parking detection through your camera. Point it at a parking area and watch spaces light up as cars come and go.</p>
    </div>
    <button class="hero-close" onclick="dismissHero()">Got it</button>
  </div>

  <div class="steps" id="steps">
    <div class="step">
      <div class="step-num">1</div>
      <div class="step-text"><h3>Start your camera</h3><p>Click Start Camera below. Allow access when your browser asks.</p></div>
    </div>
    <div class="step">
      <div class="step-num">2</div>
      <div class="step-text"><h3>Calibrate spaces</h3><p>Go to Calibrate tab and draw boxes where parking spots are.</p></div>
    </div>
    <div class="step">
      <div class="step-num">3</div>
      <div class="step-text"><h3>Watch it work</h3><p>Spaces turn red when occupied, green when free. Updates every 2 seconds.</p></div>
    </div>
  </div>

  <div class="page-head">
    <div class="page-title">Live Detection</div>
    <div class="page-sub">Real-time parking space monitoring</div>
  </div>

  <div class="grid-3">
    <div class="card stat good"><div class="stat-num" id="kpi-available">&mdash;</div><div class="stat-label">Available</div></div>
    <div class="card stat bad"><div class="stat-num" id="kpi-occupied">&mdash;</div><div class="stat-label">Occupied</div></div>
    <div class="card stat accent"><div class="stat-num" id="kpi-time">&mdash;</div><div class="stat-label">Last Update</div></div>
  </div>

  <div class="recommend" id="rec">
    <div class="recommend-icon">&rarr;</div>
    <div>
      <div style="font-size:11px;color:var(--text-3);text-transform:uppercase;letter-spacing:.5px;">Recommended space</div>
      <div class="recommend-pick" id="rec-pick">&mdash;</div>
      <div class="recommend-why" id="rec-why">&mdash;</div>
    </div>
  </div>

  <div class="grid-2">
    <div class="card">
      <div class="card-title">Camera Feed</div>
      <div class="camera-container" id="camera-box">
        <video id="video" autoplay playsinline muted></video>
        <canvas class="camera-overlay" id="overlay"></canvas>
        <canvas id="snap-canvas" style="display:none;"></canvas>
        <div class="camera-badge" id="cam-badge">
          <div class="dot"></div>
          <span id="cam-status">Camera off</span>
        </div>
        <div class="camera-empty" id="cam-empty">
          <div class="camera-empty-icon">&#x1F4F7;</div>
          <div>Click <strong>Start Camera</strong> below</div>
        </div>
      </div>
      <div class="btn-group">
        <button class="btn btn-primary" id="btn-start" onclick="startCamera()">&#9654; Start Camera</button>
        <button class="btn btn-danger hidden" id="btn-stop" onclick="stopCamera()">&#9632; Stop</button>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Parking Spaces</div>
      <div class="spaces" id="spaces-grid"><div class="events-empty">Start camera to begin</div></div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Recent Activity</div>
    <div class="events" id="event-list"><div class="events-empty">No detections yet</div></div>
  </div>
</div>

<div id="page-analytics" class="page">
  <div class="page-head">
    <div class="page-title">Analytics</div>
    <div class="page-sub">Patterns and predictions from detection history</div>
  </div>
  <div class="warning-banner" id="no-data-warn">
    <span>&#8505;</span>
    <div style="flex:1;"><strong>Not much data yet.</strong> Load demo data to see how analytics look.</div>
    <button class="btn btn-primary" onclick="loadDemoData()" style="padding:6px 12px;font-size:12px;">Load Demo Data</button>
  </div>
  <div class="grid-2">
    <div class="card stat accent">
      <div class="stat-num" id="pred-value">&mdash;</div>
      <div class="stat-label">Next Hour Prediction</div>
      <div style="font-size:11px;color:var(--text-3);margin-top:4px;" id="pred-info"></div>
    </div>
    <div class="card stat">
      <div class="stat-num" id="records-count">0</div>
      <div class="stat-label">Records Collected</div>
    </div>
  </div>
  <div class="card" style="margin-bottom:16px;">
    <div class="card-title">Occupancy by Hour of Day</div>
    <canvas id="hourly-chart"></canvas>
  </div>
  <div class="grid-2">
    <div class="card">
      <div class="card-title">By Day of Week</div>
      <canvas id="daily-chart"></canvas>
    </div>
    <div class="card">
      <div class="card-title">Busiest Hours</div>
      <div id="peaks-list" style="padding:8px 0;"><div class="events-empty">Need more data</div></div>
    </div>
  </div>
  <div class="card">
    <div class="card-title">Weekly Heatmap</div>
    <div id="heatmap-box" style="overflow-x:auto;"></div>
    <p style="font-size:10px;color:var(--text-3);margin-top:8px;text-align:center;">Darker = busier</p>
  </div>
</div>

<div id="page-calibrate" class="page">
  <div class="page-head">
    <div class="page-title">Calibrate Parking Spaces</div>
    <div class="page-sub">Mark where each parking spot is so the system knows what to watch</div>
  </div>
  <div class="calib-steps">
    <div class="calib-step active"><div class="calib-step-num">1</div><div class="calib-step-label">Start camera</div></div>
    <div class="calib-step"><div class="calib-step-num">2</div><div class="calib-step-label">Capture snapshot</div></div>
    <div class="calib-step"><div class="calib-step-num">3</div><div class="calib-step-label">Draw spaces</div></div>
    <div class="calib-step"><div class="calib-step-num">4</div><div class="calib-step-label">Save</div></div>
  </div>
  <div class="card" style="margin-bottom:16px;">
    <div class="card-title">How to Calibrate</div>
    <p style="font-size:13px;color:var(--text-2);line-height:1.8;">
      <strong style="color:var(--text);">Step 1:</strong> Go to Live tab and start your camera.<br>
      <strong style="color:var(--text);">Step 2:</strong> Come back here and click Capture Snapshot.<br>
      <strong style="color:var(--text);">Step 3:</strong> Click and drag to draw each parking space. Name them A1, A2, B1, etc.<br>
      <strong style="color:var(--text);">Step 4:</strong> Click Save Spaces when done.
    </p>
  </div>
  <div class="card" style="margin-bottom:16px;">
    <div class="card-title">Drawing Area</div>
    <div class="btn-group" style="margin:0 0 12px 0;">
      <button class="btn btn-primary" onclick="captureForCalibrate()">&#x1F4F7; Capture Snapshot</button>
      <button class="btn btn-outline" onclick="loadDefaultLayout()">Use Default Layout</button>
    </div>
    <p style="font-size:12px;color:var(--text-3);margin-bottom:10px;">Click and drag to draw a rectangle around each parking space.</p>
    <canvas id="calibrate-canvas" width="640" height="480"></canvas>
    <div class="btn-group">
      <button class="btn btn-primary" onclick="saveSpaces()">&#x1F4BE; Save Spaces</button>
      <button class="btn btn-outline" onclick="clearSpaces()">&#x1F5D1; Clear All</button>
    </div>
  </div>
  <div class="card">
    <div class="card-title">Defined Spaces</div>
    <div class="space-list" id="space-list"><div class="events-empty">No spaces defined yet</div></div>
  </div>
</div>

<div class="footer">Built by Mouad Waseem Syed</div>
<div class="toast" id="toast"></div>
</div>

<script>
let stream=null,detecting=false,detectInterval=null;
let hourlyChart=null,dailyChart=null;
let calibSpaces={},drawStart=null,isDrawing=false,snapshotImg=null;
let lastDetections=[],lastSpacesConfig={},lastStatuses={};

function toast(msg,isError){
  const t=document.getElementById('toast');
  t.textContent=msg;
  t.style.borderColor=isError?'var(--bad)':'var(--good)';
  t.style.color=isError?'var(--bad)':'var(--good)';
  t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'),3000);
}

function dismissHero(){
  document.getElementById('hero').style.display='none';
  document.getElementById('steps').style.display='none';
  localStorage.setItem('hero_dismissed','1');
}
if(localStorage.getItem('hero_dismissed')==='1'){
  document.getElementById('hero').style.display='none';
  document.getElementById('steps').style.display='none';
}

function showPage(name,el){
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('page-'+name).classList.add('active');
  if(el) el.classList.add('active');
  if(name==='analytics') loadAnalytics();
  if(name==='calibrate') loadCalibSpaces();
}

async function startCamera(){
  try{
    stream=await navigator.mediaDevices.getUserMedia({
      video:{facingMode:'environment',width:{ideal:960},height:{ideal:540}}
    });
    const video=document.getElementById('video');
    video.srcObject=stream;
    video.addEventListener('loadedmetadata',()=>{resizeOverlay();drawOverlay();});
    document.getElementById('btn-start').classList.add('hidden');
    document.getElementById('btn-stop').classList.remove('hidden');
    document.getElementById('cam-badge').classList.add('live');
    document.getElementById('cam-status').textContent='Live';
    document.getElementById('cam-empty').style.display='none';
    detecting=true;
    detectInterval=setInterval(detectFrame,2000);
    setTimeout(detectFrame,500);
    toast('Camera started');
  }catch(err){
    toast('Camera access denied',true);
    console.error(err);
  }
}

function stopCamera(){
  detecting=false;
  if(detectInterval) clearInterval(detectInterval);
  if(stream) stream.getTracks().forEach(t=>t.stop());
  stream=null;
  document.getElementById('video').srcObject=null;
  document.getElementById('btn-start').classList.remove('hidden');
  document.getElementById('btn-stop').classList.add('hidden');
  document.getElementById('cam-badge').classList.remove('live');
  document.getElementById('cam-status').textContent='Camera off';
  document.getElementById('cam-empty').style.display='flex';
  clearOverlay();
}

function resizeOverlay(){
  const video=document.getElementById('video');
  const overlay=document.getElementById('overlay');
  overlay.width=video.videoWidth||640;
  overlay.height=video.videoHeight||480;
}

function clearOverlay(){
  const overlay=document.getElementById('overlay');
  overlay.getContext('2d').clearRect(0,0,overlay.width,overlay.height);
}

function drawOverlay(){
  const overlay=document.getElementById('overlay');
  const ctx=overlay.getContext('2d');
  ctx.clearRect(0,0,overlay.width,overlay.height);
  for(const [sid,[x,y,w,h]] of Object.entries(lastSpacesConfig)){
    const status=lastStatuses[sid]||'available';
    const isFree=status==='available';
    const color=isFree?'#00B894':'#FF6B6B';
    const fillColor=isFree?'rgba(0,184,148,.15)':'rgba(255,107,107,.15)';
    ctx.fillStyle=fillColor;
    ctx.fillRect(x,y,w,h);
    ctx.strokeStyle=color;
    ctx.lineWidth=3;
    ctx.strokeRect(x,y,w,h);
    ctx.fillStyle=color;
    ctx.fillRect(x,y-24,56,24);
    ctx.fillStyle='#fff';
    ctx.font='bold 13px DM Sans, sans-serif';
    ctx.textAlign='left';
    ctx.fillText(sid,x+6,y-7);
    ctx.fillStyle=color;
    ctx.font='bold 11px DM Sans, sans-serif';
    ctx.fillText(isFree?'FREE':'OCCUPIED',x+6,y+h-8);
  }
  ctx.strokeStyle='rgba(108,92,231,.6)';
  ctx.lineWidth=1.5;
  for(const det of lastDetections){
    const [bx1,by1,bx2,by2]=det.bbox;
    ctx.strokeRect(bx1,by1,bx2-bx1,by2-by1);
  }
}

async function detectFrame(){
  if(!detecting||!stream) return;
  const video=document.getElementById('video');
  const canvas=document.getElementById('snap-canvas');
  canvas.width=video.videoWidth||640;
  canvas.height=video.videoHeight||480;
  canvas.getContext('2d').drawImage(video,0,0);
  canvas.toBlob(async(blob)=>{
    if(!blob) return;
    try{
      const formData=new FormData();
      formData.append('frame',blob,'frame.jpg');
      const resp=await fetch('/api/detect',{method:'POST',body:formData});
      const data=await resp.json();
      if(data.error){console.warn(data.error);return;}
      document.getElementById('kpi-available').textContent=data.available;
      document.getElementById('kpi-occupied').textContent=data.occupied;
      document.getElementById('kpi-time').textContent=new Date(data.timestamp).toLocaleTimeString();
      const grid=document.getElementById('spaces-grid');
      grid.innerHTML='';
      for(const [id,st] of Object.entries(data.status)){
        const ok=st==='available';
        grid.innerHTML+=`<div class="space ${ok?'free':'used'}"><div class="space-id">${id}</div><div class="space-st">${ok?'free':'occupied'}</div></div>`;
      }
      lastDetections=data.detections||[];
      lastSpacesConfig=data.spaces_config||{};
      lastStatuses=data.status||{};
      resizeOverlay();
      drawOverlay();
      updateRecommendation();
      addEvent(data);
    }catch(err){console.error(err);}
  },'image/jpeg',0.8);
}

function updateRecommendation(){
  fetch('/api/recommend').then(r=>r.json()).then(d=>{
    const rec=d.recommendation;
    const box=document.getElementById('rec');
    if(rec&&rec.recommended){
      box.classList.add('show');
      document.getElementById('rec-pick').textContent=rec.recommended;
      document.getElementById('rec-why').textContent=rec.reason;
    }else{box.classList.remove('show');}
  }).catch(()=>{});
}

function addEvent(data){
  const list=document.getElementById('event-list');
  const t=new Date(data.timestamp).toLocaleTimeString();
  const msg=`${data.available} free, ${data.occupied} occupied, ${data.detection_count} vehicles seen`;
  const el=document.createElement('div');
  el.className='event';
  el.innerHTML=`<span class="event-time">${t}</span><span class="event-msg">${msg}</span>`;
  if(list.querySelector('.events-empty')) list.innerHTML='';
  list.insertBefore(el,list.firstChild);
  while(list.children.length>20) list.removeChild(list.lastChild);
}

window.addEventListener('resize',()=>{if(stream){resizeOverlay();drawOverlay();}});

function colorFor(pct){
  if(pct===null||pct===undefined) return 'var(--surface-2)';
  const i=pct/100;
  const r=Math.round(255*i+20*(1-i));
  const g=Math.round(107*i+184*(1-i));
  const b=Math.round(107*i+148*(1-i));
  return `rgb(${r},${g},${b})`;
}

function chartOpts(){
  return {
    responsive:true,maintainAspectRatio:false,
    scales:{
      y:{beginAtZero:true,max:100,ticks:{color:'#5A6478',font:{size:10}},grid:{color:'#1E2740'}},
      x:{ticks:{color:'#5A6478',font:{size:10}},grid:{display:false}}
    },
    plugins:{legend:{display:false}}
  };
}

function loadAnalytics(){
  fetch('/api/analytics/summary').then(r=>r.json()).then(d=>{
    document.getElementById('records-count').textContent=d.total_records.toLocaleString();
    document.getElementById('no-data-warn').classList.toggle('show',d.total_records<10);
    const p=d.prediction;
    document.getElementById('pred-value').textContent=p.predicted_percent!==null?p.predicted_percent+'%':'—';
    document.getElementById('pred-info').textContent=p.predicted_percent!==null?`For ${p.for_hour}:00, ${p.confidence} confidence`:'Need more data';
    const hours=Array.from({length:24},(_,i)=>i);
    const hVals=hours.map(h=>d.hourly[h]||0);
    if(hourlyChart) hourlyChart.destroy();
    hourlyChart=new Chart(document.getElementById('hourly-chart'),{
      type:'bar',
      data:{labels:hours.map(h=>h+':00'),datasets:[{data:hVals,backgroundColor:hVals.map(v=>colorFor(v)),borderRadius:4}]},
      options:chartOpts()
    });
    const dL=Object.keys(d.daily),dV=Object.values(d.daily);
    if(dailyChart) dailyChart.destroy();
    dailyChart=new Chart(document.getElementById('daily-chart'),{
      type:'bar',
      data:{labels:dL,datasets:[{data:dV,backgroundColor:dV.map(v=>colorFor(v)),borderRadius:4}]},
      options:chartOpts()
    });
    const pE=document.getElementById('peaks-list');
    pE.innerHTML=d.peaks.length?d.peaks.map((p,i)=>
      `<div class="event"><span class="event-time">#${i+1}</span><span class="event-msg" style="font-family:'JetBrains Mono',monospace;">${p.hour}:00, ${p.occupancy_percent}%</span></div>`
    ).join(''):'<div class="events-empty">Need more data</div>';
    const hm=d.heatmap;
    let html='<div class="heatmap"><div></div>';
    for(let h=0;h<24;h++) html+=`<div class="hm-h">${h}</div>`;
    hm.days.forEach((day,di)=>{
      html+=`<div class="hm-d">${day}</div>`;
      for(let h=0;h<24;h++){
        const v=hm.grid[di][h];
        const bg=v!==null?colorFor(v):'var(--surface-2)';
        const tx=v!==null?Math.round(v):'';
        const col=v>50?'#fff':'var(--text-2)';
        html+=`<div class="hm-c" style="background:${bg};color:${col}" title="${day} ${h}:00">${tx}</div>`;
      }
    });
    html+='</div>';
    document.getElementById('heatmap-box').innerHTML=html;
  }).catch(err=>console.error(err));
}

function loadDemoData(){
  if(!confirm('Load 30 days of demo data to see analytics?')) return;
  toast('Loading demo data...');
  fetch('/api/demo-data',{method:'POST'}).then(r=>r.json()).then(d=>{
    toast(`Loaded ${d.added} records`);
    loadAnalytics();
  }).catch(err=>toast('Failed',true));
}

function captureForCalibrate(){
  const video=document.getElementById('video');
  if(!video.srcObject){toast('Start camera on Live page first',true);return;}
  const canvas=document.getElementById('calibrate-canvas');
  const ctx=canvas.getContext('2d');
  canvas.width=video.videoWidth||640;
  canvas.height=video.videoHeight||480;
  ctx.drawImage(video,0,0,canvas.width,canvas.height);
  snapshotImg=ctx.getImageData(0,0,canvas.width,canvas.height);
  drawAllSpaces();
  toast('Snapshot captured');
}

function loadDefaultLayout(){
  calibSpaces={"A1":[50,30,140,100],"A2":[200,30,140,100],"A3":[350,30,140,100],
               "B1":[50,160,140,100],"B2":[200,160,140,100],"B3":[350,160,140,100]};
  drawAllSpaces();
  renderSpaceList();
  toast('Default layout loaded');
}

function loadCalibSpaces(){
  fetch('/api/spaces').then(r=>r.json()).then(d=>{
    calibSpaces=d;
    renderSpaceList();
  });
}

const calCanvas=document.getElementById('calibrate-canvas');
calCanvas.addEventListener('mousedown',e=>{
  const rect=calCanvas.getBoundingClientRect();
  const scaleX=calCanvas.width/rect.width;
  const scaleY=calCanvas.height/rect.height;
  drawStart={x:(e.clientX-rect.left)*scaleX,y:(e.clientY-rect.top)*scaleY};
  isDrawing=true;
});

calCanvas.addEventListener('mousemove',e=>{
  if(!isDrawing||!drawStart) return;
  const rect=calCanvas.getBoundingClientRect();
  const scaleX=calCanvas.width/rect.width;
  const scaleY=calCanvas.height/rect.height;
  const endX=(e.clientX-rect.left)*scaleX;
  const endY=(e.clientY-rect.top)*scaleY;
  drawAllSpaces();
  const ctx=calCanvas.getContext('2d');
  ctx.strokeStyle='#6C5CE7';
  ctx.lineWidth=2;
  ctx.setLineDash([5,3]);
  ctx.strokeRect(Math.min(drawStart.x,endX),Math.min(drawStart.y,endY),
                 Math.abs(endX-drawStart.x),Math.abs(endY-drawStart.y));
  ctx.setLineDash([]);
});

calCanvas.addEventListener('mouseup',e=>{
  if(!isDrawing||!drawStart) return;
  isDrawing=false;
  const rect=calCanvas.getBoundingClientRect();
  const scaleX=calCanvas.width/rect.width;
  const scaleY=calCanvas.height/rect.height;
  const endX=(e.clientX-rect.left)*scaleX;
  const endY=(e.clientY-rect.top)*scaleY;
  const x=Math.min(drawStart.x,endX);
  const y=Math.min(drawStart.y,endY);
  const w=Math.abs(endX-drawStart.x);
  const h=Math.abs(endY-drawStart.y);
  if(w<20||h<20){drawAllSpaces();return;}
  const name=prompt('Name this space (e.g. A1, B2):');
  if(!name){drawAllSpaces();return;}
  calibSpaces[name.toUpperCase()]=[Math.round(x),Math.round(y),Math.round(w),Math.round(h)];
  drawAllSpaces();
  renderSpaceList();
});

function drawAllSpaces(){
  const canvas=document.getElementById('calibrate-canvas');
  const ctx=canvas.getContext('2d');
  if(snapshotImg){
    ctx.putImageData(snapshotImg,0,0);
  }else{
    ctx.fillStyle='#1A2035';
    ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle='#8892A8';
    ctx.font='14px DM Sans';
    ctx.textAlign='center';
    ctx.fillText('Capture a snapshot first',canvas.width/2,canvas.height/2);
  }
  for(const [name,[x,y,w,h]] of Object.entries(calibSpaces)){
    ctx.strokeStyle='#00B894';
    ctx.lineWidth=2;
    ctx.strokeRect(x,y,w,h);
    ctx.fillStyle='rgba(0,184,148,.85)';
    ctx.fillRect(x,y,32,18);
    ctx.fillStyle='#fff';
    ctx.font='bold 12px DM Sans';
    ctx.textAlign='left';
    ctx.fillText(name,x+4,y+13);
  }
}

function renderSpaceList(){
  const list=document.getElementById('space-list');
  const entries=Object.entries(calibSpaces);
  if(!entries.length){list.innerHTML='<div class="events-empty">No spaces defined yet</div>';return;}
  list.innerHTML=entries.map(([name,[x,y,w,h]])=>
    `<div class="space-item"><span style="font-family:'JetBrains Mono',monospace;">${name}, [${x}, ${y}, ${w}x${h}]</span><button onclick="deleteSpace('${name}')">Delete</button></div>`
  ).join('');
}

function deleteSpace(name){
  delete calibSpaces[name];
  drawAllSpaces();
  renderSpaceList();
}

function clearSpaces(){
  if(!confirm('Clear all spaces?')) return;
  calibSpaces={};
  drawAllSpaces();
  renderSpaceList();
}

function saveSpaces(){
  if(Object.keys(calibSpaces).length===0){toast('Draw at least one space first',true);return;}
  fetch('/api/spaces',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(calibSpaces)})
    .then(r=>r.json()).then(d=>{toast(`Saved ${d.count} spaces`);})
    .catch(err=>toast('Error saving',true));
}

drawAllSpaces();
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/detect', methods=['POST'])
def api_detect():
    if 'frame' not in request.files:
        return jsonify({"error": "No frame uploaded"}), 400
    return jsonify(detect_from_frame(request.files['frame'].read()))

@app.route('/api/recommend')
def api_recommend():
    records = get_all_records()
    usage = per_space_usage(records) if records else {}
    return jsonify({"recommendation": recommend_next_space(current_status, usage)})

@app.route('/api/analytics/summary')
def api_analytics():
    records = get_all_records()
    return jsonify({
        "total_records": len(records),
        "hourly": hourly_average(records),
        "daily": daily_average(records),
        "heatmap": heatmap_data(records),
        "prediction": predict_next_hour(records),
        "peaks": peak_times(records, 3)
    })

@app.route('/api/spaces', methods=['GET', 'POST'])
def api_spaces():
    global SPACES, current_status
    if request.method == 'POST':
        new_spaces = request.json
        SPACES = new_spaces
        save_spaces_file(new_spaces)
        current_status = {sid: "available" for sid in SPACES}
        return jsonify({"status": "saved", "count": len(new_spaces)})
    return jsonify(SPACES)

@app.route('/api/demo-data', methods=['POST'])
def api_demo_data():
    count = generate_demo_data(days=30)
    return jsonify({"status": "ok", "added": count})

@app.route('/api/clear-data', methods=['POST'])
def api_clear():
    clear_db()
    return jsonify({"status": "cleared"})


if __name__ == '__main__':
    import socket
    def find_port(start=5050, end=5100):
        for p in range(start, end):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.bind(('0.0.0.0', p))
                s.close()
                return p
            except OSError:
                continue
        return None

    port = find_port()
    if not port:
        print("No free port found 5050-5100")
        exit(1)

    print("\n" + "="*50)
    print("  Smart Parking AI")
    print(f"  Open http://localhost:{port} in your browser")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False)