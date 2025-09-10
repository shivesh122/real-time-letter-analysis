# stroke_analyzer.py
import math
import numpy as np
from typing import List, Tuple, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import time
from fastapi.middleware.cors import CORSMiddleware

Point = Tuple[float, float]
Stroke = List[Point]

# -----------------------
# Utilities: normalization & resampling
# -----------------------
def bounding_box(points: List[Point]):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)

def flatten_strokes(strokes: List[Stroke]) -> List[Point]:
    pts = []
    for s in strokes:
        pts.extend(s)
    return pts

def normalize_strokes(strokes: List[Stroke], canvas_w: int, canvas_h: int, ref_size=100.0):
    """Scale + translate strokes to a canonical ref_size (preserve aspect ratio)."""
    pts = flatten_strokes(strokes)
    if not pts:
        return strokes
    minx, miny, maxx, maxy = bounding_box(pts)
    w = maxx - minx
    h = maxy - miny
    if w == 0 and h == 0:
        return strokes
    scale = ref_size / max(w, h)
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    out = []
    for s in strokes:
        ns = [((p[0]-cx)*scale, (p[1]-cy)*scale) for p in s]
        out.append(ns)
    return out

def resample_points(points: List[Point], n: int) -> List[Point]:
    """Resample a polyline to n equally spaced points (by arc length)."""
    if len(points) < 2:
        return points + [(points[-1] if points else (0,0))]*(n-len(points))
    dists = [0.0]
    for i in range(1,len(points)):
        dx = points[i][0]-points[i-1][0]
        dy = points[i][1]-points[i-1][1]
        dists.append(math.hypot(dx,dy))
    cum = np.cumsum(dists)
    total = cum[-1]
    if total == 0:
        return [points[0]]*n
    target = np.linspace(0, total, n)
    res = []
    j = 0
    for t in target:
        while j < len(cum)-1 and cum[j+1] < t:
            j += 1
        if j == len(cum)-1:
            res.append(points[-1])
        else:
            t0, t1 = cum[j], cum[j+1]
            p0, p1 = points[j], points[j+1]
            if t1 - t0 == 0:
                res.append(p0)
            else:
                alpha = (t - t0) / (t1 - t0)
                x = p0[0] + alpha*(p1[0]-p0[0])
                y = p0[1] + alpha*(p1[1]-p0[1])
                res.append((x,y))
    return res

# -----------------------
# Fast similarity: DTW (small sequences)
# -----------------------
def euclid(a: Point, b: Point):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def dtw_distance(seq1: List[Point], seq2: List[Point], window=None):
    n, m = len(seq1), len(seq2)
    if window is None:
        window = max(n, m)
    window = max(window, abs(n-m))
    INF = 1e9
    D = [[INF]*(m+1) for _ in range(n+1)]
    D[0][0] = 0
    for i in range(1, n+1):
        jstart = max(1, i - window)
        jend = min(m, i + window)
        for j in range(jstart, jend+1):
            cost = euclid(seq1[i-1], seq2[j-1])
            D[i][j] = cost + min(D[i-1][j], D[i][j-1], D[i-1][j-1])
    return D[n][m]

# -----------------------
# Feedback generator
# -----------------------
class FeedbackGenerator:
    def __init__(self, thresholds=None):
        # thresholds can be tuned with real data
        self.thresh = thresholds or {
            'stroke_good': 25.0,    # DTW distance threshold per stroke (lower is better)
            'letter_good': 60.0,
            'angle_error': 20.0,    # degrees
            'endpoint_tol': 20.0
        }

    def stroke_feedback(self, dist, expected_len_ratio=1.0):
        if dist <= self.thresh['stroke_good']:
            return "correct", 0.9
        elif dist <= self.thresh['stroke_good']*2:
            return "partial", 0.6
        else:
            return "incorrect", max(0.1, 1 - (dist/self.thresh['letter_good']))

    def letter_feedback(self, total_dist):
        if total_dist <= self.thresh['letter_good']:
            return "correct", 0.9
        elif total_dist <= self.thresh['letter_good']*2:
            return "partial", 0.6
        else:
            return "incorrect", max(0.1, 1 - (total_dist/(self.thresh['letter_good']*3)))

# -----------------------
# StrokeAnalyzer
# -----------------------
class StrokeAnalyzer:
    def __init__(self, letter_templates: Dict[str, List[Stroke]] = None):
        # letter_templates: mapping letter -> list of stroke polylines (points)
        self.letter_templates = letter_templates or {}
        self.feedback_gen = FeedbackGenerator()

    def add_template(self, letter: str, strokes: List[Stroke]):
        self.letter_templates[letter] = strokes

    def analyze_stroke_realtime(self, letter: str, stroke_number: int, stroke_points: Stroke, canvas_dimensions: Dict[str,int]):
        """Compare current stroke to template stroke and return feedback + confidence quickly."""
        t0 = time.time()
        templates = self.letter_templates.get(letter)
        if not templates or stroke_number-1 >= len(templates):
            return {"feedback":"unknown_letter_or_stroke","confidence":0.0, "latency_ms": int((time.time()-t0)*1000)}
        template_stroke = templates[stroke_number-1]
        # Normalize both to same ref frame
        norm_in = normalize_strokes([stroke_points], canvas_dimensions['width'], canvas_dimensions['height'])[0]
        norm_tmpl = normalize_strokes([template_stroke], canvas_dimensions['width'], canvas_dimensions['height'])[0]
        # Resample
        S = 32
        r_in = resample_points(norm_in, S)
        r_t = resample_points(norm_tmpl, S)
        dist = dtw_distance(r_in, r_t, window=8)
        fb, conf = self.feedback_gen.stroke_feedback(dist)
        latency = int((time.time()-t0)*1000)
        return {"feedback": fb, "confidence": round(conf, 3), "distance": round(dist,2), "latency_ms": latency}

    def analyze_complete_letter(self, letter: str, all_strokes: List[Stroke], canvas_dimensions: Dict[str,int]):
        t0 = time.time()
        templates = self.letter_templates.get(letter)
        if not templates:
            return {"feedback": "unknown_letter", "confidence": 0.0}
        # Normalize both sets
        norm_in = normalize_strokes(all_strokes, canvas_dimensions['width'], canvas_dimensions['height'])
        norm_tmpl = normalize_strokes(templates, canvas_dimensions['width'], canvas_dimensions['height'])
        # Compare stroke counts and order
        results = []
        total_dist = 0.0
        for i, tmpl in enumerate(norm_tmpl):
            if i >= len(norm_in):
                results.append({"stroke": i+1, "feedback":"missing", "confidence": 0.0})
                total_dist += 80.0
                continue
            ri = resample_points(norm_in[i], 40)
            rt = resample_points(tmpl, 40)
            d = dtw_distance(ri, rt, window=10)
            total_dist += d
            fb, conf = self.feedback_gen.stroke_feedback(d)
            results.append({"stroke": i+1, "feedback": fb, "confidence": conf, "distance": round(d,2)})
        # Check for extra strokes
        if len(norm_in) > len(norm_tmpl):
            for j in range(len(norm_tmpl), len(norm_in)):
                results.append({"stroke": j+1, "feedback":"extra_stroke", "confidence":0.2})
                total_dist += 40.0
        letter_fb, letter_conf = self.feedback_gen.letter_feedback(total_dist)
        latency = int((time.time()-t0)*1000)
        return {"letter_feedback": letter_fb, "confidence": round(letter_conf,3), "strokes": results, "total_distance": round(total_dist,2), "latency_ms": latency}

# -----------------------
# Example usage & simple API
# -----------------------

# Simple Pydantic models for the endpoint
class PointModel(BaseModel):
    x: float
    y: float

class StrokeModel(BaseModel):
    points: List[PointModel]

class StrokeAnalyzeRequest(BaseModel):
    letter: str
    stroke_number: int
    stroke: StrokeModel
    canvas_dimensions: Dict[str,int]

class LetterAnalyzeRequest(BaseModel):
    letter: str
    strokes: List[StrokeModel]
    canvas_dimensions: Dict[str,int]

# Create analyzer and add a simple template (example)
analyzer = StrokeAnalyzer()
# Add simple 'A' template (example coordinates)
analyzer.add_template('A', [
    [(310,300),(350,200),(390,300)],  # left diag
    [(350,200),(370,250),(390,300)],  # right diag
    [(320,270),(380,270)]             # crossbar
])

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins (for demo)
    allow_credentials=True,
    allow_methods=["*"],   # allow POST, GET, OPTIONS, etc.
    allow_headers=["*"],   # allow all headers
)

import random

@app.post("/analyze/letter")
async def analyze_letter(request: dict):
    letter = request.get("letter", "?")
    strokes = request.get("strokes", [])

    # Dummy scoring logic for demo
    score = round(random.uniform(0.5, 1.0), 2)
    feedback = "Looks good!" if score > 0.75 else "Keep practicing"

    stroke_results = []
    for s in strokes:
        conf = round(random.uniform(0.6, 0.95), 2)
        stroke_results.append({
            "stroke_number": s.get("stroke_number"),
            "feedback": "correct" if conf > 0.75 else "partial",
            "confidence": conf
        })

    return {
        "letter": letter,
        "score": score,
        "feedback": feedback,
        "strokes": stroke_results
    }


import random

@app.post("/analyze/stroke")
async def analyze_stroke(request: dict):
    # Extract stroke number if provided
    stroke_number = request.get("stroke_number", 1)

    confidence = round(random.uniform(0.6, 0.95), 2)
    feedback = "correct" if confidence > 0.75 else "partial"
    corrections = [] if feedback == "correct" else ["make the line straighter"]

    return {
        "stroke_number": stroke_number,
        "feedback": feedback,
        "confidence": confidence,
        "corrections": corrections,
        "visual_indicators": [{"x": 150, "y": 250, "type": "hint"}]
    }

# If running locally:
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
