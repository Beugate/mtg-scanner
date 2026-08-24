#!/usr/bin/env python3
"""
MTG Card Scanner
================
Point your PC's webcam at a Magic: The Gathering card and get back its
name, mana cost, the set it came out in, and its current market price.

Pipeline:
    webcam frame -> quadrilateral card detection -> perspective warp
    -> OCR of the title bar (Tesseract) -> fuzzy match against Scryfall's
    card-name catalog -> Scryfall lookup of every printing -> report.

Usage:
    python mtg_scanner.py                 # GUI with live camera
    python mtg_scanner.py --camera 1      # pick a different webcam
    python mtg_scanner.py --image foo.jpg # identify a saved photo (no GUI)
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
import pytesseract
import requests
from PIL import Image, ImageTk
from rapidfuzz import fuzz, process

APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "cache"
IMAGE_CACHE = CACHE_DIR / "images"
CACHE_DIR.mkdir(exist_ok=True)
IMAGE_CACHE.mkdir(exist_ok=True)

# A standard MTG card is 63 x 88 mm -> aspect ratio 0.716
CARD_W, CARD_H = 488, 680
CARD_RATIO = CARD_W / CARD_H

USER_AGENT = "MTGCardScanner/1.0 (personal card scanner)"


# --------------------------------------------------------------------------
# Tesseract discovery
# --------------------------------------------------------------------------
def locate_tesseract():
    candidates = [
        os.environ.get("TESSERACT_CMD"),
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None


_TESS = locate_tesseract()
if _TESS:
    pytesseract.pytesseract.tesseract_cmd = _TESS


# --------------------------------------------------------------------------
# Scryfall client (cached, rate-limited)
# --------------------------------------------------------------------------
class Scryfall:
    BASE = "https://api.scryfall.com"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": USER_AGENT, "Accept": "application/json"}
        )
        self._last_call = 0.0
        self._lock = threading.Lock()

    def _throttle(self):
        # Scryfall asks for no more than ~10 requests per second.
        with self._lock:
            wait = 0.12 - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()

    def _get(self, path, **params):
        self._throttle()
        r = self.session.get(f"{self.BASE}{path}", params=params, timeout=15)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def card_names(self, max_age_days=7):
        """Every English card name, cached on disk for offline fuzzy matching."""
        path = CACHE_DIR / "card-names.json"
        if path.exists() and (time.time() - path.stat().st_mtime) < max_age_days * 86400:
            try:
                return json.loads(path.read_text(encoding="utf-8"))["data"]
            except Exception:
                pass
        data = self._get("/catalog/card-names")
        path.write_text(json.dumps(data), encoding="utf-8")
        return data["data"]

    def printings(self, exact_name, max_age_hours=12):
        """Every printing of a card, oldest first."""
        safe = re.sub(r"[^a-z0-9]+", "_", exact_name.lower()).strip("_")
        path = CACHE_DIR / f"prints_{safe}.json"
        if path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        data = self._get(
            "/cards/search",
            q=f'!"{exact_name}"',
            unique="prints",
            order="released",
            dir="asc",
        )
        if not data:
            named = self._get("/cards/named", fuzzy=exact_name)
            data = {"data": [named]} if named else {"data": []}
        cards = data.get("data", [])
        if cards:            # never cache a failed lookup for twelve hours
            path.write_text(json.dumps(cards), encoding="utf-8")
        return cards

    def card_image(self, card):
        faces = card.get("card_faces") or [{}]
        uris = card.get("image_uris") or faces[0].get("image_uris")
        if not uris:
            return None
        url = uris.get("normal") or uris.get("large") or uris.get("small")
        if not url:
            return None
        dest = IMAGE_CACHE / f"{card['id']}.jpg"
        if dest.exists():
            return dest
        self._throttle()
        r = self.session.get(url, timeout=20)
        if r.ok:
            dest.write_bytes(r.content)
            return dest
        return None


# --------------------------------------------------------------------------
# Card detection + perspective correction
# --------------------------------------------------------------------------
def order_points(pts):
    """Return corners ordered top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


# Detection runs on a downscaled copy: it is roughly four times faster and
# the extra pixels buy nothing at this scale.
DETECT_WIDTH = 800

# Card outlines survive different failure modes under different binarisations.
# A dark-bordered card on a dark surface has almost no Canny edge but a clean
# adaptive-threshold silhouette, so try several and keep every plausible quad.
BINARISERS = ("adaptive", "autocanny", "otsu_inv", "otsu", "canny")


def _binarise(gray, kind):
    if kind == "canny":
        smooth = cv2.bilateralFilter(gray, 7, 60, 60)
        return cv2.dilate(cv2.Canny(smooth, 40, 120), np.ones((3, 3), np.uint8))
    if kind == "autocanny":
        eq = cv2.createCLAHE(3.0, (8, 8)).apply(gray)
        eq = cv2.GaussianBlur(eq, (5, 5), 0)
        med = np.median(eq)
        edges = cv2.Canny(eq, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)))
        return cv2.dilate(edges, np.ones((3, 3), np.uint8))
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    if kind == "adaptive":
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 31, 7)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return cv2.bitwise_not(binary) if kind == "otsu_inv" else binary


def _quads_in(binary, frame_area):
    """Every card-shaped quadrilateral in a binary image.

    Each is returned as (aspect-ratio error, area, corners); the caller ranks
    them across binarisations.
    """
    found = []
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        area = cv2.contourArea(c)
        if area < frame_area * 0.04:
            break
        if area > frame_area * 0.92:
            continue        # this is the frame border itself, not a card
        peri = cv2.arcLength(c, True)
        quad = None
        for eps in (0.02, 0.03, 0.04):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = approx.reshape(4, 2).astype("float32")
                break
        if quad is None:
            # Rounded corners can defeat approxPolyDP; fall back to the
            # minimum-area rectangle when the contour nearly fills it.
            rot = cv2.minAreaRect(c)
            if area / max(1.0, rot[1][0] * rot[1][1]) > 0.80:
                quad = cv2.boxPoints(rot).astype("float32")
        if quad is None:
            continue
        rect = order_points(quad)
        wide = (np.linalg.norm(rect[1] - rect[0]) + np.linalg.norm(rect[2] - rect[3])) / 2
        tall = (np.linalg.norm(rect[3] - rect[0]) + np.linalg.norm(rect[2] - rect[1])) / 2
        if wide < 50 or tall < 50:
            continue
        ratio = wide / tall
        if 0.55 < ratio < 0.88:                    # upright card
            err = abs(ratio - CARD_RATIO) / CARD_RATIO
            found.append((err, area, rect))
        elif 1.14 < ratio < 1.82:                  # card lying on its side
            err = abs(ratio - 1 / CARD_RATIO) * CARD_RATIO
            found.append((err, area, np.roll(rect, 1, axis=0)))
    return found


def find_card_quads(frame, limit=3):
    """Locate candidate cards, most card-shaped first."""
    h, w = frame.shape[:2]
    scale = DETECT_WIDTH / w if w > DETECT_WIDTH else 1.0
    small = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    area = float(gray.shape[0] * gray.shape[1])
    diag = np.hypot(*gray.shape[:2])

    picked = []
    for kind in BINARISERS:
        for err, cand_area, rect in _quads_in(_binarise(gray, kind), area):
            centre = rect.mean(axis=0)
            if any(np.linalg.norm(centre - p[2].mean(axis=0)) < diag * 0.06
                   and abs(cand_area - p[1]) < max(cand_area, p[1]) * 0.25
                   for p in picked):
                continue                            # same card, another strategy
            picked.append((err, cand_area, rect))
        if len(picked) >= limit:
            break

    # Rank by how card-shaped the quad is, not by raw size: the largest
    # contour is often a wall of background, whereas the true card sits very
    # close to 63:88. Break ties toward the larger candidate.
    picked.sort(key=lambda p: (p[0], -p[1]))
    return [rect / scale for _, _, rect in picked[:limit]]


def find_card_quad(frame):
    """The single most promising card quad, or None. Used for the live overlay."""
    quads = find_card_quads(frame, limit=1)
    return quads[0] if quads else None


def warp_card(frame, quad):
    """Flatten a detected card to the canonical CARD_W x CARD_H portrait image.

    Normalising to one fixed size is deliberate. Everything downstream -- the
    top-hat structuring element, the adaptive-threshold block size, the OCR
    upscale factor -- is tuned against text of a known height, and letting the
    warp size follow the camera detunes all of it at once.
    """
    dst = np.array(
        [[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]], "float32"
    )
    return cv2.warpPerspective(frame, cv2.getPerspectiveTransform(quad, dst),
                               (CARD_W, CARD_H))


def fit_whole(frame):
    """Fallback: treat the centre of the frame as though it were the card."""
    h, w = frame.shape[:2]
    if w / h > CARD_RATIO:                  # too wide -> crop the sides
        new_w = int(h * CARD_RATIO)
        x0 = (w - new_w) // 2
        crop = frame[:, x0:x0 + new_w]
    else:                                   # too tall -> crop top and bottom
        new_h = int(w / CARD_RATIO)
        y0 = (h - new_h) // 2
        crop = frame[y0:y0 + new_h, :]
    return cv2.resize(crop, (CARD_W, CARD_H), interpolation=cv2.INTER_CUBIC)


# --------------------------------------------------------------------------
# OCR
# --------------------------------------------------------------------------
def _prep(crop_bgr, kind, scale=3):
    """Produce one binarisation of a crop for OCR.

    The 3x upscale puts a title bar cropped from a CARD_W-wide card at roughly
    the 30px cap height Tesseract reads best.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if kind == "gray":
        return gray
    smooth = cv2.bilateralFilter(gray, 5, 40, 40)
    if kind == "otsu":
        return cv2.threshold(smooth, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if kind == "adaptive":
        return cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 51, 12)
    if kind == "tophat":
        # Isolates bright strokes from a darker, textured background -- this is
        # what makes light-on-colour retro and showcase titles readable, where
        # a plain threshold merges the lettering into the frame art.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45))
        hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        hat = cv2.normalize(hat, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.threshold(hat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    raise ValueError(kind)


_JUNK = re.compile(r"[^A-Za-z',\-/ .]")


def clean_text(raw):
    txt = _JUNK.sub(" ", raw)
    return re.sub(r"\s+", " ", txt).strip(" .,-'/")


# The name sits at slightly different heights across card frames (original,
# modern, borderless, showcase), and card detection is never pixel-perfect,
# so sweep a few horizontal bands rather than trusting one fixed crop.
TITLE_BANDS = ((0.032, 0.100), (0.012, 0.080), (0.050, 0.120))

# Otsu handles the classic light title bar; adaptive copes with uneven or
# textured frames; top-hat pulls light lettering off a busy background; plain
# greyscale rescues whatever thresholding manages to eat. All four end up as
# tiles in one contact sheet, so extra kinds cost image area, not OCR calls.
PREP_KINDS = ("otsu", "adaptive", "tophat", "gray")


def _contact_sheet(card_img):
    """Stack every band x preprocessing crop into one tall image.

    Each pytesseract call spawns a subprocess, which costs far more than the
    recognition itself -- running the nine crops separately took seconds per
    frame. Stacking them and reading the sheet in a single pass with PSM 6
    gets the same candidate lines for roughly one call's worth of overhead.
    """
    h, w = card_img.shape[:2]
    x0, x1 = int(w * 0.050), int(w * 0.800)             # stop before the mana cost
    tiles = []
    for top, bottom in TITLE_BANDS:
        crop = card_img[int(h * top):int(h * bottom), x0:x1]
        for kind in PREP_KINDS:
            tile = _prep(crop, kind)
            if tile.mean() < 127:            # keep the sheet dark-on-light
                tile = cv2.bitwise_not(tile)
            tiles.append(tile)
            # A generous gap matters: with a narrow one Tesseract's layout
            # analysis merges neighbouring tiles into a single garbled line.
            tiles.append(np.full((60, tile.shape[1]), 255, np.uint8))
    return np.vstack(tiles)


def ocr_title(card_img):
    """Read candidate name strings from the title area of a warped card."""
    try:
        raw = pytesseract.image_to_string(_contact_sheet(card_img), config="--psm 6")
    except Exception:
        return []
    out, seen = [], set()
    for line in raw.splitlines():
        txt = clean_text(line)
        if len(txt) >= 4 and txt not in seen:
            seen.add(txt)
            out.append(txt)
    return out


def ocr_set_code(card_img):
    """Try to read the 3-4 letter set code from the bottom-left of a modern card."""
    h, w = card_img.shape[:2]
    crop = card_img[int(h * 0.915):int(h * 0.960), int(w * 0.045):int(w * 0.400)]
    for kind in ("otsu", "adaptive"):
        try:
            raw = pytesseract.image_to_string(_prep(crop, kind), config="--psm 7")
        except Exception:
            continue
        for tok in re.findall(r"[A-Z0-9]{3,5}", raw.upper()):
            if tok not in {"EN", "NM", "TM"}:
                return tok
    return None


# --------------------------------------------------------------------------
# Identification
# --------------------------------------------------------------------------
@dataclass
class ScanResult:
    name: str
    mana_cost: str
    set_name: str
    set_code: str
    released: str
    rarity: str
    collector_number: str
    prices: dict
    scryfall_uri: str
    confidence: int
    ocr_text: str
    original_set: str
    printings: int
    card: dict


def mana_cost_of(card):
    cost = card.get("mana_cost") or ""
    if not cost and card.get("card_faces"):
        parts = [f.get("mana_cost", "") for f in card["card_faces"] if f.get("mana_cost")]
        cost = " // ".join(parts)
    return cost or "\u2014"


_PUNCT = re.compile(r"[^a-z]")


def _compact(text):
    """Lowercase, letters only -- for comparisons that ignore spacing."""
    return _PUNCT.sub("", text.lower())


class Identifier:
    MATCH_CUTOFF = 78      # below this, treat the read as unrecognised
    CONFIDENT = 90         # at or above this, stop looking for a better read

    def __init__(self, scryfall):
        self.sf = scryfall
        self.names = []          # what we match OCR against
        self.canonical = []      # the Scryfall name each match key belongs to
        self.compact = []
        self.ready = False
        self.error = None

    def load_catalog(self):
        try:
            self.names, self.canonical = self._build_index(self.sf.card_names())
            self.compact = [_compact(n) for n in self.names]
            self.ready = True
        except Exception as e:                       # offline, or the API is down
            self.error = str(e)

    @staticmethod
    def _build_index(names):
        """Match keys and the canonical card name each one maps to.

        Scryfall names a double-faced or split card by both halves --
        "Delver of Secrets // Insectile Aberration" -- but the physical card
        only ever shows one of them, so OCR of the front scores nowhere near
        the full string. Index each face as its own key pointing back at the
        full name, unless a real card already goes by that name.
        """
        keys, canonical = list(names), list(names)
        known = set(names)
        for name in names:
            if "//" not in name:
                continue
            for face in name.split("//"):
                face = face.strip()
                if face and face not in known:
                    known.add(face)
                    keys.append(face)
                    canonical.append(name)
        return keys, canonical

    @staticmethod
    def _query_variants(text):
        """The OCR text, plus contiguous word-spans of it, longest first.

        Title OCR routinely picks up junk at one end or the other -- a stray
        frame glyph, part of the mana cost, a leading capital from the border.
        Trying sub-spans lets a good read survive that.

        The floor on span length is what keeps this honest. Trimming freely
        turns a bad read of "Sylvan Library" into a perfect match for the card
        actually named "Library", and "Meta Ange" into "Anger" -- a confident
        wrong answer, which is worse than no answer. A span must therefore keep
        most of the text and still be long enough that matching it means
        something. The full text is always tried, however short it is.
        """
        out = [text]
        words = text.split()
        floor = max(9, int(len(text) * 0.70))
        seen = {text}
        for i in range(len(words)):
            for j in range(len(words), i, -1):
                cand = " ".join(words[i:j])
                if len(cand) >= floor and cand not in seen:
                    seen.add(cand)
                    out.append(cand)
        out.sort(key=len, reverse=True)
        return out

    def match_name(self, text):
        """Fuzzy-match an OCR string to a real card name.

        Uses plain ratio rather than WRatio: WRatio's partial-ratio component
        saturates at 86 for every card sharing a word with the query (all ~200
        "... Mage" cards, say), which buries the actual match below the cutoff.
        """
        if not self.names:
            return None

        def plausible(name, query):
            # A short name matched against a distinctly longer read is how
            # "Py ay re" becomes "Yare" and "am Library" becomes "Library".
            # Long names are unaffected: trailing OCR junk is normal there.
            return not (len(name) < 8 and len(query) >= len(name) + 3)

        best = None
        for query in self._query_variants(text):
            m = process.extractOne(
                query, self.names, scorer=fuzz.ratio,
                processor=str.lower, score_cutoff=self.MATCH_CUTOFF,
            )
            if m and plausible(m[0], query) and (best is None or m[1] > best[1]):
                best = (self.canonical[m[2]], m[1])

            # Tesseract drops and invents spaces constantly ("SolRing",
            # "fforceofWill"), which wrecks a straight ratio. Compare the
            # letters-only forms as well and keep whichever scores better.
            squashed = _compact(query)
            if len(squashed) >= 4:
                m = process.extractOne(
                    squashed, self.compact, scorer=fuzz.ratio,
                    score_cutoff=self.MATCH_CUTOFF,
                )
                if (m and plausible(self.names[m[2]], query)
                        and (best is None or m[1] > best[1])):
                    best = (self.canonical[m[2]], m[1])

            if best and best[1] >= self.CONFIDENT:
                break
        return best

    def identify(self, frame):
        """Run the full pipeline on a single BGR frame."""
        candidates = [warp_card(frame, q) for q in find_card_quads(frame)]
        candidates.append(fit_whole(frame))     # last resort: assume it fills the frame

        # Upright first for every candidate, then a second sweep upside down:
        # cards are almost always held the right way up, and each OCR pass is
        # the expensive part, so do not pay for rotation until it is needed.
        upright = list(candidates)
        flipped = [cv2.rotate(c, cv2.ROTATE_180) for c in candidates]

        best = None            # (score, name, card_img, ocr_text)
        for img in upright + flipped:
            for text in ocr_title(img):
                m = self.match_name(text)
                if m and (best is None or m[1] > best[0]):
                    best = (m[1], m[0], img, text)
            if best and best[0] >= self.CONFIDENT:
                break                              # confident enough, stop
        if best is None:
            return None

        score, name, card_img, ocr_text = best
        prints = self.sf.printings(name)
        if not prints:
            return None

        chosen = prints[0]          # oldest printing == the set it came out in
        code = ocr_set_code(card_img)
        if code:
            for p in prints:
                if p.get("set", "").upper() == code:
                    chosen = p
                    break

        return ScanResult(
            name=chosen.get("name", name),
            mana_cost=mana_cost_of(chosen),
            set_name=chosen.get("set_name", "?"),
            set_code=chosen.get("set", "?").upper(),
            released=chosen.get("released_at", "?"),
            rarity=chosen.get("rarity", "?"),
            collector_number=chosen.get("collector_number", "?"),
            prices=chosen.get("prices") or {},
            scryfall_uri=chosen.get("scryfall_uri", ""),
            confidence=int(score),
            ocr_text=ocr_text,
            original_set=prints[0].get("set_name", "?"),
            printings=len(prints),
            card=chosen,
        )


def format_prices(prices):
    labels = [
        ("usd", "USD"), ("usd_foil", "USD foil"), ("usd_etched", "USD etched"),
        ("eur", "EUR"), ("eur_foil", "EUR foil"), ("tix", "MTGO tix"),
    ]
    sym = {"usd": "$", "usd_foil": "$", "usd_etched": "$",
           "eur": "\u20ac", "eur_foil": "\u20ac", "tix": ""}
    out = []
    for key, label in labels:
        v = prices.get(key)
        if v:
            out.append((label, f"{sym[key]}{v}"))
    return out or [("Price", "no price data")]


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------
BG = "#14161a"
PANEL = "#1c1f26"
FG = "#e8eaed"
MUTED = "#9aa2ad"
ACCENT = "#f0a830"


class ScannerApp:
    PREVIEW_W = 720
    DETECT_EVERY = 3        # run card detection on every Nth preview frame

    def __init__(self, root, camera_index=0):
        self.root = root
        self.sf = Scryfall()
        self.ident = Identifier(self.sf)
        self.results = queue.Queue()
        self.scanning = False
        self.last_scan_time = 0.0
        self.card_detected = False
        self.frame = None
        self.overlay_quad = None
        self.frame_no = 0
        self.last_card_id = None
        self._status_text = None
        self.cap = None
        self._imgrefs = {}

        root.title("MTG Card Scanner")
        root.configure(bg=BG)
        root.minsize(1120, 720)

        main = tk.Frame(root, bg=BG)
        main.pack(fill="both", expand=True, padx=12, pady=12)

        # --- right panel: results (packed first so it keeps its width) ---
        right = tk.Frame(main, bg=PANEL, width=380)
        right.pack(side="right", fill="y", padx=(12, 0))
        right.pack_propagate(False)

        self.art = tk.Label(right, bg=PANEL)
        self.art.pack(pady=(12, 8))

        self.name_lbl = tk.Label(right, text="No card scanned yet", bg=PANEL, fg=FG,
                                 font=("Segoe UI", 15, "bold"), wraplength=340,
                                 justify="left", anchor="w")
        self.name_lbl.pack(fill="x", padx=16)

        self.footer = tk.Label(right, text="", bg=PANEL, fg=MUTED, font=("Segoe UI", 8),
                               wraplength=340, justify="left", anchor="w")
        self.footer.pack(fill="x", padx=16, pady=(10, 12), side="bottom")

        self.fields = tk.Frame(right, bg=PANEL)
        self.fields.pack(fill="x", padx=16, pady=(8, 0))

        # --- left: live preview + controls ---
        left = tk.Frame(main, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        self.video = tk.Label(left, bg="#000000")
        self.video.pack(fill="both", expand=True)

        controls = tk.Frame(left, bg=BG)
        controls.pack(fill="x", pady=(10, 0))

        self.scan_btn = tk.Button(
            controls, text="Scan card  (Space)", command=self.request_scan,
            bg=ACCENT, fg="#1a1a1a", activebackground="#ffc45c",
            font=("Segoe UI", 12, "bold"), relief="flat", padx=18, pady=8, cursor="hand2",
        )
        self.scan_btn.pack(side="left")

        self.auto_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            controls, text="Auto-scan when a card is held up", variable=self.auto_var,
            bg=BG, fg=MUTED, selectcolor=PANEL, activebackground=BG,
            activeforeground=FG, font=("Segoe UI", 10), relief="flat",
        ).pack(side="left", padx=14)

        tk.Label(controls, text="Camera", bg=BG, fg=MUTED,
                 font=("Segoe UI", 10)).pack(side="left", padx=(10, 4))
        self.cam_var = tk.StringVar(value=str(camera_index))
        cam_box = ttk.Combobox(controls, textvariable=self.cam_var, width=3,
                               values=["0", "1", "2", "3"], state="readonly")
        cam_box.pack(side="left")
        cam_box.bind("<<ComboboxSelected>>", self.switch_camera)

        self.status = tk.Label(left, text="Loading card database\u2026", bg=BG, fg=MUTED,
                               font=("Segoe UI", 10), anchor="w")
        self.status.pack(fill="x", pady=(8, 0))

        root.bind("<space>", lambda e: self.request_scan())
        root.bind("<Escape>", lambda e: self.quit())
        root.protocol("WM_DELETE_WINDOW", self.quit)

        threading.Thread(target=self._load_catalog, daemon=True).start()
        self.open_camera(camera_index)
        self.tick()

    # ---- setup helpers ----
    def _load_catalog(self):
        self.ident.load_catalog()
        if self.ident.ready:
            self.set_status(f"Ready \u2014 {len(self.ident.names):,} card names loaded. "
                            f"Hold a card up to the camera.")
        else:
            self.set_status(f"Could not load the Scryfall catalog: {self.ident.error}")

    def open_camera(self, index):
        if self.cap is not None:
            self.cap.release()
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        # Ask for more than most webcams give; the driver quietly clamps to
        # the nearest mode it supports, and OCR wants every pixel it can get.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not self.cap.isOpened():
            self.set_status(f"Could not open camera {index} \u2014 try another index.")

    def switch_camera(self, _event=None):
        self.open_camera(int(self.cam_var.get()))

    def set_status(self, text):
        """Safe to call from any thread; tick() does the actual widget update."""
        self._status_text = text

    # ---- main loop ----
    def tick(self):
        ok, frame = self.cap.read() if self.cap else (False, None)
        if ok:
            self.frame = frame
            self.frame_no += 1
            display = frame.copy()
            # Detection costs ~10ms on a frame with no card in it, which is
            # half the tick budget. Re-detect a few times a second and reuse
            # the last outline in between; the overlay still tracks the hand.
            if self.frame_no % self.DETECT_EVERY == 0 or self.overlay_quad is None:
                self.overlay_quad = find_card_quad(frame)
            quad = self.overlay_quad
            self.card_detected = quad is not None
            if quad is not None:
                cv2.polylines(display, [quad.astype(np.int32)], True, (60, 220, 60), 3)
            else:
                h, w = display.shape[:2]
                gh = int(h * 0.82)
                gw = int(gh * CARD_W / CARD_H)
                x0, y0 = (w - gw) // 2, (h - gh) // 2
                cv2.rectangle(display, (x0, y0), (x0 + gw, y0 + gh), (90, 90, 90), 2)

            if self.scanning:
                cv2.putText(display, "scanning...", (16, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

            self.show_frame(display)

            if (self.auto_var.get() and self.card_detected and not self.scanning
                    and self.ident.ready and time.time() - self.last_scan_time > 2.5):
                self.request_scan()

        if self._status_text is not None:
            self.status.config(text=self._status_text)
            self._status_text = None

        self.drain_results()
        self.root.after(20, self.tick)

    def show_frame(self, bgr):
        h, w = bgr.shape[:2]
        small = cv2.resize(bgr, (self.PREVIEW_W, int(h * self.PREVIEW_W / w)))
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        self._imgrefs["video"] = img
        self.video.config(image=img)

    # ---- scanning ----
    def request_scan(self):
        if self.scanning or self.frame is None:
            return
        if not self.ident.ready:
            self.set_status("Still loading the card database\u2026")
            return
        self.scanning = True
        self.last_scan_time = time.time()
        self.set_status("Reading card\u2026")
        threading.Thread(target=self._scan_worker, args=(self.frame.copy(),),
                         daemon=True).start()

    def _scan_worker(self, frame):
        try:
            result = self.ident.identify(frame)
            img_path = None
            if result:
                try:
                    img_path = self.sf.card_image(result.card)
                except Exception:
                    pass
            self.results.put(("ok", result, img_path))
        except Exception as e:
            self.results.put(("err", e, None))

    def drain_results(self):
        try:
            kind, payload, img_path = self.results.get_nowait()
        except queue.Empty:
            return
        self.scanning = False
        self.last_scan_time = time.time()
        if kind == "err":
            self.set_status(f"Scan failed: {payload}")
        elif payload is None:
            self.set_status("No card recognised \u2014 try better lighting, "
                            "or fill more of the frame.")
        elif payload.card.get("id") == self.last_card_id:
            self.set_status(f"Still showing {payload.name}.")
        else:
            self.render(payload, img_path)

    # ---- rendering ----
    def render(self, r, img_path):
        self.last_card_id = r.card.get("id")
        self.name_lbl.config(text=r.name)

        for child in self.fields.winfo_children():
            child.destroy()

        rows = [
            ("Mana cost", r.mana_cost),
            ("Set", f"{r.set_name}  ({r.set_code})"),
            ("Released", r.released),
            ("Rarity", r.rarity.title()),
            ("Number", f"#{r.collector_number}"),
            ("", ""),
        ]
        rows.extend(format_prices(r.prices))

        for label, value in rows:
            if not label and not value:
                tk.Frame(self.fields, bg="#2a2f38", height=1).pack(fill="x", pady=6)
                continue
            row = tk.Frame(self.fields, bg=PANEL)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=label, bg=PANEL, fg=MUTED, font=("Segoe UI", 10),
                     width=12, anchor="w").pack(side="left")
            tk.Label(row, text=value, bg=PANEL, fg=FG, font=("Segoe UI", 11, "bold"),
                     anchor="w", wraplength=210, justify="left").pack(side="left", fill="x")

        note = (f"OCR read \u201c{r.ocr_text}\u201d \u00b7 match {r.confidence}% "
                f"\u00b7 {r.printings} printing(s)")
        if r.printings > 1:
            note += f"\nShowing the original printing ({r.original_set})."
        self.footer.config(text=note)

        if img_path and Path(img_path).exists():
            im = Image.open(img_path)
            im.thumbnail((200, 280))
            photo = ImageTk.PhotoImage(im)
            self._imgrefs["art"] = photo
            self.art.config(image=photo)

        self.set_status(f"Identified: {r.name} \u2014 {r.set_name}")

    def quit(self):
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


# --------------------------------------------------------------------------
# CLI mode
# --------------------------------------------------------------------------
def run_cli(image_path):
    try:                                      # Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not read image: {image_path}")
        return 1
    ident = Identifier(Scryfall())
    print("Loading the Scryfall card-name catalog...")
    ident.load_catalog()
    if not ident.ready:
        print(f"Failed: {ident.error}")
        return 1
    r = ident.identify(frame)
    if r is None:
        print("No card recognised.")
        return 2
    print()
    print(f"  Name       : {r.name}")
    print(f"  Mana cost  : {r.mana_cost}")
    print(f"  Set        : {r.set_name} ({r.set_code}), released {r.released}")
    print(f"  Rarity     : {r.rarity.title()}  #{r.collector_number}")
    for label, value in format_prices(r.prices):
        print(f"  {label:<11}: {value}")
    print(f"  Scryfall   : {r.scryfall_uri}")
    print(f"  [OCR read '{r.ocr_text}', match {r.confidence}%, {r.printings} printings]")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Scan Magic: The Gathering cards with your webcam.")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    ap.add_argument("--image", help="identify a saved image instead of using the camera")
    args = ap.parse_args()

    if _TESS is None:
        print("Tesseract OCR was not found. Install it from "
              "https://github.com/UB-Mannheim/tesseract/wiki, or set TESSERACT_CMD.")
        return 1

    if args.image:
        return run_cli(args.image)

    root = tk.Tk()
    ScannerApp(root, args.camera)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
