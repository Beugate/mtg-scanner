#!/usr/bin/env python3
"""
MTG Card Scanner
================
Point your PC's webcam at a Magic: The Gathering card and get back its
name, mana cost, which printing it is, and that printing's market price.

Pipeline:
    webcam frame -> quadrilateral card detection -> perspective warp
    -> OCR of the title bar (Tesseract) -> fuzzy match against Scryfall's
    card-name catalog -> Scryfall lookup of every printing -> OCR of the
    set line to pick the one being held -> report.

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
        self._sets = None

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

    def set_index(self, max_age_days=7):
        """Set code -> set object, cached on disk and in memory.

        Wanted for `printed_size`: the total a card prints beside its collector
        number ("248/383") is the size of the set as it was *printed*, which
        drifts from Scryfall's card_count once tokens and promos are counted.
        """
        if self._sets is not None:
            return self._sets
        path = CACHE_DIR / "sets.json"
        data = None
        if path.exists() and (time.time() - path.stat().st_mtime) < max_age_days * 86400:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                data = None
        if data is None:
            try:
                data = self._get("/sets")
            except Exception:                        # offline: fall back to codes
                data = None
            if data:
                path.write_text(json.dumps(data), encoding="utf-8")
        self._sets = {s["code"].upper(): s for s in (data or {}).get("data", [])}
        return self._sets

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


def warp_card(frame, quad, scale=1):
    """Flatten a detected card to a CARD_W x CARD_H portrait image.

    Normalising to one fixed size is deliberate. Everything downstream -- the
    top-hat structuring element, the adaptive-threshold block size, the OCR
    upscale factor -- is tuned against text of a known height, and letting the
    warp size follow the camera detunes all of it at once.

    `scale` multiplies that canonical size. Only the set line asks for it: the
    printing details are set in type far smaller than the title, and at scale 1
    they land under Tesseract's floor no matter how good the camera is. Warping
    the source frame again at 3x keeps those pixels instead of upscaling a crop
    that has already been thrown away.
    """
    w, h = CARD_W * scale, CARD_H * scale
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], "float32")
    return cv2.warpPerspective(frame, cv2.getPerspectiveTransform(quad, dst), (w, h))


def fit_whole(frame, scale=1):
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
    return cv2.resize(crop, (CARD_W * scale, CARD_H * scale),
                      interpolation=cv2.INTER_CUBIC)


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


# The bottom info line is the only place a card states which printing it is.
# Where it sits moved with the frame: M15 (2014) and later print the collector
# number, set code and language on two small lines under the text box, while
# every frame before that tucks the number onto the end of the copyright line.
# Sweep both bands rather than trying to guess the frame first.
SET_LINE_BANDS = ((0.940, 0.992), (0.895, 0.992))

# How much larger than the canonical warp to re-cut the card for this one read.
SET_LINE_SCALE = 5

# How many framings of the card to try before giving up on the set line.
SET_LINE_VIEWS = 3

# Otsu carries the ordinary white-on-dark line; top-hat rescues it where the
# art behind it is bright and busy; adaptive copes with an uneven exposure.
SET_LINE_KINDS = ("otsu", "tophat", "adaptive")

# Tokens that turn up in the bottom line and are not set codes.
_NOT_SET_CODES = {"EN", "DE", "FR", "IT", "JP", "ES", "PT", "RU", "KR", "CN",
                  "NM", "TM", "LLC", "INC", "THE", "OF", "AND", "ALL", "ILLUS",
                  "WIZARDS", "COAST", "RIGHTS", "RESERVED"}

# M15 and later print "<set> * <language>" under the text box. Anchoring on
# the language is what separates the set code from surrounding OCR debris.
_CODE_BESIDE_LANGUAGE = (r"\b([A-Z0-9]{3,5})\s*[^A-Z0-9\s]{0,2}\s*"
                         r"(?:EN|DE|FR|IT|ES|PT|JA|JP|KO|RU|ZH|ZHS|ZHT)\b")

# Tesseract renders the slash of "248/383" as almost any thin glyph, and on a
# busy background drops it altogether.
_SLASH = r"[/\|1lI:;,.·• ]"


@dataclass
class SetLine:
    """Candidate readings of a card's bottom info line.

    Every field is a tuple of possibilities rather than one value, because the
    line is small enough that several binarisations disagree about it. Sorting
    out which reading is real is pick_printing's job -- it has the actual list
    of printings to check them against, which this does not.
    """
    codes: tuple = ()        # 3-5 char set codes; modern frames only
    numbers: tuple = ()      # (collector number, printed set size) pairs
    years: tuple = ()        # copyright years; the last one is the print year


def parse_set_line(text):
    """Pull set codes, collector numbers and copyright years from an OCR read."""
    up = re.sub(r"\s+", " ", text.upper())
    years = tuple(int(y) for y in re.findall(r"(?:19|20)\d{2}", up))

    # A set code counts only where the frame prints it beside the language,
    # "2XM * EN". Any bare three-letter token would otherwise be enough to pick
    # a printing on its own, and OCR turns artist names and the copyright line
    # into plenty of those -- several of which are real set codes.
    codes = tuple(t for t in re.findall(_CODE_BESIDE_LANGUAGE, up)
                  if t not in _NOT_SET_CODES and not t.isdigit())

    numbers = []
    for a, b in re.findall(r"(\d{1,4})\s*" + _SLASH + r"\s*(\d{2,4})", up):
        n, t = int(a), int(b)
        if n in years and t in years:
            continue                     # a copyright range, "1993-2007"
        numbers.append((n, t))

    # The eaten-slash case: "248/383" comes back as one run of digits. Offer
    # every split; only one an actual printing agrees with will ever score.
    # Years go first so "1993-2007" cannot masquerade as a collector number.
    for run in re.findall(r"\d{5,8}", re.sub(r"(?:19|20)\d{2}", " ", up)):
        for cut in range(1, len(run) - 1):
            n, t = int(run[:cut]), int(run[cut:])
            if t >= 20:
                numbers.append((n, t))

    return SetLine(codes, tuple(numbers), years)


def read_set_line(card_img):
    """OCR the bottom info line of a warped card, whichever frame it uses.

    Both bands and every preprocessing go into a single contact sheet, for the
    same reason ocr_title uses one: each pytesseract call spawns a subprocess,
    and six of those per candidate framing costs more than the whole rest of
    the scan put together.

    Each output line is parsed on its own so that readings cannot contaminate
    each other -- a year lifted off one tile must not pair up with digits from
    the next.
    """
    h, w = card_img.shape[:2]
    tiles = []
    for top, bottom in SET_LINE_BANDS:
        crop = card_img[int(h * top):int(h * bottom), 0:w]
        if crop.size == 0:
            continue
        for kind in SET_LINE_KINDS:
            tile = _prep(crop, kind, scale=1)
            if tile.mean() < 127:            # keep the sheet dark-on-light
                tile = cv2.bitwise_not(tile)
            tiles.append(tile)
            tiles.append(np.full((60, tile.shape[1]), 255, np.uint8))
    if not tiles:
        return SetLine()
    try:
        raw = pytesseract.image_to_string(np.vstack(tiles), config="--psm 6")
    except Exception:
        return SetLine()

    codes, numbers, years = [], [], []
    for line in raw.splitlines():
        got = parse_set_line(line)
        codes.extend(got.codes)
        numbers.extend(got.numbers)
        years.extend(got.years)
    return SetLine(tuple(codes), tuple(numbers), tuple(years))


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
    set_evidence: str = ""      # what identified the printing; "" == unreadable


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


# How much each thing the bottom line can tell us is worth. The set code is
# unambiguous where it is printed at all; a collector number and set size
# together are nearly so; either alone is suggestive; and the copyright year
# only ever separates printings that something else already narrowed down --
# every core set of an era shares a size, so "249 cards" means M10 through M13
# until the year picks one.
SCORE_SET_CODE = 6
SCORE_NUMBER_AND_SIZE = 5
SCORE_NUMBER_ONLY = 2
SCORE_SIZE_ONLY = 2
SCORE_YEAR = 2

# What the evidence has to add up to, and beat the next set by, before a
# printing other than the original is reported. Deliberately strict, for the
# same reason the name matching is: a wrong set stated confidently is worse
# than falling back to the original and admitting the line was unreadable.
PRINTING_MIN_SCORE = 4
PRINTING_MIN_MARGIN = 2


def _printed_size(set_obj):
    """How many cards the set says it has *on the card*, not in Scryfall."""
    return set_obj.get("printed_size") or set_obj.get("card_count") or 0


def pick_printing(prints, line, sets):
    """Which printing was scanned, and the evidence for saying so.

    Returns (card, evidence). Evidence is a short string naming what agreed, or
    "" when the bottom line could not be read well enough to tell -- in which
    case the caller gets the oldest printing and should say so, rather than
    presenting a guess as a fact.

    Scores rather than matching on one field because no single field is
    available across all frames: only M15 (2014) and later print a set code,
    the collector number goes back to 1998, and the copyright year is on
    everything but is far too coarse to decide on alone.
    """
    if len(prints) == 1:
        return prints[0], "only printing"
    if not (line.codes or line.numbers or line.years):
        return prints[0], ""

    year = max(line.years) if line.years else 0
    scored = []
    for p in prints:
        code = (p.get("set") or "").upper()
        size = _printed_size(sets.get(code, {}))
        digits = re.sub(r"\D", "", p.get("collector_number") or "")
        num = int(digits) if digits else 0
        released = (p.get("released_at") or "")[:4]

        score, why = 0, []
        if code and code in line.codes:
            score += SCORE_SET_CODE
            why.append(code)
        if num and size and (num, size) in line.numbers:
            score += SCORE_NUMBER_AND_SIZE         # the whole "248/383" agrees
            why.append("%d/%d" % (num, size))
        elif num and any(n == num for n, _ in line.numbers):
            score += SCORE_NUMBER_ONLY
            why.append("#%d" % num)
        elif size and any(t == size for _, t in line.numbers):
            score += SCORE_SIZE_ONLY
            why.append("?/%d" % size)
        if year and released == str(year):
            score += SCORE_YEAR
            why.append(released)
        scored.append((score, p, ", ".join(why)))

    scored.sort(key=lambda s: -s[0])
    top, card, why = scored[0]

    # The runner-up that matters is the best-scoring *other set*. Scryfall
    # lists a set several times over -- foil, alternate frame, promo -- and
    # those entries score identically because they share a collector number
    # and a release date. Two of them tying says nothing about which set is
    # being held, and treating it as an ambiguity throws the answer away.
    top_code = (card.get("set") or "").upper()
    runner_up = next((score for score, p, _ in scored
                      if (p.get("set") or "").upper() != top_code), 0)

    if top >= PRINTING_MIN_SCORE and top - runner_up >= PRINTING_MIN_MARGIN:
        return card, why
    return prints[0], ""


class _View:
    """One flattened reading of the frame, re-cuttable at a higher resolution.

    Keeping the quad instead of just the warped image is what lets the set line
    be read at SET_LINE_SCALE afterwards, without paying to warp every
    candidate at that size up front when all but one get discarded.
    """

    def __init__(self, frame, quad, flipped=False):
        self.frame = frame
        self.quad = quad                       # None -> the centre-crop fallback
        self.flipped = flipped
        self._cache = {}

    def image(self, scale=1):
        img = self._cache.get(scale)
        if img is None:
            img = (warp_card(self.frame, self.quad, scale) if self.quad is not None
                   else fit_whole(self.frame, scale))
            if self.flipped:
                img = cv2.rotate(img, cv2.ROTATE_180)
            self._cache[scale] = img
        return img

    def upside_down(self):
        return _View(self.frame, self.quad, not self.flipped)


class Identifier:
    # Below this, treat the read as unrecognised. 82 rather than something
    # looser because a title the preprocessings cannot resolve at all still
    # emits noise lines, and a long enough one lands in the high 70s against
    # some real card by chance: a Prophecy-frame Rhystic Study reads as
    # "PRESSION GRASSY", which is a 80% match for "Suppression Ray". Measured
    # over testdata/, every genuine match scores 83 or better, so the window
    # between the two is where the cutoff belongs.
    MATCH_CUTOFF = 82
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

    def _resolve_printing(self, prints, winner, views):
        """Work out which printing is being held, trying each framing in turn.

        Quad detection is tuned to find the *name*, and a quad landing slightly
        inside the card border still reads that perfectly while cutting off the
        bottom line, which sits within a few percent of the edge. So when the
        view that won the title yields no answer, fall back through the other
        framings rather than concluding the card does not say.

        A framing counts as settled only once the reading actually resolves to
        a printing. Stopping as soon as any digits come back would let the
        splitting guesswork in parse_set_line end the search with noise.
        """
        ordered = [winner] + [v for v in views if v is not winner]
        sets = self.sf.set_index()
        for view in ordered[:SET_LINE_VIEWS]:
            chosen, evidence = pick_printing(
                prints, read_set_line(view.image(SET_LINE_SCALE)), sets)
            if evidence:
                return chosen, evidence
        return prints[0], ""

    def identify(self, frame):
        """Run the full pipeline on a single BGR frame."""
        views = [_View(frame, q) for q in find_card_quads(frame)]
        views.append(_View(frame, None))        # last resort: assume it fills the frame

        # Upright first for every candidate, then a second sweep upside down:
        # cards are almost always held the right way up, and each OCR pass is
        # the expensive part, so do not pay for rotation until it is needed.
        best = None            # (score, name, view, ocr_text)
        for view in views + [v.upside_down() for v in views]:
            for text in ocr_title(view.image()):
                m = self.match_name(text)
                if m and (best is None or m[1] > best[0]):
                    best = (m[1], m[0], view, text)
            if best and best[0] >= self.CONFIDENT:
                break                              # confident enough, stop
        if best is None:
            return None

        score, name, view, ocr_text = best
        prints = self.sf.printings(name)
        if not prints:
            return None

        # Which printing is in front of the camera, not which came first.
        chosen, evidence = self._resolve_printing(prints, view, views)

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
            set_evidence=evidence,
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
            note += (f"\nPrinting read off the card ({r.set_evidence})."
                     if r.set_evidence else
                     "\nCould not read the set line \u2014 showing the original "
                     f"printing ({r.original_set}).")
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
    if r.printings > 1:
        print(f"  Printing   : "
              + (f"read off the card ({r.set_evidence})" if r.set_evidence
                 else f"UNREADABLE - showing the original ({r.original_set})"))
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
