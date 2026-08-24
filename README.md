# MTG Card Scanner

Hold a Magic: The Gathering card up to your webcam and get back its **name**,
**mana cost**, **which printing you are holding**, and that printing's
**current price**.

```
python mtg_scanner.py
```

![The scanner window](screenshot.png)

A window opens with the live camera feed. When a card is detected it gets a
green outline and is scanned automatically; you can also press **Space** or the
**Scan card** button. Results appear on the right, along with the card image
from Scryfall so you can confirm it read the right card.

## Requirements

Python 3.9+, plus [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
(already installed at `C:\Program Files\Tesseract-OCR` on this machine — the app
finds it there automatically, or set `TESSERACT_CMD` to point elsewhere).

```
pip install -r requirements.txt
```

Prices and card data come from the [Scryfall API](https://scryfall.com/docs/api).
An internet connection is needed for the first run; after that the card-name
catalog is cached locally for a week and printings for 12 hours.

## Options

```
python mtg_scanner.py                  # GUI, default camera
python mtg_scanner.py --camera 1       # a different webcam
python mtg_scanner.py --image card.jpg # identify a saved photo, print to stdout
```

The camera dropdown switches webcams without restarting. **Esc** quits.

## Getting good scans

* Fill a decent part of the frame with the card — roughly half the height.
* Even lighting; avoid glare across the title bar, which is the part being read.
* Sleeved cards are fine. Angled cards are fine, the card gets flattened first.
* Upside-down cards are handled, they just take a second longer.

## How it works

1. **Detect** the card as a quadrilateral in the frame. Five different
   binarisations are tried (adaptive threshold, auto-Canny on a contrast-
   equalised image, Otsu and its inverse, plain Canny) because a dark-bordered
   card on a dark table has almost no edge under one method and a clean
   silhouette under another. Candidates are ranked by how close they are to a
   card's 63:88 aspect ratio, not by size — the largest contour in a frame is
   usually the background.
2. **Flatten** it to a fixed 488x680 image with a perspective transform.
3. **Read** the title bar with Tesseract. Three vertical bands (frames put the
   name at slightly different heights) times four preprocessings (Otsu,
   adaptive, top-hat, plain greyscale) are stacked into a single tall image and
   OCR'd in one call — each `pytesseract` call spawns a subprocess, so batching
   the twelve crops took a scan from ~20s down to ~2s.
4. **Match** the resulting lines against every Scryfall card name with RapidFuzz.
5. **Look up** every printing of the matched card, then work out which one is
   actually in front of the camera from the small print along the bottom edge.
   That line is a fraction of the height of the title, so it gets its own warp
   of the source frame at 5x the canonical size — at the size the title is read
   at, the type is below anything Tesseract can resolve.

### About the answers it gives

The set and price shown are for **the printing you are holding**, read off the
bottom edge of the card. Which signals are there to read depends on how old the
frame is:

| Printed since | Signal | Example |
| --- | --- | --- |
| 2014 (M15 frame) | set code beside the language | `2XM • EN` |
| ~1998 | collector number and set size | `248/383` |
| ~1995 | copyright year | `™ & © 1993-2007 Wizards of the Coast` |

None of these is available everywhere, so all of them are scored together
rather than trusted one at a time. A set code is close to decisive where it is
printed at all; a collector number and set size together nearly so; either
alone is only suggestive; and the year is far too coarse to decide anything on
its own — every core set of an era shares a size, so `249` means M10 through
M13 until the year picks one out. A printing has to clear a threshold *and*
beat every other set before it is reported.

When the evidence does not get there, the panel falls back to the card's
original printing and says the set line was unreadable, rather than presenting
the fallback as a finding. Cards printed before about 1998 carry no collector
number at all, so they land here by design. This is the same instinct as the
name matching below: no answer beats a confident wrong one.

Matching is deliberately conservative: it would rather say *"no card
recognised"* and let you rescan than give a confident wrong answer. Two rules do
most of that work — a fuzzy match must keep most of the OCR text (so a bad read
of *Sylvan Library* cannot quietly become the card actually named *Library*),
and a short card name cannot be matched by a distinctly longer read.

## Tests

`test_scanner.py` runs the pipeline over the images in `testdata/` and checks
them against recorded ground truth.

```
python test_scanner.py           # all sets
python test_scanner.py val       # just one
```

Current results — 36/40 cards named correctly, **0 wrong answers**:

| Set | What it is | Result |
| --- | --- | --- |
| `scans` | flat card scans, no perspective | 5/5 |
| `photos` | synthetic webcam shots: perspective, noise, blur | 12/12 |
| `val` | harsher and unseen: rotation, small cards, lighting gradient, JPEG artefacts | 16/20 |
| `multiface` | double-faced and split cards | 3/3 |

The four `val` failures return "not recognised" rather than a wrong card. That
set is deliberately harder than a real webcam session — cards as little as half
the frame height, rotated up to 12°, at JPEG quality 72.

`printings` is scored separately, because "which printing is this" is a
different question from "which card is this". Every image in it is a reprint,
so always reporting the original printing would score zero:

| Result | Count |
| --- | --- |
| right set | 19/23 |
| set line unreadable, fell back and said so | 3 |
| **wrong set** | **0** |

Two of the three fall-backs are a 1995 and a 1997 card, which print no
collector number to read. Reading the set line costs roughly one extra second
per scan — it is a second full-size OCR pass, and the two overlapping crops it
uses are what keep the wrong-answer count at zero.

## Files

| File | Purpose |
| --- | --- |
| `mtg_scanner.py` | the whole application |
| `test_scanner.py` | regression tests |
| `testdata/` | test images and their ground truth |
| `screenshot.png` | the window, mid-scan |
| `cache/` | Scryfall responses and card images (git-ignored) |
