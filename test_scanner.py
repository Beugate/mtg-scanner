#!/usr/bin/env python3
"""Regression tests for mtg_scanner.

Runs the identification pipeline over the images in testdata/ and checks the
results against the recorded ground truth. The important number is not just
how many cards are recognised but how many are recognised *wrongly*: a
confident wrong answer is worse than "not recognised", because the user has no
signal to rescan.

    python test_scanner.py            # every set
    python test_scanner.py val        # one set
"""

import json
import sys
import time
from pathlib import Path

import cv2

import mtg_scanner as scanner

HERE = Path(__file__).resolve().parent

# Flat card scans, no perspective. The set the pipeline was developed against.
SCANS = {
    "testdata/Lightning_Bolt.jpg": "Lightning Bolt",
    "testdata/Sol_Ring.jpg": "Sol Ring",
    "testdata/Craterhoof_Behemoth.jpg": "Craterhoof Behemoth",
    "testdata/Snapcaster_Mage.jpg": "Snapcaster Mage",
    "testdata/Force_of_Will.jpg": "Force of Will",
}

# Double-faced and split cards, where the printed title is only half of the
# name Scryfall knows the card by.
MULTIFACE = {
    "testdata/dfc/Delver_of_Secrets.jpg": "Delver of Secrets // Insectile Aberration",
    "testdata/dfc/Fire_x_Ice.jpg": "Fire // Ice",
    "testdata/dfc/Brutal_Cathar.jpg": "Brutal Cathar // Moonrage Brute",
}


def load(name):
    if name == "scans":
        return SCANS
    if name == "multiface":
        return MULTIFACE
    path = HERE / "testdata" / name / "truth.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run(label, truth, ident):
    if not truth:
        print(f"{label}: no images, skipped")
        return 0, 0, 0
    correct = wrong = 0
    times = []
    for filename, expected in truth.items():
        frame = cv2.imread(str(HERE / filename))
        if frame is None:
            print(f"  ?? missing image {filename}")
            continue
        started = time.time()
        result = ident.identify(frame)
        times.append(time.time() - started)
        got = result.name if result else None
        if got == expected:
            correct += 1
        elif got is None:
            print(f"  -- not recognised: {expected}")
        else:
            wrong += 1
            print(f"  !! {expected}  ->  {got}   (read {result.ocr_text!r})")
    print(f"{label}: {correct}/{len(truth)} correct, {wrong} wrong, "
          f"mean {sum(times) / len(times):.1f}s, max {max(times):.1f}s")
    return correct, wrong, len(truth)


def run_printings(ident):
    """Does it report the printing being held, rather than the oldest one?

    Every image in this set is a reprint, so a scanner that always reported the
    original printing scores zero. Three outcomes are distinguished, because
    they are not equally bad: the right set, an unreadable set line (which
    falls back to the original and says so), and a wrong set stated as fact.
    Only the last one is a failure.
    """
    path = HERE / "testdata" / "printings" / "truth.json"
    if not path.exists():
        print("printings: no images, skipped")
        return 0, 0, 0
    truth = json.loads(path.read_text(encoding="utf-8"))

    correct = wrong = 0
    for filename, want in truth.items():
        frame = cv2.imread(str(HERE / filename))
        if frame is None:
            print(f"  ?? missing image {filename}")
            continue
        result = ident.identify(frame)
        if result is None or result.name != want["name"]:
            got = result.name if result else "nothing"
            print(f"  -- name not matched: {want['name']} -> {got}")
        elif result.set_code == want["set"]:
            correct += 1
        elif not result.set_evidence:
            print(f"  -- set line unreadable: {want['name']} "
                  f"({want['set']}, {want['released'][:4]})")
        else:
            wrong += 1
            print(f"  !! {want['name']}  {want['set']} -> {result.set_code}"
                  f"   (evidence {result.set_evidence!r})")
    print(f"printings: {correct}/{len(truth)} identified to the right set, "
          f"{wrong} wrong")
    return correct, wrong, len(truth)


def main():
    wanted = sys.argv[1:] or ["scans", "photos", "val", "multiface", "printings"]
    ident = scanner.Identifier(scanner.Scryfall())
    print("Loading Scryfall card-name catalog...")
    ident.load_catalog()
    if not ident.ready:
        print(f"Could not load the catalog: {ident.error}")
        return 1
    print(f"{len(ident.names):,} match keys ready\n")

    correct = wrong = total = 0
    set_wrong = 0
    for name in wanted:
        # Kept out of the totals below: this set measures which *printing* was
        # reported, which is a different question from which card it is.
        if name == "printings":
            _, set_wrong, _ = run_printings(ident)
            print()
            continue
        c, w, t = run(name, load(name), ident)
        correct += c
        wrong += w
        total += t
        print()

    print(f"TOTAL: {correct}/{total} correct, {wrong} wrong answers")
    return 1 if (wrong or set_wrong) else 0


if __name__ == "__main__":
    sys.exit(main())
