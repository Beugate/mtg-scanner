# Dokumentacija — MTG Card Scanner

Tehnička dokumentacija projekta. Za upute za instalaciju i korištenje pogledajte
[README.md](README.md).

---

## 1. Sažetak

Aplikacija prepoznaje *Magic: The Gathering* karte iz slike web kamere. Iz jedne
sličice videa određuje:

* **naziv karte** — čitanjem naslovne trake i približnim podudaranjem s
  katalogom svih naziva karata,
* **izdanje karte**
* **informacije o karti** tog izdanja — dohvatom
  sa Scryfall API-ja.


## 2. Tehnologije

| Biblioteka | Uloga |
| --- | --- |
| OpenCV (`opencv-python`) | detekcija karte, perspektivna transformacija, binarizacije |
| NumPy | rad s matricama slika |
| Tesseract OCR (`pytesseract`) | optičko prepoznavanje znakova |
| RapidFuzz | približno podudaranje naziva karata |
| Requests | komunikacija sa Scryfall API-jem |
| Pillow | prikaz slika u Tkinter sučelju |
| Tkinter | grafičko sučelje (dio standardne biblioteke) |

