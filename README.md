# MTG Card Scanner

Prislonite kartu pred kamerom i očitati će bitne podatke o karti

```
python mtg_scanner.py
```

![Prozor skenera](screenshot.png)

Otvara se prozor sa slikom kamere uživo. Kada se karta prepozna, dobiva zeleni
obrub i automatski se skenira; skeniranje možete pokrenuti i tipkom **Space**
ili gumbom **Scan card**. Rezultati se prikazuju s desne strane, zajedno sa
slikom karte sa Scryfalla, kako biste mogli potvrditi da je točna
karta.

## Preduvjeti

Python 3.9+ i [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki).

Aplikacija sama pronalazi Tesseract: redom provjerava varijablu okoline
`TESSERACT_CMD`, zatim `tesseract` na `PATH`-u, pa uobičajene lokacije
instalacije (`C:\Program Files\Tesseract-OCR\tesseract.exe` na Windowsu,
`/usr/bin/tesseract` i `/usr/local/bin/tesseract` na Linuxu). Ako je Tesseract
negdje drugdje, postavite `TESSERACT_CMD`.

```
pip install -r requirements.txt
```

Podaci o kartama i cijene dolaze sa [Scryfall API-ja](https://scryfall.com/docs/api).
Za prvo pokretanje potrebna je internetska veza; nakon toga se katalog naziva
karata lokalno sprema tjedan dana, a popisi izdanja 12 sati.

## Opcije

```
python mtg_scanner.py                  # GUI, zadana kamera
python mtg_scanner.py --camera 1       # druga web kamera
python mtg_scanner.py --image karta.jpg  # prepoznavanje spremljene slike, ispis u stdout
```

Padajući izbornik s kamerama omogućuje prebacivanje između web kamera bez
ponovnog pokretanja. **Esc** zatvara aplikaciju.



## Kako radi

1. **Detekcija** karte kao četverokuta u sličici. Isprobava se pet različitih
   binarizacija (adaptivni prag, automatski Canny na slici s izjednačenim
   kontrastom, Otsu i njegov inverz te obični Canny), jer karta s tamnim
   obrubom na tamnom stolu pod jednom metodom gotovo da nema rub, a pod drugom
   daje čistu siluetu. Kandidati se rangiraju po tome koliko su blizu omjeru
   stranica karte 63:88, a ne po veličini — najveća kontura u sličici obično je
   pozadina.
2. **Poravnanje** na sliku fiksne veličine 488x680 perspektivnom
   transformacijom.
3. **Čitanje** naslovne trake Tesseractom. Tri okomita pojasa (različiti okviri
   karata stavljaju naziv na malo različite visine) puta četiri predobrade
   (Otsu, adaptivni prag, top-hat, obična sivkasta slika) slažu se u jednu
   visoku sliku i OCR-aju u jednom pozivu — svaki `pytesseract` poziv pokreće
   podproces, pa je grupiranje dvanaest isječaka spustilo trajanje skena s
   ~20 s na ~2 s.
4. **Podudaranje** dobivenih redaka sa svim nazivima karata sa Scryfalla pomoću
   biblioteke RapidFuzz.
5. **Dohvat** svih izdanja pronađene karte, pa utvrđivanje koje se od njih
   zapravo nalazi pred kamerom, na temelju sitnog teksta uz donji rub karte.
   Taj je redak višestruko manji od naslova, pa dobiva vlastito poravnanje
   izvorne sličice na 5x kanonsku veličinu — pri veličini na kojoj se čita
   naslov, ta su slova ispod svega što Tesseract može razlučiti.


## Datoteke

| Datoteka | Namjena |
| --- | --- |
| `mtg_scanner.py` | cijela aplikacija |
| `test_scanner.py` | regresijski testovi |
| `testdata/` | testne slike i njihovi očekivani rezultati |
| `screenshot.png` | prozor aplikacije, usred skeniranja |
| `cache/` | Scryfall odgovori i slike karata (izuzeto iz gita) |

