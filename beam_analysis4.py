"""
beam_analysis4.py
=================
Bildverarbeitungs- und Analysemodule für die Motorsteuerung.

Enthält:
  - bild_laden()              → Bild einlesen & als float32-Array zurückgeben
  - neuestes_bild()           → Neueste Datei in einem Ordner finden
  - berechne_strahlzentrum()  → Iterativer 3-Stufen-Zentrumsalgorithmus (2D grob → 1D feiner → 1D fein)
  - berechne_i0()             → Referenzintensität I₀ (5-Pixel-Mittelwert)
  - extract_profiles()        → Robuste 1D-Profile (X und Y) + finales Zentrum aus 2D-Bild
  - analyse_feldgroesse()     → Homo & Sym für fest vorgegebene Feldgröße
  - analyze_profile()         → Homogenität + Symmetrie aus 1D-Profil
  - untergrund_aus_rand()     → Untergrundwert aus Randpixeln schätzen

Zentrumsbestimmung
------------------
Alle Funktionen verwenden AUSSCHLIESSLICH das von extract_profiles() gelieferte
Zentrum (c_y, c_x). Es wird EINMALIG pro Bild in berechne_strahlzentrum()
berechnet und danach unverändert durch alle Analysefunktionen durchgereicht.
Keine Funktion außer berechne_strahlzentrum() und extract_profiles() darf
eine eigene Zentrums-Berechnung durchführen.

ABLAGEORT:
    Die Datei muss im GLEICHEN Ordner wie das Hauptskript liegen, z.B.:
        MeinProjekt/
        ├── motorsteuerung.py   ← Hauptskript
        └── beam_analysis4.py   ← dieses Modul
"""

import os
import glob
import warnings

import numpy as np

warnings.filterwarnings("ignore")

try:
    from scipy.ndimage import median_filter   # median_filter(input, size) → ndarray
    from PIL import Image                     # Image.open(pfad) → PIL.Image
except ImportError:
    raise ImportError(
        "[FEHLER] Bitte benötigte Pakete installieren: "
        "pip install scipy pillow"
    )


# ══════════════════════════════════════════════════════════════════════
#  1.  DATEI-HILFSFUNKTIONEN
# ══════════════════════════════════════════════════════════════════════

def neuestes_bild(bild_ordner: str, datei_endung: str = "*.png") -> str | None:
    """
    Gibt den Pfad des zuletzt geänderten Bildes im Ordner zurück.

    Parameter
    ---------
    bild_ordner  : Pfad zum Suchordner (z.B. "C:/Daten/Bilder")
    datei_endung : Glob-Muster für Dateityp, z.B. "*.png" oder "*.tif"

    Rückgabe
    --------
    str  → absoluter Dateipfad des neuesten Bildes
    None → kein passendes Bild gefunden

    Hinweis: Dateien deren Name "_profil_" enthält werden übersprungen
             (Profil-Exportdateien der BeamAnalyzer-App).
    """
    alle_dateien = glob.glob(os.path.join(bild_ordner, datei_endung))
    alle_dateien = [d for d in alle_dateien if "_profil_" not in os.path.basename(d)]
    return max(alle_dateien, key=os.path.getmtime) if alle_dateien else None


# ══════════════════════════════════════════════════════════════════════
#  2.  BILD LADEN
# ══════════════════════════════════════════════════════════════════════

def bild_laden(bild_pfad: str) -> np.ndarray | None:
    """
    Lädt ein 16-Bit-Graustufenbild und gibt es als float32-Array zurück.

    Unterstützte Formate / Farbtiefen
    ----------------------------------
    - 16-Bit-Graustufen  ("I;16", "I")  → Rohdaten als float32
    - Alle anderen Formate               → Fehlermeldung, gibt None zurück

    Parameter
    ---------
    bild_pfad : Dateipfad zum Bild (z.B. "C:/Daten/aufnahme.png")

    Rückgabe
    --------
    np.ndarray  shape=(Hoehe, Breite), dtype=float32
    None        bei Lesefehler oder falschem Bildformat
    """
    try:
        bild       = Image.open(bild_pfad)
        bild_modus = bild.mode

        if bild_modus in ("I;16", "I"):
            bild_arr = np.array(bild, dtype=np.float32)
        else:
            print("[Fehler] Bild enthält nicht nur Grauwerte:")
            return None

        print(
            f"[INFO] Bild geladen: {os.path.basename(bild_pfad)}  "
            f"shape={bild_arr.shape}  max={bild_arr.max():.4f}"
        )
        return bild_arr

    except Exception as fehler:
        print(f"[WARNUNG] Bild konnte nicht geladen werden ({bild_pfad}): {fehler}")
        return None


# ══════════════════════════════════════════════════════════════════════
#  3.  STRAHLZENTRUM  (einzige autoritative Zentrums-Berechnung)
# ══════════════════════════════════════════════════════════════════════

def berechne_strahlzentrum(bild_arr: np.ndarray) -> tuple[int, int]:
    """
    Bestimmt das Strahlzentrum in drei Verfeinerungsstufen.

    Das Ergebnis (c_y, c_x) ist das EINZIGE Zentrum das im gesamten
    Analyse-Pipeline verwendet wird. Alle anderen Funktionen erhalten
    es als Parameter – keine eigene Zentrums-Berechnung ist erlaubt.

    Algorithmus
    -----------
    Stufe 1 – Grob (2D-Schwerpunkt)
        Alle Pixel ≥ 50 % des Bildmaximums → Schwerpunkt (y_g, x_g).
        Fallback: Bildmitte.

    Stufe 2 – Feiner (1D via grobem Zentrum)
        X-Profil: Zeile y_g, Median-Filter k=5 → 1D-Profil.
        Y-Profil: Spalte x_g, Median-Filter k=5 → 1D-Profil.
        Zentrum je Achse: Mittelpunkt der 50%-Zone + Avg der 5 zentralen Pixel.
        Ergibt (y1, x1).

    Stufe 3 – Fein (1D via verfeinertem Zentrum)
        Identisch zu Stufe 2, aber ausgehend von (y1, x1).
        Ergibt finales (c_y, c_x).

    Parameter
    ---------
    bild_arr : 2D float-Array (Hoehe × Breite), z.B. aus bild_laden()

    Rückgabe
    --------
    (c_y, c_x) : finales Strahlzentrum in Pixeln (Zeile, Spalte)
    """
    bild_hoehe, bild_breite = bild_arr.shape

    # ── Stufe 1: Grob – 2D-Schwerpunkt aller Pixel ≥ 50 % Max ──────────────
    schwelle = bild_arr.max() * 0.5
    y_idx, x_idx = np.where(bild_arr >= schwelle)

    if len(y_idx) == 0:
        y_g, x_g = bild_hoehe // 2, bild_breite // 2  # Fallback: Bildmitte
    else:
        y_g = int(np.round(np.mean(y_idx)))
        x_g = int(np.round(np.mean(x_idx)))

    # ── Hilfsfunktion: 1D-Profil durch einen Punkt → verfeinertes Zentrum ───
    def _verfeinere_zentrum_1d(profil_1d: np.ndarray, start: int) -> int:
        """
        Median-Filter (k=5) auf das Rohprofil, dann Mittelpunkt der 50%-Zone,
        dann Mittelwert der 5 Zentralpixel als sub-pixel-genaue Korrektur.

        Parameter
        ---------
        profil_1d : 1D float-Array (eine Zeile oder Spalte des Bildes)
        start     : vorheriger Schätzwert (wird als Fallback verwendet)

        Rückgabe
        --------
        int : verfeinerter Zentrumsindex
        """
        gefiltert = median_filter(profil_1d.astype(np.float64), size=5)

        if gefiltert.max() == 0:
            return start  # kein Signal → Fallback

        # Mittelpunkt der 50%-Zone
        idx_50 = np.where(gefiltert >= gefiltert.max() * 0.5)[0]
        if len(idx_50) == 0:
            return start
        mitte = int((idx_50[0] + idx_50[-1]) // 2)

        # Avg der 5 Zentralpixel → sub-pixel Schwerpunkt → ganzzahliges Zentrum
        lo = max(0, mitte - 2)
        hi = min(len(gefiltert), mitte + 3)
        fenster = gefiltert[lo:hi]
        positionen = np.arange(lo, hi, dtype=np.float64)
        gewichte   = fenster - fenster.min()  # auf 0 normieren bevor Schwerpunkt
        if gewichte.sum() == 0:
            return mitte
        return int(np.round(np.sum(positionen * gewichte) / gewichte.sum()))

    # ── Stufe 2: Feiner – 1D-Profile durch grobes Zentrum ───────────────────
    zeile_grob  = np.array(bild_arr[y_g, :], dtype=np.float64)   # X-Profil durch y_g
    spalte_grob = np.array(bild_arr[:, x_g], dtype=np.float64)   # Y-Profil durch x_g

    x1 = _verfeinere_zentrum_1d(zeile_grob,  x_g)   # verfeinertes X-Zentrum
    y1 = _verfeinere_zentrum_1d(spalte_grob, y_g)   # verfeinertes Y-Zentrum

    # Grenzen sicherstellen
    y1 = max(0, min(bild_hoehe - 1, y1))
    x1 = max(0, min(bild_breite - 1, x1))

    # ── Stufe 3: Fein – 1D-Profile durch verfeinertes Zentrum ───────────────
    zeile_fein  = np.array(bild_arr[y1, :], dtype=np.float64)    # X-Profil durch y1
    spalte_fein = np.array(bild_arr[:, x1], dtype=np.float64)    # Y-Profil durch x1

    c_x = _verfeinere_zentrum_1d(zeile_fein,  x1)
    c_y = _verfeinere_zentrum_1d(spalte_fein, y1)

    # Grenzen sicherstellen
    c_y = max(0, min(bild_hoehe - 1, c_y))
    c_x = max(0, min(bild_breite - 1, c_x))

    return c_y, c_x


# ══════════════════════════════════════════════════════════════════════
#  4.  REFERENZINTENSITÄT
# ══════════════════════════════════════════════════════════════════════

def berechne_i0(profil_serie: np.ndarray, zentrum_px: int) -> float:
    """
    Berechnet die Referenzintensität I₀ als Mittelwert der 5 Zentralpixel.

    Parameter
    ---------
    profil_serie : 1D float-Array (untergrundkorrigiertes Profil)
    zentrum_px   : Zentrumsindex [px] – IMMER aus extract_profiles() übernehmen

    Rückgabe
    --------
    float : I₀ (Mittelwert der Pixel zentrum_px-2 … zentrum_px+2)
    """
    lo = max(0, zentrum_px - 2)
    hi = min(len(profil_serie), zentrum_px + 3)
    return float(np.mean(profil_serie[lo:hi]))


# ══════════════════════════════════════════════════════════════════════
#  5.  PROFILE EXTRAHIEREN
# ══════════════════════════════════════════════════════════════════════

def extract_profiles(
    bild_arr: np.ndarray,
    n_linien: int = 5,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Bestimmt das Strahlzentrum und erzeugt robuste 1D-Profile (X und Y).

    Das Zentrum wird EINMALIG via berechne_strahlzentrum() (3-Stufen-Algorithmus)
    bestimmt und dann für die Profilextraktion verwendet. Alle nachgelagerten
    Analyse-Funktionen erhalten (c_y, c_x) als Parameter – keine eigene
    Zentrums-Berechnung findet mehr statt.

    Vorgehen
    --------
    1. berechne_strahlzentrum() → finales (c_y, c_x)
    2. X-Profil:
         Streifen c_y ± n_linien (= 2*n_linien+1 Zeilen).
         Jede Zeile: Median-Filter (k=5) → Rauschunterdrückung.
         Spaltenweiser Mittelwert über alle gefilterten Zeilen.
    3. Y-Profil: analog mit Spalten um c_x (transponiert behandelt).

    Parameter
    ---------
    bild_arr : float-Array (Hoehe × Breite), Werte typisch 0–65535 bei 16-Bit
    n_linien : Halbbreite des Mittelungsstreifens (Standard 5 → 11 Linien gesamt)

    Rückgabe
    --------
    profil_x : 1D-Array der Länge Breite  (horizontales Strahlprofil)
    profil_y : 1D-Array der Länge Hoehe   (vertikales  Strahlprofil)
    c_y      : Strahlzentrum Zeile  [px]  – für alle Folgerechnungen verwenden
    c_x      : Strahlzentrum Spalte [px]  – für alle Folgerechnungen verwenden
    """
    bild_hoehe, bild_breite = bild_arr.shape

    # ── Einmalige Zentrums-Bestimmung (3 Stufen) ────────────────────────────
    c_y, c_x = berechne_strahlzentrum(bild_arr)

    # ── Hilfsfunktion: Streifen → Median-gefiltert + spaltenweise gemittelt ──
    def _streifen_profil(streifen: np.ndarray) -> np.ndarray:
        """
        Median-Filter (k=5) auf jede Zeile, dann spaltenweiser Mittelwert.

        Parameter
        ---------
        streifen : 2D-Array (n_zeilen × Breite)

        Rückgabe
        --------
        1D-Array der Länge Breite → robustes gemitteltes Profil
        """
        gefiltert = np.stack(
            [median_filter(zeile.astype(np.float64), size=5) for zeile in streifen]
        )                              # shape: (n_zeilen, Breite)
        return gefiltert.mean(axis=0) # spaltenweise mitteln → 1D

    # ── X-Profil: Zeilen um c_y ─────────────────────────────────────────────
    z_unten = max(0, c_y - n_linien)
    z_oben  = min(bild_hoehe, c_y + n_linien + 1)
    profil_x = _streifen_profil(bild_arr[z_unten:z_oben, :])

    # ── Y-Profil: Spalten um c_x (transponiert → gleiche Hilfsfunktion) ─────
    s_links  = max(0, c_x - n_linien)
    s_rechts = min(bild_breite, c_x + n_linien + 1)
    profil_y = _streifen_profil(bild_arr[:, s_links:s_rechts].T)

    return profil_x, profil_y, c_y, c_x


# ══════════════════════════════════════════════════════════════════════
#  6.  UNTERGRUNDKORREKTUR
# ══════════════════════════════════════════════════════════════════════

def untergrund_aus_rand(bild_arr: np.ndarray, rand_breite: int) -> float:
    """
    Schätzt den Untergrundwert aus den Randpixeln des Bildes.

    Alle vier Ränder (oben, unten, links, rechts) mit der Breite
    `rand_breite` werden ausgeschnitten, zusammengefügt und gemittelt.

    Parameter
    ---------
    bild_arr    : 2D-Array (Hoehe × Breite), z.B. aus bild_laden()
    rand_breite : Breite des Randstreifens in Pixeln (z.B. 5)

    Rückgabe
    --------
    float : Mittelwert aller Randpixel (= geschätzter Untergrundwert)

    Fehler
    ------
    ValueError : wenn rand_breite zu groß für das Bild ist
    """
    bild_hoehe, bild_breite = bild_arr.shape[:2]

    if rand_breite * 2 >= bild_hoehe or rand_breite * 2 >= bild_breite:
        raise ValueError("rand_breite ist zu groß für die Bilddimensionen.")

    rand_oben   = bild_arr[0:rand_breite, :]
    rand_unten  = bild_arr[bild_hoehe - rand_breite:bild_hoehe, :]
    rand_links  = bild_arr[:, 0:rand_breite]
    rand_rechts = bild_arr[:, bild_breite - rand_breite:bild_breite]

    alle_randpixel = np.concatenate([
        rand_oben.flatten(),
        rand_unten.flatten(),
        rand_links.flatten(),
        rand_rechts.flatten()
    ])
    return float(np.mean(alle_randpixel))


# ══════════════════════════════════════════════════════════════════════
#  7.  ANALYSE FÜR FESTE FELDGRÖSSE
# ══════════════════════════════════════════════════════════════════════

def analyse_feldgroesse(
    profil_serie: np.ndarray,
    zentrum_px: int,
    feld_halb_px: int,
    pixel_mm: float,
    untergrund_wert: float = 0.0,
) -> dict:
    """
    Berechnet Homogenität und Symmetrie für eine FEST vorgegebene Feldgröße.

    Das Fenster ist fix: ±feld_halb_px um das übergebene zentrum_px.
    KEINE eigene Zentrums-Berechnung – zentrum_px kommt immer aus extract_profiles().

    Untergrundkorrektur
    -------------------
    untergrund_wert wird intern abgezogen, negative Werte auf 0 geclippt.
    Die Rohdaten bleiben unverändert.

    Berechnungen
    ------------
    I₀          : Mittelwert der 5 Pixel um zentrum_px (nach Untergrundabzug)
    Homogenität : (max - min) / I₀ × 100  [%]  im Fenster [l … r]
    Symmetrie   : Σ(rechts[k] - links[k]) / n_sym × 100 / I₀  [%]

    Parameter
    ---------
    profil_serie    : 1D float-Array (rohes X- oder Y-Profil aus extract_profiles)
    zentrum_px      : Strahlzentrum [px] – AUS extract_profiles() übernehmen, c_x oder c_y
    feld_halb_px    : Halbbreite des Feldes in Pixeln (= feld_mm / 2 / pixel_mm)
    pixel_mm        : Umrechnungsfaktor Pixel → mm
    untergrund_wert : Untergrundwert (z.B. aus untergrund_aus_rand()), Standard 0.0

    Rückgabe
    --------
    dict mit Schlüsseln:
        homo      – Homogenität im Fenster [%]  (nan wenn I₀=0 oder Feld zu klein)
        sym       – Symmetrie im Fenster [%]    (0.0 wenn n_sym=0)
        l         – linker Feldrand [px]
        r         – rechter Feldrand [px]
        feld_mm   – tatsächliche Feldbreite [mm]
        zentrum_px– übergebenes (und verwendetes) Zentrum [px]
    """
    # ── Untergrundkorrektur ──────────────────────────────────────────────────
    profil_korr = np.clip(
        np.asarray(profil_serie, dtype=np.float64) - untergrund_wert, 0.0, None
    )

    # ── I₀ am übergebenen Zentrum ────────────────────────────────────────────
    i0 = berechne_i0(profil_korr, zentrum_px)

    # ── Fenster auf Bildgrenzen beschneiden ──────────────────────────────────
    feld_links  = max(0, zentrum_px - feld_halb_px)
    feld_rechts = min(len(profil_korr) - 1, zentrum_px + feld_halb_px)
    feld_werte  = profil_korr[feld_links : feld_rechts + 1]

    if i0 == 0 or len(feld_werte) < 3:
        return dict(homo=np.nan, sym=np.nan,
                    l=feld_links, r=feld_rechts,
                    feld_mm=feld_halb_px * 2 * pixel_mm,
                    zentrum_px=zentrum_px)

    homo = (feld_werte.max() - feld_werte.min()) / i0 * 100.0

    # ── Symmetrie ────────────────────────────────────────────────────────────
    n_sym = min(feld_halb_px, zentrum_px - feld_links, feld_rechts - zentrum_px)
    sym = (
        sum(profil_korr[zentrum_px + k] - profil_korr[zentrum_px - k]
            for k in range(1, n_sym + 1))
        / n_sym * 100.0 / i0
    ) if n_sym else 0.0

    return dict(homo=homo, sym=sym,
                l=feld_links, r=feld_rechts,
                feld_mm=feld_halb_px * 2 * pixel_mm,
                zentrum_px=zentrum_px)


# ══════════════════════════════════════════════════════════════════════
#  8.  PROFIL ANALYSIEREN  (Homogenität + Symmetrie, adaptive Feldsuche)
# ══════════════════════════════════════════════════════════════════════

def analyze_profile(
    profil_serie: np.ndarray,
    ziel_homogenitaet: float,
    pixel_mm: float,
    untergrund_wert: float,
    zentrum_px: int,
) -> dict | None:
    """
    Berechnet Homogenität und Symmetrie für ein 1D-Intensitätsprofil.

    Untergrundkorrektur
    -------------------
    untergrund_wert wird intern abgezogen, negative Werte auf 0 geclippt.
    Die Rohdaten aus extract_profiles() bleiben unverändert.

    Algorithmus
    -----------
    Schritt 1 – I₀ (Referenzintensität)
        Mittelwert der 5 Pixel um zentrum_px (untergrundkorrigiert).

    Schritt 2 – Feldbreite & Homogenität
        Läuft von k=1 aufwärts (symmetrische Halbbreite).
        Homogenität im Fenster [zentrum-k … zentrum+k]:
            homo = (max - min) / I₀ × 100  [%]
        Bricht beim ERSTEN Überschreiten von ziel_homogenitaet ab.
        Feldbreite = k_max * 2 * pixel_mm  [mm]

    Schritt 3 – Symmetrie
        Symmetrieradius = min(zentrum_px, len-1-zentrum_px, k_max).
        sym = Σ(rechts[k] − links[k]) / (sym_radius × I₀) × 100  [%]

    Parameter
    ---------
    profil_serie      : 1D float-Array (X- oder Y-Profil aus extract_profiles)
    ziel_homogenitaet : Ziel-Homogenität in % (z.B. 10.0)
    pixel_mm          : Umrechnungsfaktor Pixel → mm (z.B. 0.0588)
    untergrund_wert   : Untergrundwert (z.B. aus untergrund_aus_rand())
    zentrum_px        : Strahlzentrum [px] – AUS extract_profiles() übernehmen, c_x oder c_y

    Rückgabe
    --------
    dict mit Schlüsseln:
        zentrum_px   – übergebenes Zentrum [px]
        homogenitaet – Homogenität im max. gültigen Feld [%]
        breite_mm    – Feldbreite bei homo ≤ ziel_homogenitaet [mm]
        symmetrie    – Symmetrie [%], positiv = rechts/unten stärker
        sym_radius   – Symmetrieradius [px]  (für Diagnose)
    None  → Signal nach Untergrundabzug null.

    Hinweis zum Vorzeichen der Symmetrie
    -------------------------------------
    Positiv = rechte/untere Seite intensiver als linke/obere Seite.
    Im Hauptskript wird das Vorzeichen mit kalibrierungsrichtung_x/y
    multipliziert, um die physikalische Verfahrrichtung anzupassen.
    """
    # ── Untergrundkorrektur ──────────────────────────────────────────────────
    if profil_serie.max() == 0:
        return None

    profil_korr = np.clip(profil_serie.astype(np.float64) - untergrund_wert, 0.0, None)

    if profil_korr.max() == 0:
        return None

    # ── I₀: Referenzintensität am übergebenen Zentrum ────────────────────────
    i0 = berechne_i0(profil_korr, zentrum_px)

    if i0 == 0:
        return None

    # ── Maximale Feldbreite bei Homogenität ≤ ziel_homogenitaet % ────────────
    k_max = 0

    for k in range(1, len(profil_korr)):
        idx_links  = zentrum_px - k
        idx_rechts = zentrum_px + k

        if idx_links < 0 or idx_rechts >= len(profil_korr):
            break

        fenster      = profil_korr[idx_links : idx_rechts + 1]
        homo_aktuell = (fenster.max() - fenster.min()) / i0 * 100.0

        if homo_aktuell <= ziel_homogenitaet:
            k_max = k
        else:
            break

    # Homogenität im finalen Fenster
    feld_links  = max(0, zentrum_px - k_max)
    feld_rechts = min(len(profil_korr) - 1, zentrum_px + k_max)
    feld_werte  = profil_korr[feld_links : feld_rechts + 1]
    homo_final  = (
        (feld_werte.max() - feld_werte.min()) / i0 * 100.0
        if k_max else float("nan")
    )

    # ── FWHM-Radius: welche Seite fällt zuerst unter 50% i0 ─────────────────
    halb = 0.5 * i0

    r_rechts = 0
    for i in range(1, len(profil_korr) - zentrum_px):
        if profil_korr[zentrum_px + i] >= halb:
            r_rechts = i
        else:
            break

    r_links = 0
    for i in range(1, zentrum_px + 1):
        if profil_korr[zentrum_px - i] >= halb:
            r_links = i
        else:
            break

    # Die Seite die früher abbricht bestimmt den Radius
    sym_radius = min(r_rechts, r_links)

    # ── Symmetrie: Integration von links nach rechts im Fenster ──────────────
    if sym_radius <= 0:
        symmetrie = 0.0
    else:
        fenster       = profil_korr[zentrum_px - sym_radius : zentrum_px + sym_radius + 1]
        mitte_idx     = sym_radius  # Index des Zentrums im Fenster
        seite_links   = fenster[:mitte_idx]          # links  → Mitte
        seite_rechts  = fenster[mitte_idx + 1:]      # Mitte  → rechts
        sym_summe     = np.sum(seite_rechts - seite_links[::-1])
        symmetrie     = float(sym_summe * 100.0 / (sym_radius * i0))

    return {
        "zentrum_px":   zentrum_px,
        "homogenitaet": homo_final,
        "breite_mm":    k_max * 2 * pixel_mm,
        "symmetrie":    symmetrie,
        "sym_radius":   int(sym_radius),
    }
