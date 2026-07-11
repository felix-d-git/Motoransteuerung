import sys
import os
import glob
import warnings

import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # TkAgg = Matplotlib-Backend das ein Tkinter-Fenster öffnet
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Analyse-Modul (beam_analysis4.py muss im selben Ordner liegen) ──
from beam_analysis4 import (
    neuestes_bild,       # bild_ordner, datei_endung → neuester Bildpfad oder None
    bild_laden,          # bild_pfad → float32-Array oder None
    extract_profiles,    # bild_arr, n_linien → (profil_x, profil_y, c_y, c_x)
    analyse_feldgroesse, # profil, zentrum_px, feld_halb_px, pixel_mm, untergrund → dict
    analyze_profile,     # profil, ziel_homo, pixel_mm, untergrund, zentrum_px → dict
    untergrund_aus_rand, # bild_arr, rand_breite → float
)

# ── Config-Modul (config_utils.py muss im selben Ordner liegen) ──
from config_utils import basis_ordner_ermitteln, lade_config, speichere_config

# ──────────────────────────────────────────────────────────────
#  KONFIGURATION
# ──────────────────────────────────────────────────────────────
basis_ordner = basis_ordner_ermitteln()  # EXE-Ordner oder Skript-Ordner

DEFAULT_CONFIG = {
    # ── Beam-Analyzer-spezifisch ──────────────────────────────────────────────
    "bild_ordner":                  str(basis_ordner / "data"),  # Ordner für Kamerabilder
    "feld_groesse_mm":              20.0,    # Feste Feldgröße für Homo-Anzeige [mm]
    # ── Gemeinsame Keys (identisch mit serial_slider) ─────────────────────────
    "pixel_mm":                     0.0588,  # Umrechnungsfaktor Pixel → mm
    "ziel_homogenitaet":            10.0,    # Ziel-Homogenität in % für adaptive Feldbreitensuche
    "datei_endung":                 "*.png", # Glob-Muster für Bildtyp
    "rand_breite_untergrund":       5,       # Randbreite [px] für untergrund_aus_rand()
    # ── Serial-Slider-Keys (werden mitgespeichert, hier als Defaults) ─────────
    "serieller_port":               "COM9",
    "baudrate":                     115200,
    "fenster_titel":                "Motorsteuerung",
    "geschwindigkeit_max":          28000,
    "schritt_werte":                [-10, -2, -0.2, -0.02, -0.01, 0, 0.01, 0.02, 0.2, 2, 10],
    "kalibrierungsgeschwindigkeit": 1000,
    "kalibrierungs_bildpfad":       str(basis_ordner),
    "mm_pro_prozent_zweiring":      0.066,
    "mm_pro_prozent_vierring":      0.214,
    "sym_toleranz_prozent":         0.2,
    "streuer_modus":                4,
    "kalibrierungsrichtung_x":      -1,
    "kalibrierungsrichtung_y":      +1,
}


# ──────────────────────────────────────────────────────────────
#  SPEICHERN MIT NUMMERIERUNG
# ──────────────────────────────────────────────────────────────

def naechste_speichernummer(bild_ordner: str) -> int:
    """
    Gibt die nächste freie Dateinummer für _profil_NNN.png zurück.

    Durchsucht den Ordner nach Profil-Dateien (Muster: *_profil_NNN.png)
    und gibt max(vorhandene Nummer) + 1 zurück.

    Parameter
    ---------
    bild_ordner : Ordner, der durchsucht werden soll

    Rückgabe
    --------
    int : nächste freie Dateinummer (startet bei 1 wenn noch keine vorhanden)
    """
    # glob.glob() → alle vorhandenen Profil-Dateien finden
    vorhandene = glob.glob(os.path.join(bild_ordner, "*_profil_*.png"))

    nummern = []
    for pfad in vorhandene:
        datei_basis = os.path.basename(pfad)                   # nur Dateiname, ohne Ordner
        teile = datei_basis.rsplit("_profil_", 1)              # am letzten "_profil_" aufteilen
        if len(teile) == 2:
            try:
                # Zahl aus dem hinteren Teil extrahieren, ".png" entfernen
                nummern.append(int(teile[1].replace(".png", "")))
            except ValueError:
                pass  # Nicht-numerischer Suffix → ignorieren

    # max(..., default=0) → gibt 0 zurück wenn die Liste leer ist → erste Nummer = 1
    return max(nummern, default=0) + 1


# ──────────────────────────────────────────────────────────────
#  HAUPT-APP
# ──────────────────────────────────────────────────────────────

class BeamAnalyzerApp:
    """
    Matplotlib-basierte GUI zur Analyse von Strahlprofilen.

    Lädt automatisch das neueste Bild aus dem konfigurierten Ordner,
    berechnet X- und Y-Intensitätsprofile und zeigt Homogenität sowie
    Symmetrie für zwei Analysemodi an:
      - Feste Feldgröße (TextBox)  →  analyse_feldgroesse()
      - Ziel-Homogenität (TextBox) →  analyze_profile()
    """

    def __init__(self):
        self.cfg       = lade_config(DEFAULT_CONFIG, basis_ordner)  # Konfiguration aus config.json laden
        self.bild_pfad = None   # Pfad des aktuell geladenen Bildes
        self.bild_arr  = None   # 2D float-Array des Bildes
        self.profil_x  = None   # 1D X-Profil (horizontal, aus extract_profiles)
        self.profil_y  = None   # 1D Y-Profil (vertikal,   aus extract_profiles)
        self.c_y       = 0      # Y-Koordinate des Strahlzentrums [px]
        self.c_x       = 0      # X-Koordinate des Strahlzentrums [px]
        self._letztes_ergebnis = {}  # Zwischenspeicher der letzten Berechnung für _plot()

        self._erstelle_fenster()   # GUI-Elemente aufbauen
        self._laden()              # Direkt beim Start das neueste Bild laden
        plt.show()                 # Matplotlib-Ereignisschleife starten (blockiert bis Fenster zu)

    # ── Figure erstellen ─────────────────────────────────────

    def _erstelle_fenster(self):
        """Erstellt das matplotlib-Fenster mit allen Buttons und Textboxen."""

        # plt.subplots() → Figure + Axes in einem Schritt erstellen
        # figsize=(13, 7.5) → Fenstergröße in Zoll
        self.fig, self.ax = plt.subplots(figsize=(13, 7.5))

        # subplots_adjust() → Abstände der Plot-Fläche vom Fensterrand (0–1)
        self.fig.subplots_adjust(left=0.07, right=0.97, top=0.91, bottom=0.22)
        self.fig.canvas.manager.set_window_title("Beam Profile Analyzer")

        # ── Buttons & Textboxen (unten) ──────────────────────

        # fig.add_axes([links, unten, breite, hoehe]) → Axes an absoluter Position (0–1)
        ax_btn_laden = self.fig.add_axes([0.02, 0.05, 0.09, 0.06])
        self.btn_laden = Button(ax_btn_laden, "Neu laden", color="#dddddd", hovercolor="#aaaaaa")
        # lambda e: → anonyme Funktion die den Button-Event ignoriert und _laden() aufruft
        self.btn_laden.on_clicked(lambda e: self._laden())

        ax_btn_speichern = self.fig.add_axes([0.12, 0.05, 0.09, 0.06])
        self.btn_speichern = Button(ax_btn_speichern, "Speichern", color="#dddddd", hovercolor="#aaaaaa")
        self.btn_speichern.on_clicked(lambda e: self._speichern())

        # Label für Feldgröße-Eingabe
        ax_lbl_feld = self.fig.add_axes([0.23, 0.09, 0.11, 0.03])
        ax_lbl_feld.axis("off")  # axis("off") → keine Achsenlinien, nur Text anzeigen
        ax_lbl_feld.text(0, 0.5, "Feldgröße [mm]:", fontsize=9, va="center")

        # TextBox: Eingabefeld für die feste Feldgröße [mm]
        ax_tb_feld = self.fig.add_axes([0.23, 0.05, 0.09, 0.045])
        self.tb_feld = TextBox(ax_tb_feld, "", initial=str(self.cfg["feld_groesse_mm"]))
        # on_submit → Callback wenn der Benutzer Enter drückt oder das Feld verlässt
        self.tb_feld.on_submit(lambda txt: self._neuberechnen())

        # Label für Ziel-Homogenität-Eingabe
        ax_lbl_homo = self.fig.add_axes([0.34, 0.09, 0.13, 0.03])
        ax_lbl_homo.axis("off")
        ax_lbl_homo.text(0, 0.5, "Ziel-Homogenität [%]:", fontsize=9, va="center")

        # TextBox: Eingabefeld für die Ziel-Homogenität [%]
        ax_tb_homo = self.fig.add_axes([0.34, 0.05, 0.09, 0.045])
        self.tb_homo = TextBox(ax_tb_homo, "", initial=str(self.cfg["ziel_homogenitaet"]))
        self.tb_homo.on_submit(lambda txt: self._neuberechnen())

        # Infobox für Symmetrieergebnisse (unten Mitte)
        self.ax_sym_box = self.fig.add_axes([0.45, 0.03, 0.22, 0.09])
        self.ax_sym_box.axis("off")
        self.lbl_sym_box = self.ax_sym_box.text(
            0, 0.5, "", fontsize=9, va="center", ha="left",
            family="monospace",   # monospace = gleichbreite Schrift für übersichtliche Spalten
            bbox=dict(facecolor="#fcfcfc", alpha=1.0, edgecolor="#bbbbbb", boxstyle="round,pad=0.4")
        )

        # Status-Text (Dateiname + Fehlermeldungen), ganz rechts unten
        self.ax_status = self.fig.add_axes([0.60, 0.04, 0.29, 0.07])
        self.ax_status.axis("off")
        self.lbl_status = self.ax_status.text(
            0, 0.5, "Bereit.", fontsize=8, va="center", ha="left",
            family="monospace", color="#333333"
        )

    # ── Laden & Berechnen ────────────────────────────────────

    def _laden(self):
        """
        Lädt das neueste Bild aus dem konfigurierten Ordner,
        extrahiert die Profile und startet die Neuberechnung.
        """
        bild_ordner  = self.cfg["bild_ordner"]
        datei_endung = self.cfg.get("datei_endung", "*.png")  # .get() → Default falls Schlüssel fehlt

        # Neuestes Bild im Ordner suchen (filtert _profil_-Dateien heraus)
        bild_pfad = neuestes_bild(bild_ordner, datei_endung)
        if bild_pfad is None:
            self._status(f"Kein Bild in '{bild_ordner}' gefunden.")
            return

        self.bild_pfad = bild_pfad
        self.bild_arr  = bild_laden(bild_pfad)

        # Robuste 1D-Profile + finales Zentrum (einmalige 3-Stufen-Berechnung pro Bild)
        self.profil_x, self.profil_y, self.c_y, self.c_x = extract_profiles(self.bild_arr)

        self._neuberechnen()

    def _neuberechnen(self):
        """
        Liest die aktuellen TextBox-Werte, berechnet Homo & Sym für beide
        Analysemodi und ruft _plot() auf.

        Modus 1 – Feste Feldgröße:
            analyse_feldgroesse() → Fenster fix = ±feld_halb_px
        Modus 2 – Ziel-Homogenität:
            analyze_profile() → sucht größtes Feld mit homo ≤ ziel_h
        """
        if self.bild_arr is None:
            return

        # Feldgröße aus TextBox lesen (Komma als Dezimaltrenner auch erlaubt)
        try:
            feld_mm = float(self.tb_feld.text.replace(",", "."))
        except ValueError:
            self._status("[Fehler] Ungültige Feldgröße")
            return

        # Ziel-Homogenität aus TextBox lesen
        try:
            ziel_h = float(self.tb_homo.text.replace(",", "."))
        except ValueError:
            self._status("[Fehler] Ungültige Homogenität")
            return

        # Neue Werte in Config übernehmen und persistent speichern
        self.cfg["feld_groesse_mm"]   = feld_mm
        self.cfg["ziel_homogenitaet"] = ziel_h
        speichere_config(self.cfg, basis_ordner)

        pixel_mm    = self.cfg["pixel_mm"]
        rand_breite = self.cfg["rand_breite_untergrund"]

        # Halbbreite des Feldes in Pixeln berechnen: feld_mm / 2 / pixel_mm → px
        # max(1, ...) → mindestens 1 Pixel damit das Feld nicht leer ist
        feld_halb_px = max(1, int(round(feld_mm / 2.0 / pixel_mm)))

        # Untergrundwert einmal für beide Achsen berechnen
        untergrund = untergrund_aus_rand(self.bild_arr, rand_breite)

        # ── Modus 1: Feste Feldgröße ────────────────────────────────────────
        # c_x → Zentrum für X-Profil (horizontal), c_y → Zentrum für Y-Profil (vertikal)
        ergebnis_x_feld = analyse_feldgroesse(self.profil_x, self.c_x, feld_halb_px, pixel_mm, untergrund)
        ergebnis_y_feld = analyse_feldgroesse(self.profil_y, self.c_y, feld_halb_px, pixel_mm, untergrund)

        # ── Modus 2: Ziel-Homogenität → maximale Feldbreite ─────────────────
        res_x = analyze_profile(self.profil_x, ziel_h, pixel_mm, untergrund, self.c_x)
        res_y = analyze_profile(self.profil_y, ziel_h, pixel_mm, untergrund, self.c_y)

        # Ergebnisse aus analyze_profile in einheitliches dict-Format überführen
        ergebnis_x_ziel = self._ergebnis_zu_dict(res_x, self.profil_x, self.c_x, pixel_mm)
        ergebnis_y_ziel = self._ergebnis_zu_dict(res_y, self.profil_y, self.c_y, pixel_mm)

        # Alle Ergebnisse für _plot() zwischenspeichern
        self._letztes_ergebnis = {
            "c_x":             self.c_x,
            "c_y":             self.c_y,
            "ergebnis_x_feld": ergebnis_x_feld,
            "ergebnis_y_feld": ergebnis_y_feld,
            "ergebnis_x_ziel": ergebnis_x_ziel,
            "ergebnis_y_ziel": ergebnis_y_ziel,
            "feld_halb_px":    feld_halb_px,
            "pixel_mm":        pixel_mm,
        }
        self._plot()

    def _ergebnis_zu_dict(self, res: dict | None, profil: np.ndarray, zentrum: int, pixel_mm: float) -> dict:
        """
        Wandelt das Ergebnis von analyze_profile() in ein einheitliches Dict um.

        Parameter
        ---------
        res      : Rückgabe von analyze_profile() oder None
        profil   : Das zugehörige 1D-Profil (für Grenzberechnung)
        zentrum  : Strahlzentrum [px]
        pixel_mm : Umrechnungsfaktor Pixel → mm

        Rückgabe
        --------
        dict mit Schlüsseln: homo, sym, l, r, max_feld_mm
        """
        if res is None:
            # Kein Ergebnis → Nullwerte zurückgeben
            return dict(homo=np.nan, sym=0.0, l=zentrum, r=zentrum, max_feld_mm=0.0)

        # Feldränder aus der Breite zurückrechnen
        halb_px = int(round(res["breite_mm"] / 2.0 / pixel_mm))
        return dict(
            homo        = res["homogenitaet"],
            sym         = res["symmetrie"],
            l           = max(0, zentrum - halb_px),            # linker Rand (min. 0)
            r           = min(len(profil) - 1, zentrum + halb_px),  # rechter Rand (max. Array-Ende)
            max_feld_mm = res["breite_mm"],
        )

    # ── Plot ─────────────────────────────────────────────────

    def _plot(self):
        """Zeichnet X- und Y-Profile mit Feldmarkierungen und Ergebnis-Infobox."""
        d        = self._letztes_ergebnis
        pixel_mm = d["pixel_mm"]
        c_x      = d["c_x"]   # Strahlzentrum X [px]
        c_y      = d["c_y"]   # Strahlzentrum Y [px]

        self.ax.cla()  # ax.cla() → Axes leeren (clear axes), damit alter Plot überschrieben wird

        # Pixelachse in mm-Abstand vom jeweiligen Zentrum umrechnen
        abstand_x_mm = np.array([(i - c_x) * pixel_mm for i in range(len(self.profil_x))])
        abstand_y_mm = np.array([(i - c_y) * pixel_mm for i in range(len(self.profil_y))])

        # ax.plot() → Liniengraph; lw=2 = Linienbreite 2; label = Legende
        self.ax.plot(abstand_x_mm, self.profil_x, color="#2980b9", lw=2, label="X-Profil")
        self.ax.plot(abstand_y_mm, self.profil_y, color="#c0392b", lw=2, label="Y-Profil")

        # axvspan() → vertikale farbige Fläche zwischen zwei X-Werten
        halb_feld_mm = d["feld_halb_px"] * pixel_mm
        self.ax.axvspan(-halb_feld_mm, halb_feld_mm, color="#4dd414", alpha=0.10,
                        label=f"Feld {self.cfg['feld_groesse_mm']:.1f} mm")

        # Farbige Flächen: maximales Feld bei Ziel-Homogenität (blau = X, rot = Y)
        halb_x_ziel_mm = (d["ergebnis_x_ziel"]["r"] - c_x) * pixel_mm
        halb_y_ziel_mm = (d["ergebnis_y_ziel"]["r"] - c_y) * pixel_mm
        self.ax.axvspan(-halb_x_ziel_mm, halb_x_ziel_mm, color="#2980b9", alpha=0.2,
                        label=f"X-Maxfeld H < {self.cfg['ziel_homogenitaet']:.1f}%")
        self.ax.axvspan(-halb_y_ziel_mm, halb_y_ziel_mm, color="#c0392b", alpha=0.2,
                        label=f"Y-Maxfeld H < {self.cfg['ziel_homogenitaet']:.1f}%")

        self.ax.axvline(0, color="gray", lw=0.8, ls="--")  # senkrechte Linie bei x=0 (Zentrum)
        self.ax.set_xlabel("Strahlprofil / (mm)", fontsize=14)
        self.ax.set_ylabel("Intensität", fontsize=14)
        self.ax.legend(fontsize=14, loc="upper right")
        self.ax.grid(True, which='both', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

        def fmt(wert):
            """Formatiert float als 'x.xx' oder '–' wenn None oder NaN."""
            if wert is None or np.isnan(wert):
                return "–"
            return f"{wert:.2f}"

        # Textbox oben links: Ergebnisse beider Analysemodi
        info_text = (
            f"Feldgröße {self.cfg['feld_groesse_mm']:.1f} mm\n"
            f"  X:\n   Homo={fmt(d['ergebnis_x_feld']['homo'])}%\n"
            f"  Y:\n   Homo={fmt(d['ergebnis_y_feld']['homo'])}%\n"
            f"\n"
            f"Homo < {self.cfg['ziel_homogenitaet']:.1f}%  → Feld\n"
            f"  X:  {fmt(d['ergebnis_x_ziel']['max_feld_mm'])} mm\n"
            f"   Homo={fmt(d['ergebnis_x_ziel']['homo'])}%\n"
            f"  Y:  {fmt(d['ergebnis_y_ziel']['max_feld_mm'])} mm\n"
            f"   Homo={fmt(d['ergebnis_y_ziel']['homo'])}%"
        )
        # ax.text(..., transform=self.ax.transAxes) → Koordinaten relativ zur Axes (0=links, 1=rechts)
        self.ax.text(0.02, 0.97, info_text,
                     transform=self.ax.transAxes, va="top", ha="left",
                     fontsize=14, family="monospace",
                     bbox=dict(facecolor="white", alpha=0.85,
                               edgecolor="#cccccc", boxstyle="round,pad=0.5"))

        # Symmetrie-Infobox (separate Axes unten Mitte)
        sym_text = (
            f"Symmetrie\n"
            f"  X-Zentrum: {c_x} px\n"
            f"  Y-Zentrum: {c_y} px\n"
            f"    X-Sym: {fmt(d['ergebnis_x_ziel']['sym'])}%\n"
            f"    Y-Sym: {fmt(d['ergebnis_y_ziel']['sym'])}%"
        )
        self.lbl_sym_box.set_text(sym_text)

        dateiname = os.path.basename(self.bild_pfad) if self.bild_pfad else ""
        self._status(f"Geladen: {dateiname}")
        self.fig.canvas.draw_idle()  # draw_idle() → Plot neu zeichnen (nur wenn nötig)

    # ── Speichern ────────────────────────────────────────────

    def _speichern(self):
        """
        Speichert den aktuellen Plot als PNG mit fortlaufender Nummerierung.
        Format: <bildname>_profil_NNN.png im selben Ordner wie das Quellbild.
        """
        if self.bild_pfad is None:
            self._status("Kein Bild geladen – nichts zu speichern.")
            return

        bild_ordner = os.path.dirname(self.bild_pfad)            # Ordner des geladenen Bildes
        datei_basis = os.path.splitext(os.path.basename(self.bild_pfad))[0]  # Dateiname ohne Endung
        nummer      = naechste_speichernummer(bild_ordner)
        ziel_pfad   = os.path.join(bild_ordner, f"{datei_basis}_profil_{nummer:03d}.png")  # z.B. bild_profil_001.png

        import matplotlib.transforms as mtransforms

        # Nur den Plot-Bereich (ax) speichern, nicht das gesamte Fenster
        bereich   = self.ax.get_tightbbox(self.fig.canvas.get_renderer())  # exakter Bounding-Box des Plots
        bereich   = bereich.expanded(1.02, 1.02)  # leicht vergrößern damit nichts abgeschnitten wird

        # Bbox von Pixeln in Zoll umrechnen (fig.dpi = Punkte pro Zoll)
        bbox_zoll = mtransforms.Bbox(
            [[bereich.x0 / self.fig.dpi, bereich.y0 / self.fig.dpi],
             [bereich.x1 / self.fig.dpi, bereich.y1 / self.fig.dpi]]
        )
        self.fig.savefig(ziel_pfad, dpi=150, bbox_inches=bbox_zoll, facecolor="white")
        self._status(f"Gespeichert: {os.path.basename(ziel_pfad)}")

    # ── Status-Text ──────────────────────────────────────────

    def _status(self, nachricht: str):
        """
        Setzt den Status-Text unten rechts im Fenster.

        Parameter
        ---------
        nachricht : Anzuzeigende Meldung (z.B. Dateiname oder Fehlertext)
        """
        self.lbl_status.set_text(nachricht)
        self.fig.canvas.draw_idle()


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BeamAnalyzerApp()
