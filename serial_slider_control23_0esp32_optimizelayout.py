import serial               # serial.Serial(port, baud) → Verbindung zum Mikrocontroller über USB
import matplotlib.pyplot as plt  # Grafik-Fenster und Plot-Funktionen
from matplotlib.widgets import Slider, Button, TextBox  # GUI-Elemente: Schieberegler, Knöpfe, Eingabefelder
import threading            # threading.Thread(target, daemon) → Parallelausführung im Hintergrund
import time                 # time.sleep(s) → Pause; time.time() → aktuelle Unix-Zeit in Sekunden
import os
import sys
import glob
import warnings
import numpy as np
import json
import queue
from pathlib import Path

warnings.filterwarnings("ignore")  # Unterdrückt alle Python-Warnmeldungen im Terminal

try:
    import cv2                                       # OpenCV: Bildverarbeitung
    from scipy.signal import medfilt                 # medfilt(data, kernel_size) → 1D Median-Filter
    from scipy.ndimage import median_filter          # median_filter(input, size) → nD Median-Filter
    from PIL import Image                            # Image.open(pfad) → PIL.Image: Bilddatei laden
    from beam_analysis4 import (                     # Eigenes Analyse-Modul
        bild_laden,                                  # bild_pfad → float32-Array oder None
        neuestes_bild,                               # bild_ordner, datei_endung → neuester Bildpfad
        extract_profiles,                            # bild_arr, n_linien → (profil_x, profil_y, c_y, c_x)
        analyze_profile,                             # profil_serie, ziel_homo, pixel_mm, untergrund, zentrum_px → dict
        untergrund_aus_rand,                         # bild_arr, rand_breite → float
    )
    from config_utils import basis_ordner_ermitteln, lade_config, speichere_config
except ImportError:
    print("[FEHLER] Bitte benötigte Pakete installieren: pip install opencv-python scipy pillow")

# Thread-sicherer Kanal: Hintergrund-Threads schreiben hier rein, der Hauptthread (GUI) liest daraus
_gui_queue = queue.SimpleQueue()

skript_ordner = basis_ordner_ermitteln()  # EXE-Ordner oder Skript-Ordner (aus config_utils)


# ──────────────────────────────────────────────────────────────
#  KONFIGURATION
# ──────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    # ── Motorsteuerung-spezifisch ─────────────────────────────────────────────
    "serieller_port":               "COM9",       # COM-Port des Mikrocontrollers
    "baudrate":                     115200,        # Übertragungsrate in Baud
    "fenster_titel":                "Motorsteuerung",
    "geschwindigkeit_max":          28000,         # Maximale Motorgeschwindigkeit (Steps/s)
    "schritt_werte":                [-10, -2, -0.2, -0.02, -0.01, 0, 0.01, 0.02, 0.2, 2, 10],
    "kalibrierungsgeschwindigkeit": 1000,          # Motorgeschwindigkeit bei automatischer Kalibrierung
    "kalibrierungs_bildpfad":       str(skript_ordner),  # Ordner für Kalibrierungsbilder
    "mm_pro_prozent_zweiring":      0.0582,        # Korrektur-Schrittweite [mm/%] für 2-Ring-Streuer
    "mm_pro_prozent_vierring":      0.1104,        # Korrektur-Schrittweite [mm/%] für 4-Ring-Streuer
    "sym_toleranz_prozent":         0.2,           # Toleranzband [%]: kein Step wenn |sym| < Toleranz
    "streuer_modus":                4,             # Aktiver Streuer: 2 oder 4 (Ringe)
    "kalibrierungsrichtung_x":      -1,            # Vorzeichen der X-Korrektur (+1 oder -1)
    "kalibrierungsrichtung_y":      +1,            # Vorzeichen der Y-Korrektur (+1 oder -1)
    # ── Gemeinsame Keys (identisch mit beam_analyzer) ─────────────────────────
    "pixel_mm":                     0.0588,        # Umrechnungsfaktor Pixel → mm
    "ziel_homogenitaet":            10.0,          # Ziel-Homogenität in % für analyze_profile
    "datei_endung":                 "*.png",       # Glob-Muster für Bildtyp
    "rand_breite_untergrund":       5,             # Randbreite [px] für untergrund_aus_rand()
    # ── Beam-Analyzer-Keys (werden mitgespeichert, hier als Defaults) ─────────
    "bild_ordner":                  str(skript_ordner / "data"),  # Bildordner für BeamAnalyzer
    "feld_groesse_mm":              20.0,          # Feste Feldgröße [mm] für BeamAnalyzer
}

config = lade_config(DEFAULT_CONFIG, skript_ordner)  # Config laden via config_utils

# ── Konfigurationswerte in globale Variablen laden ──────────────────────────
serieller_port               = config["serieller_port"]
baudrate                     = config["baudrate"]
fenster_titel                = config["fenster_titel"]
geschwindigkeit_max          = config["geschwindigkeit_max"]
schritt_werte                = config["schritt_werte"]
kalibrierungsgeschwindigkeit = config["kalibrierungsgeschwindigkeit"]
kalibrierungs_bildpfad       = config["kalibrierungs_bildpfad"]
pixel_mm                     = config["pixel_mm"]
ziel_homogenitaet            = config["ziel_homogenitaet"]
datei_endung                 = config["datei_endung"]
mm_pro_prozent_zweiring      = config["mm_pro_prozent_zweiring"]
mm_pro_prozent_vierring      = config["mm_pro_prozent_vierring"]
sym_toleranz                 = config["sym_toleranz_prozent"]
streuer_modus                = config["streuer_modus"]
kalibrierungsrichtung_x      = config["kalibrierungsrichtung_x"]
kalibrierungsrichtung_y      = config["kalibrierungsrichtung_y"]
rand_breite_untergrund       = config["rand_breite_untergrund"]


def get_mm_pro_prozent() -> float:
    """
    Gibt die Korrektur-Schrittweite [mm pro 1% Asymmetrie] zurück,
    abhängig vom aktuell gewählten Streuer-Typ (2- oder 4-Ring).

    Rückgabe
    --------
    float : mm/% Schrittweite
    """
    return mm_pro_prozent_vierring if streuer_modus == 4 else mm_pro_prozent_zweiring


# ──────────────────────────────────────────────────────────────
#  DESIGN / FARBEN
# ──────────────────────────────────────────────────────────────
farbe_hintergrund = '#f5f6fa'
farbe_achse       = '#ffffff'
farbe_motor = {       # Motor-ID als String → Farbe als Hex-Code
    '1': '#3498db',   # blau
    '2': '#e67e22',   # orange
    '3': '#2ecc71',   # grün
    '4': '#9b59b6',   # lila
}

stil_label    = {'fontsize': 11, 'fontweight': 'bold', 'color': '#353b48'}
stil_position = {'fontsize': 13, 'fontweight': 'bold', 'color': '#e84118'}


# ──────────────────────────────────────────────────────────────
#  GLOBALE ZUSTANDSVARIABLEN
# ──────────────────────────────────────────────────────────────
motor_positionen       = {'1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0}  # Aktuelle Position jedes Motors [mm]
motor_positionen_prev  = {'1': None, '2': None, '3': None, '4': None}  # Vorige Position (für Bewegungserkennung)
motor_positionen_prev2 = {'1': None, '2': None, '3': None, '4': None}  # Vorvorherige Position
serielles_log          = ["--- Terminal Bereit ---"]  # Ringpuffer für Terminal-Nachrichten (max. 25 Einträge)
programm_laeuft        = True    # Flag: False beendet den Reader-Thread
serielle_verbindung    = None    # serial.Serial-Objekt; None wenn nicht verbunden
aktive_stickies        = []      # Hervorgehobene Nachrichten mit Ablaufzeit [{msg, expiry}]
reader_thread          = None    # Thread-Objekt des seriellen Lesers

kalibrierung_laeuft    = False   # True während Kalibrierung läuft
kalibrierungs_thread   = None    # Thread-Objekt des Kalibrierungs-Loops
warte_auf_motor_fertig = set()   # Motor-IDs (str) auf die noch gewartet wird
kalib_bild_pfad        = None    # Pfad des aktuell in Bearbeitung befindlichen Kalibrierungsbildes
bild_zum_loeschen      = None    # Pfad des Bildes der vorigen Kalibrierung; wird erst beim nächsten Start gelöscht
WARTE_AUF_BILD_MAX_S   = 10       # Max. Wartezeit [s] auf ein neues Bild nach dem Löschen
WARTE_AUF_BILD_TAKT_S  = 1        # Prüfintervall [s] beim Warten auf ein neues Bild


# ──────────────────────────────────────────────────────────────
#  HILFSFUNKTIONEN
# ──────────────────────────────────────────────────────────────

def _pruefe_motor_fertig(motor_nr: str):
    """
    Wird aufgerufen wenn der ESP32 "INFO_Motor_X_Fahrt_abgeschlossen" sendet.
    Entfernt motor_nr aus der Warteliste. Wenn alle Motoren fertig sind, wird
    die GUI zurückgesetzt. Das Kalibrierungsbild wird NICHT sofort gelöscht,
    sondern erst beim nächsten Start der Auto-Kalibrierung (siehe bild_zum_loeschen).

    Parameter
    ---------
    motor_nr : Motor-ID als String (z.B. "3" oder "4")
    """
    global warte_auf_motor_fertig, kalib_bild_pfad, bild_zum_loeschen, kalibrierung_laeuft

    if motor_nr not in warte_auf_motor_fertig:
        return  # Nicht auf diesen Motor gewartet → ignorieren

    warte_auf_motor_fertig.discard(motor_nr)  # set.discard() → entfernt Element wenn vorhanden
    log_hinzufuegen(f"Motor {motor_nr} bestätigt fertig.")

    if warte_auf_motor_fertig:
        return  # Noch andere Motoren ausstehend → warten

    # Alle Motoren fertig → Bild für die spätere Löschung vormerken (nicht sofort löschen)
    if kalib_bild_pfad:
        bild_zum_loeschen = kalib_bild_pfad
        kalib_bild_pfad = None

    kalibrierung_laeuft = False
    _kalib_gui_zuruecksetzen()


def log_hinzufuegen(nachricht: str, ist_sticky: bool = False):
    """
    Fügt eine Nachricht zum seriellen Log hinzu.

    Normale Nachrichten werden in den Ringpuffer (max. 25 Einträge) geschrieben.
    Sticky-Nachrichten erscheinen hervorgehoben für 5 Sekunden am Ende des Logs.

    Parameter
    ---------
    nachricht  : Anzuzeigende Nachricht (max. 60 Zeichen)
    ist_sticky : True → Nachricht wird als Sticky angezeigt (Großbuchstaben + !!!)
    """
    global serielles_log, aktive_stickies

    nachricht_kurz = nachricht.strip()[:60]  # Auf 60 Zeichen kürzen, Leerzeichen am Rand entfernen

    if ist_sticky:
        # Sticky-Format: Großbuchstaben und !!! als Rahmen
        nachricht_kurz = f"!!! {nachricht_kurz.upper()} !!!"

        # Prüfen ob diese Sticky-Nachricht bereits existiert
        vorhandene = next(
            (s for s in aktive_stickies if s['msg'] == nachricht_kurz), None
        )
        if vorhandene:
            # Ablaufzeit verlängern (Countdown neu starten)
            vorhandene['expiry'] = time.time() + 5
        else:
            # Neue Sticky-Nachricht mit Ablaufzeit in 5 Sekunden
            aktive_stickies.append({'msg': nachricht_kurz, 'expiry': time.time() + 5})
    else:
        serielles_log.append(nachricht_kurz)
        if len(serielles_log) > 25:
            serielles_log.pop(0)  # list.pop(0) → älteste Nachricht entfernen (Ringpuffer)


def sende_befehl(befehl: str, wert: float):
    """
    Formatiert und sendet einen Befehl an den Mikrocontroller.

    Format: "<befehl><wert_als_float_mit_2_dezimalstellen>\\n"
    Beispiel: "4step0.50\\n" → Motor 4, Step von 0.5 mm

    Parameter
    ---------
    befehl : Befehlsstring (z.B. "4step", "1speed", "3step")
    wert   : Zahlenwert zum Befehl (Geschwindigkeit oder Schrittweite)
    """
    # f-String mit Formatierung: :.2f → immer 2 Nachkommastellen
    text_befehl = f"{befehl}{float(wert):.2f}\n"

    # Gesendeten Befehl im Terminal anzeigen
    _gui_queue.put(('log', f"PC: {text_befehl.strip()}", False))

    if serielle_verbindung is not None and serielle_verbindung.is_open:
        try:
            # .encode('utf-8') → String in Bytes umwandeln (Serial kann nur Bytes senden)
            serielle_verbindung.write(text_befehl.encode('utf-8'))
        except Exception as e:
            _gui_queue.put(('log', f"Sende-Fehler: {e}", False))
    else:
        print(f"Simulation: {text_befehl.strip()}")  # Kein Port offen → nur Konsole


def serieller_reader():
    """
    Läuft in eigenem Hintergrund-Thread; liest kontinuierlich von der
    seriellen Schnittstelle und stellt Daten in _gui_queue.

    Protokoll (vom ESP32)
    ----------------------
    - "T<µs>"                              → ESP32-Loop-Zeit (z.B. "T1234")
    - "<motor_id><position>"               → Motorposition (z.B. "112.50" = Motor 1, 12.5 mm)
    - "INFO_Motor_X_Fahrt_abgeschlossen"   → Motor X hat Fahrt beendet
    - Sonstige Blöcke                      → normale Log-Nachrichten
    """
    global programm_laeuft, serielle_verbindung
    zeichen_puffer = ""  # Puffer für noch nicht abgeschlossene Zeichen-Blöcke

    while programm_laeuft:
        if serielle_verbindung and serielle_verbindung.is_open:
            try:
                if serielle_verbindung.in_waiting > 0:  # .in_waiting → Anzahl Bytes im Empfangspuffer
                    # Alle verfügbaren Bytes lesen und als UTF-8-Text dekodieren
                    text = serielle_verbindung.read(serielle_verbindung.in_waiting).decode('utf-8', errors='ignore')

                    for zeichen in text:
                        if zeichen in ' \n\r':  # Leerzeichen oder Zeilenumbruch = Block-Ende
                            block = zeichen_puffer.strip()

                            if block:
                                # ESP32-Loop-Zeit: Block beginnt mit 'T'
                                if block.startswith('T'):
                                    try:
                                        log_hinzufuegen(f"ESP32 Loop: {block[1:]}µs")  # block[1:] → ohne 'T'
                                    except:
                                        pass
                                    zeichen_puffer = ""
                                    continue

                                # Motorposition: erstes Zeichen ist Motor-ID ('1'–'4')
                                ist_position = False
                                if len(block) >= 2 and block[0] in motor_positionen:
                                    try:
                                        positions_wert = float(block[1:])  # Rest des Blocks als Zahl
                                        _gui_queue.put(('pos', block[0], positions_wert))
                                        _gui_queue.put(('log', f"M{block[0]}: {positions_wert}mm", False))
                                        ist_position = True
                                    except ValueError:
                                        pass  # Kein gültiger Float → als normale Nachricht behandeln

                                if not ist_position:
                                    ist_zeilenumbruch = zeichen in '\n\r'  # Zeilenumbrüche als Sticky markieren
                                    _gui_queue.put(('log', block, ist_zeilenumbruch))

                                    # Prüfen ob Motor seine Fahrt abgeschlossen hat
                                    if block.startswith("INFO_Motor_") and block.endswith("_Fahrt_abgeschlossen"):
                                        motor_nr = block[11]  # Position 11 in "INFO_Motor_X..." → Motor-ID
                                        _gui_queue.put(('motor_fertig', motor_nr, None))

                            zeichen_puffer = ""  # Puffer nach jedem Block leeren
                        else:
                            zeichen_puffer += zeichen  # Zeichen sammeln bis Trennzeichen kommt

            except Exception as e:
                _gui_queue.put(('log', f"Verbindung verloren: {e}", False))
                break  # Thread beenden bei Verbindungsabbruch

        time.sleep(0.05)  # 50ms Pause um die CPU nicht unnötig zu belasten


# ──────────────────────────────────────────────────────────────
#  VERBINDUNGS-STEUERUNG
# ──────────────────────────────────────────────────────────────

def verbindung_starten(event):
    """
    Öffnet die serielle Verbindung zum Mikrocontroller und startet den Reader-Thread.
    Port und Baudrate werden aus den Textboxen gelesen.

    Parameter
    ---------
    event : matplotlib Button-Event (wird nicht verwendet)
    """
    global serielle_verbindung, programm_laeuft, reader_thread, serieller_port, baudrate

    verbindung_stoppen(None)  # Eventuell bestehende Verbindung zuerst trennen

    serieller_port = txt_port.text.strip()   # .strip() → Leerzeichen am Rand entfernen
    try:
        baudrate = int(txt_baud.text.strip())  # int() → String in Ganzzahl umwandeln
    except ValueError:
        log_hinzufuegen("ERR: Ungültige Baudrate!")
        return

    try:
        # Serielle Verbindung öffnen: timeout=0.05s → Read-Aufrufe blockieren max. 50ms
        serielle_verbindung = serial.Serial(serieller_port, baudrate, timeout=0.05)
        programm_laeuft = True
        # daemon=True → Thread wird automatisch beendet wenn das Hauptprogramm endet
        reader_thread = threading.Thread(target=serieller_reader, daemon=True)
        reader_thread.start()
        log_hinzufuegen(f"Erfolg: {serieller_port} mit {baudrate} geöffnet.")
    except Exception as e:
        log_hinzufuegen(f"ERR: Verbindung fehlgeschlagen!")
        print(f"Warnung: Konnte {serieller_port} nicht öffnen. Simulation aktiv.")


def verbindung_stoppen(event):
    """
    Schließt die serielle Verbindung und stoppt den Reader-Thread.

    Parameter
    ---------
    event : matplotlib Button-Event (wird nicht verwendet)
    """
    global serielle_verbindung, programm_laeuft
    programm_laeuft = False  # Signal an Reader-Thread: Schleife beenden
    if serielle_verbindung and serielle_verbindung.is_open:
        try:
            serielle_verbindung.close()  # Port freigeben damit andere Programme ihn nutzen können
            log_hinzufuegen(f"Verbindung zu {serieller_port} getrennt.")
        except Exception as e:
            log_hinzufuegen(f"Fehler beim Schließen: {e}")
    serielle_verbindung = None


# ──────────────────────────────────────────────────────────────
#  KALIBRIERUNGS-BILDORDNER (Datei-Explorer)
# ──────────────────────────────────────────────────────────────

def bildpfad_waehlen(event):
    """
    Öffnet den System-Datei-Explorer, damit der Ordner mit den
    Kalibrierungsbildern bequem ausgewählt werden kann (statt config.json
    von Hand zu editieren). Speichert die Auswahl direkt in config.json.

    Parameter
    ---------
    event : matplotlib Button-Event (wird nicht verwendet)
    """
    global kalibrierungs_bildpfad

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()          # Kein leeres Tk-Fenster anzeigen, nur den Dialog
    root.attributes('-topmost', True)  # Dialog vor das matplotlib-Fenster bringen
    ordner = filedialog.askdirectory(
        title="Kalibrierungs-Bildordner auswählen",
        initialdir=kalibrierungs_bildpfad if os.path.isdir(kalibrierungs_bildpfad) else str(skript_ordner),
    )
    root.destroy()

    if not ordner:
        return  # Dialog abgebrochen

    kalibrierungs_bildpfad = ordner
    config["kalibrierungs_bildpfad"] = ordner
    speichere_config(config, skript_ordner)  # Persistent speichern für nächsten Start

    lbl_bildpfad.set_text(f"Bild: {ordner}")
    log_hinzufuegen(f"Kalib.-Bildordner: {ordner}")
    fig.canvas.draw_idle()


# ──────────────────────────────────────────────────────────────
#  GUI AUFBAU
# ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 9), facecolor=farbe_hintergrund)
plt.subplots_adjust(bottom=0.1, top=0.82, left=0.05, right=0.95)
#fig.suptitle(fenster_titel, fontsize=22, fontweight='bold', color='#2a2f37')

# Kopfzeile zentriert aufbauen: Verbindungs-Panel, NOT-STOPP, Auto-Kalib.+Ring-Wahl,
# Bildordner-Button (ganz rechts) werden als Block mit fixen Lücken um die Fenstermitte gelegt.
_KOPF_START_X = 0.1025  # (1 - Gesamtbreite 0.795) / 2 → zentriert die gesamte Kopfzeile

# Hintergrundbox für Port/Baud/Buttons (rein optisches Panel)
ax_panel_hg = plt.axes([_KOPF_START_X, 0.88, 0.34, 0.09], facecolor='#dcdde1')
ax_panel_hg.set_xticks([])  # Achsenbeschriftungen ausblenden (nur als Hintergrund genutzt)
ax_panel_hg.set_yticks([])

# Eingabefeld für COM-Port
ax_txt_port = plt.axes([_KOPF_START_X + 0.04, 0.895, 0.07, 0.06])
txt_port = TextBox(ax_txt_port, 'Port: ', initial=serieller_port, color='white')
txt_port.label.set_weight('bold')

# Eingabefeld für Baudrate
ax_txt_baud = plt.axes([_KOPF_START_X + 0.17, 0.895, 0.07, 0.06])
txt_baud = TextBox(ax_txt_baud, 'Baud: ', initial=str(baudrate), color='white')
txt_baud.label.set_weight('bold')

# Verbindungs-Buttons (Play = verbinden, Stop = trennen)
ax_btn_play = plt.axes([_KOPF_START_X + 0.25, 0.895, 0.035, 0.06])
btn_play = Button(ax_btn_play, '▶ Play', color='#2ecc71', hovercolor='#27ae60')
btn_play.label.set_color('white')
btn_play.label.set_weight('bold')
btn_play.on_clicked(verbindung_starten)

ax_btn_stop = plt.axes([_KOPF_START_X + 0.29, 0.895, 0.035, 0.06])
btn_stop = Button(ax_btn_stop, '⏹ Stop', color='#e74c3c', hovercolor='#c0392b')
btn_stop.label.set_color('white')
btn_stop.label.set_weight('bold')
btn_stop.on_clicked(verbindung_stoppen)

# Log-Fenster (Terminal-Ausgabe der seriellen Kommunikation)
ax_terminal = plt.axes([0.35, 0.10, 0.30, 0.50], facecolor="#2a2f37")
ax_terminal.set_xticks([])
ax_terminal.set_yticks([])
terminal_text = ax_terminal.text(
    0.05, 0.95, "",
    color="#64f8a2",   # hellgrün auf dunklem Hintergrund (Terminal-Stil)
    fontsize=9,
    family='monospace',
    va='top',
    ha='left'
)
ax_terminal.set_title("SERIAL LOG", color='#718093', fontsize=10, fontweight='bold')

# Infobox (Kalibrierungsergebnisse: Homo, Sym, Breite je Achse)
ax_infobox = plt.axes([0.65, 0.20, 0.14, 0.25])
ax_infobox.axis('off')  # axis('off') → keine Achsenlinien, nur Text
info_text = ax_infobox.text(
    0.05, 0.95, "Warte auf Daten...",
    transform=ax_infobox.transAxes,  # Koordinaten relativ zur Axes-Fläche (0–1)
    fontsize=10,
    verticalalignment='top',
    fontfamily='monospace',
    bbox=dict(boxstyle='round', facecolor='#f5f6fa', alpha=1.0, edgecolor='#dcdde1')
)

# Motor-GUI (Slider, Positions-Labels, Step-Buttons)
sliders          = {}   # Motor-ID → Slider-Objekt
positions_labels = {}   # Motor-ID → Text-Objekt (Positionsanzeige)
step_buttons     = []   # Alle Step-Button-Objekte (Referenzen halten damit Garbage Collection sie nicht löscht)


def erstelle_motor_ui(motor_id: str, x: float, y: float, orientierung: str = 'h'):
    """
    Erstellt Slider, Positions-Label und Step-Buttons für einen Motor.

    Parameter
    ---------
    motor_id    : Motor-ID als String ('1'–'4')
    x, y        : Position in Figur-Koordinaten (0–1, links/unten)
    orientierung: 'h' = horizontal (oben), 'v' = vertikal (seitlich)
    """
    farbe = farbe_motor[motor_id]  # Motorspezifische Farbe aus dem Dict

    # Positions-Label (z.B. "Pos 1: 12.50 mm")
    # fig.text() → Text direkt auf der Figure (nicht in einer Axes)
    if orientierung == 'h':
        positions_label = fig.text(x, y + 0.09, f"Pos {motor_id}: 0.00", **stil_position)
        positions_labels[motor_id] = positions_label
    else:
        positions_label = fig.text(x-0.1, y + 0.15, f"Pos {motor_id}: 0.00", **stil_position)
        positions_labels[motor_id] = positions_label

    # Slider erstellen (horizontal oder vertikal je nach Orientierung)
    if orientierung == 'h':
        ax_slider      = plt.axes([x, y+0.05, 0.4, 0.03], facecolor=farbe_achse)
        plt_orientierung = 'horizontal'
    else:
        ax_slider      = plt.axes([x, y - 0.4, 0.03, 0.55], facecolor=farbe_achse)
        plt_orientierung = 'vertical'

    # Slider(ax, label, min, max, valinit=Startwert, valstep=Schrittweite, color, orientation)
    slider = Slider(ax_slider, 'Speed',
                    -geschwindigkeit_max, geschwindigkeit_max,
                    valinit=0, valstep=1, color=farbe, orientation=plt_orientierung)

    # Bei Slider-Änderung: Geschwindigkeitsbefehl an Motor senden
    # mid=motor_id im lambda-Default speichert die ID für jede Schleifeninstanz separat
    slider.on_changed(lambda v, mid=motor_id: sende_befehl(f"{mid}speed", v))
    sliders[motor_id] = slider

    # Step-Buttons erstellen (eine Reihe pro Schritt-Wert)
    n_schritte = len(schritt_werte)

    if orientierung == 'h':
        btn_breite = 0.4 / (n_schritte + 1)
        btn_hoehe  = 0.06
    else:
        btn_breite = 0.08
        btn_hoehe  = 0.55 / (n_schritte + 1)

    for i, schritt in enumerate(schritt_werte):
        # Position des Buttons berechnen
        if orientierung == 'h':
            btn_x = x + i * (0.4 / n_schritte)
            btn_y = y - 0.04
        else:
            btn_x = x + 0.05
            btn_y = (y - 0.4) + i * (0.6 / n_schritte)

        ax_btn = plt.axes([btn_x, btn_y, btn_breite, btn_hoehe])
        btn    = Button(ax_btn, str(schritt), color='#dcdde1', hovercolor='#7f8c8d')
        btn.label.set_fontsize(7)

        # on_clicked-Callback: s=schritt und mid=motor_id als Default-Parameter speichern
        def beim_step_klick(event, s=schritt, mid=motor_id):
            sende_befehl(f"{mid}step", s)
            if s == 0:
                sliders[mid].set_val(0)  # Slider auf 0 zurücksetzen wenn Stop-Button gedrückt

        btn.on_clicked(beim_step_klick)
        step_buttons.append(btn)  # Referenz halten damit Garbage Collector den Button nicht löscht


# Motoren platzieren
erstelle_motor_ui('2', 0.05, 0.72, 'h')  # Motor 2: oben, horizontal
erstelle_motor_ui('1', 0.15,  0.45, 'v')  # Motor 1: links, vertikal
erstelle_motor_ui('4', 0.55, 0.72, 'h')  # Motor 4: oben rechts, horizontal
erstelle_motor_ui('3', 0.82,  0.45, 'v')  # Motor 3: rechts, vertikal


def gui_aktualisieren(event=None):
    """
    Wird vom matplotlib-Timer aufgerufen (alle 300 ms).
    Verarbeitet alle ausstehenden Queue-Einträge aus den Hintergrund-Threads
    und aktualisiert die GUI: Positionen, Log, Infobox, Kalibrierungs-Button.
    """
    global aktive_stickies, motor_positionen_prev, motor_positionen_prev2

    # Queue leeren (max. 50 Einträge pro Tick um GUI reaktiv zu halten)
    for _ in range(50):
        try:
            eintrag = _gui_queue.get_nowait()  # get_nowait() → nicht blockierend; wirft Empty wenn leer
        except queue.Empty:
            break

        art = eintrag[0]  # Erster Eintrag ist immer die Art der Nachricht

        if art == 'pos':
            # Motorposition aktualisieren
            _, motor_id, positions_wert = eintrag
            motor_positionen[motor_id] = positions_wert

        elif art == 'log':
            # Nachricht ins Log schreiben
            _, nachricht, ist_sticky = eintrag
            log_hinzufuegen(nachricht, ist_sticky=ist_sticky)

        elif art == 'motor_fertig':
            # Motor hat Fahrt abgeschlossen
            _, motor_nr, _ = eintrag
            _pruefe_motor_fertig(motor_nr)

        elif art == 'infobox':
            # Kalibrierungsergebnisse in Infobox anzeigen
            _, text, _ = eintrag
            info_text.set_text(text)

        elif art == 'kalib_reset':
            # Kalibrierungs-Button zurücksetzen
            btn_auto.color = "#8dcd0e"
            btn_auto.label.set_text('Auto_Kalibrierung')

    # Positions-Labels aktualisieren + Farbe: grün = Bewegung, rot = Stillstand
    for motor_id, position in motor_positionen.items():
        if motor_id in positions_labels:
            positions_labels[motor_id].set_text(f"Pos {motor_id}: {position:.2f}")

            # Farbe: Motor bewegt sich wenn aktuelle Position ≠ vorvorherige Position
            prev2 = motor_positionen_prev2[motor_id]
            if prev2 is None or position != prev2:
                positions_labels[motor_id].set_color('#27ae60')  # grün = Bewegung
            else:
                positions_labels[motor_id].set_color('#e84118')  # rot = Stillstand

            # Schieberegister: vorige → vorvorige, aktuelle → vorige
            motor_positionen_prev2[motor_id] = motor_positionen_prev[motor_id]
            motor_positionen_prev[motor_id]  = position

    # Log aufbereiten: Stickies einmischen und abgelaufene entfernen
    anzeige_log = list(serielles_log)   # Kopie erstellen damit Original nicht verändert wird
    jetzt = time.time()                 # Aktuelle Zeit in Sekunden (Unix-Timestamp)

    # Abgelaufene Stickies herausfiltern: nur die behalten deren Ablaufzeit in der Zukunft liegt
    aktive_stickies = [s for s in aktive_stickies if jetzt < s['expiry']]

    for sticky in aktive_stickies:
        if sticky['msg'] in anzeige_log:
            anzeige_log.remove(sticky['msg'])   # Duplikat entfernen
        anzeige_log.append(sticky['msg'])        # Sticky ans Ende setzen (hervorgehoben)

    if len(anzeige_log) > 25:
        anzeige_log = anzeige_log[-25:]  # Nur die letzten 25 Einträge anzeigen

    terminal_text.set_text("\n".join(anzeige_log))
    fig.canvas.draw_idle()  # draw_idle() → GUI neu zeichnen (nur wenn nötig)


# Timer: gui_aktualisieren alle 300 ms aufrufen
timer = fig.canvas.new_timer(interval=300)
timer.add_callback(gui_aktualisieren)
timer.start()


# ──────────────────────────────────────────────────────────────
#  KALIBRIERUNGS-STEUERUNG
# ──────────────────────────────────────────────────────────────

def _kalib_gui_zuruecksetzen():
    """Setzt den Kalibrierungs-Button auf Ausgangszustand zurück (über GUI-Queue)."""
    _gui_queue.put(('kalib_reset', None, None))


def auto_kalibrierungs_loop():
    """
    Kalibrierungsschleife (läuft in eigenem Thread).

    Ablauf
    ------
    1. Bild der vorigen Kalibrierung jetzt löschen (wurde nur vorgemerkt).
    2. Bis zu 10s im 1s-Takt auf ein (neues) Bild in kalibrierungs_bildpfad warten.
    3. extract_profiles() → robuste 1D-Profile für X und Y.
    4. analyze_profile() → Homogenität + Symmetrie je Achse.
    5. Korrektur-Step berechnen und senden wenn |Symmetrie| > sym_toleranz.
    6. GUI-Infobox mit Ergebnissen aktualisieren.
    7. Auf Motor-Fertig-Meldungen warten; Bild erst bei der nächsten Kalibrierung löschen.
    """
    global kalibrierung_laeuft, bild_zum_loeschen

    _gui_queue.put(('log', "Kalibrierung gestartet ...", True))

    # ── Bild der vorigen Kalibrierung jetzt löschen ──────────────────────────
    # (wurde beim Abschluss der letzten Kalibrierung nur vorgemerkt, nicht sofort
    #  gelöscht, damit man es bei Bedarf noch einsehen kann, bevor der nächste Lauf startet)
    if bild_zum_loeschen:
        try:
            os.remove(bild_zum_loeschen)
            log_hinzufuegen(f"Bild gelöscht: {os.path.basename(bild_zum_loeschen)}", ist_sticky=True)
        except OSError as e:
            log_hinzufuegen(f"FEHLER: Bild konnte nicht gelöscht werden: {e}", ist_sticky=True)
        bild_zum_loeschen = None

    # ── Auf (neues) Bild warten ───────────────────────────────────────────────
    # Nach dem Löschen dauert es (kameraseitig) bis zu 10s, bis ein neues Bild im
    # Ordner erscheint. Daher im 1s-Takt prüfen statt sofort einen Fehler zu melden.
    bild_pfad = neuestes_bild(kalibrierungs_bildpfad, datei_endung)
    gewartet_s = 0
    while bild_pfad is None and gewartet_s < WARTE_AUF_BILD_MAX_S:
        log_hinzufuegen(f"Warte auf Bild ... ({gewartet_s}/{WARTE_AUF_BILD_MAX_S}s)")
        time.sleep(WARTE_AUF_BILD_TAKT_S)
        gewartet_s += WARTE_AUF_BILD_TAKT_S
        bild_pfad = neuestes_bild(kalibrierungs_bildpfad, datei_endung)

    if bild_pfad is None:
        _gui_queue.put(('log', f"FEHLER: Kein neues Bild nach {WARTE_AUF_BILD_MAX_S}s!", True))
        _gui_queue.put(('log', "Kalibrierung abgebrochen!", True))
        kalibrierung_laeuft = False
        _kalib_gui_zuruecksetzen()
        return

    bild_arr = bild_laden(bild_pfad)  # Bild als float32-Array laden

    if bild_arr is None:
        log_hinzufuegen("FEHLER: Bild konnte nicht gelesen werden.")
        log_hinzufuegen("Kalibrierung abgebrochen!", ist_sticky=True)
        kalibrierung_laeuft = False
        _kalib_gui_zuruecksetzen()
        return

    # ── Profile + Zentrum extrahieren ───────────────────────────────────────
    profil_x, profil_y, c_y, c_x = extract_profiles(bild_arr, n_linien=5)

    # Untergrundwert einmal für beide Achsen berechnen
    untergrund = untergrund_aus_rand(bild_arr, rand_breite_untergrund)

    # ── Homogenität + Symmetrie je Achse berechnen ───────────────────────────
    # c_x → Zentrum des X-Profils (horizontale Achse)
    # c_y → Zentrum des Y-Profils (vertikale Achse)
    ergebnis_x = analyze_profile(profil_x, ziel_homogenitaet, pixel_mm, untergrund, c_x)
    ergebnis_y = analyze_profile(profil_y, ziel_homogenitaet, pixel_mm, untergrund, c_y)

    # ── Korrektur-Steps berechnen und senden ─────────────────────────────────
    def _korrektur_senden(ergebnis: dict | None, achsen_label: str, befehl: str, richtung: int):
        """
        Sendet Step-Kommando wenn Asymmetrie außerhalb der Toleranz liegt.

        Parameter
        ---------
        ergebnis     : Rückgabe von analyze_profile() (oder None wenn keine Daten)
        achsen_label : "X" oder "Y" (für Log-Nachrichten)
        befehl       : Motor-Befehl (z.B. "4step" für Motor 4 = X-Achse)
        richtung     : kalibrierungsrichtung_x/y (+1 oder -1)
        """
        if ergebnis and ergebnis["symmetrie"] is not None:
            sym = ergebnis["symmetrie"]
            if abs(sym) > sym_toleranz:
                # Step = -Symmetrie × mm/% × Richtung
                # Negativ weil die Korrektur der Abweichung entgegenwirkt
                step = -sym * get_mm_pro_prozent() * richtung
                sende_befehl(befehl, step)
                log_hinzufuegen(f"{achsen_label}-Step: {step:+.3f} mm  (Symm = {sym:.2f} %)",ist_sticky=True)
            else:
                log_hinzufuegen(f"{achsen_label}-Achse OK  (Symm = {sym:.2f} %)",ist_sticky=True)
            return True
        else:
            log_hinzufuegen(f"{achsen_label}-Achse: keine Profildaten", ist_sticky=True)
            return False

    # Motor 4 = X-Achse, Motor 3 = Y-Achse
    _korrektur_senden(ergebnis_x, "X", "4step", kalibrierungsrichtung_x)
    _korrektur_senden(ergebnis_y, "Y", "3step", kalibrierungsrichtung_y)

    # ── Ergebnisse in GUI-Infobox anzeigen ───────────────────────────────────
    def fmt(wert):
        """Formatiert float als 'x.xx' oder '–' wenn None oder NaN."""
        if wert is None or np.isnan(wert):
            return "–"
        return f"{wert:.2f}"

    def _achsen_block(label: str, ergebnis: dict | None) -> list[str]:
        """Erstellt Textzeilen für eine Achse in der Infobox."""
        if ergebnis:
            return [
                f"{label}-Achse:",
                f"  Homo:  {fmt(ergebnis['homogenitaet'])} %",
                f"  Symm:  {fmt(ergebnis['symmetrie'])} %",
                f"  Breite:{fmt(ergebnis['breite_mm'])} mm",
            ]
        return [f"{label}-Achse: Keine Daten"]

    infobox_zeilen = (
        ["=== KALIBRIERUNG ==="]
        + _achsen_block("X", ergebnis_x)
        + ["-" * 18]
        + _achsen_block("Y", ergebnis_y)
    )
    _gui_queue.put(('infobox', "\n".join(infobox_zeilen), None))

    log_hinzufuegen("Kalibrierungsrechnung abgeschlossen – warte auf Motoren ...", ist_sticky=True)

    # ── Auf Motor-Fertig-Meldungen warten ────────────────────────────────────
    global warte_auf_motor_fertig, kalib_bild_pfad
    kalib_bild_pfad = bild_pfad  # Pfad merken; Löschung erfolgt erst bei der nächsten Kalibrierung

    # Nur auf Motoren warten die tatsächlich einen Step bekommen haben
    if ergebnis_x and abs(ergebnis_x["symmetrie"]) > sym_toleranz:
        warte_auf_motor_fertig.add("4")  # Motor 4 = X-Achse
    if ergebnis_y and abs(ergebnis_y["symmetrie"]) > sym_toleranz:
        warte_auf_motor_fertig.add("3")  # Motor 3 = Y-Achse

    if not warte_auf_motor_fertig:
        # Kein Motor musste fahren → Bild für die spätere Löschung vormerken und zurücksetzen
        if kalib_bild_pfad:
            bild_zum_loeschen = kalib_bild_pfad
            kalib_bild_pfad = None
        kalibrierung_laeuft = False
        _kalib_gui_zuruecksetzen()
    # sonst übernimmt _pruefe_motor_fertig() das Zurücksetzen wenn alle Motoren gemeldet haben


def kalibrierung_umschalten(event):
    """
    Startet oder bricht die Kalibrierung ab (Toggle-Funktion).

    Parameter
    ---------
    event : matplotlib Button-Event (wird nicht verwendet)
    """
    global kalibrierung_laeuft, kalibrierungs_thread, kalib_bild_pfad, bild_zum_loeschen

    if kalibrierung_laeuft:
        # Kalibrierung läuft → abbrechen
        kalibrierung_laeuft = False
        warte_auf_motor_fertig.clear()  # set.clear() → alle Elemente entfernen
        if kalib_bild_pfad:
            bild_zum_loeschen = kalib_bild_pfad  # Löschung erfolgt erst beim nächsten Start
            kalib_bild_pfad = None
        log_hinzufuegen("Kalibrierung manuell abgebrochen!")
        _kalib_gui_zuruecksetzen()
    else:
        # Kalibrierung starten
        kalibrierung_laeuft = True
        btn_auto.color = "#e67e22"           # Button-Farbe: Orange während Kalibrierung läuft
        btn_auto.label.set_text('Läuft …')
        fig.canvas.draw_idle()
        # Kalibrierungs-Loop in eigenem Thread starten damit die GUI reaktiv bleibt
        kalibrierungs_thread = threading.Thread(target=auto_kalibrierungs_loop, daemon=True)
        kalibrierungs_thread.start()


# ──────────────────────────────────────────────────────────────
#  STEUER-BUTTONS
# ──────────────────────────────────────────────────────────────

def not_stopp(event):
    """
    Not-Stopp: Kalibrierung abbrechen und alle Motoren auf Geschwindigkeit 0 setzen.

    Parameter
    ---------
    event : matplotlib Button-Event (wird nicht verwendet)
    """
    global kalibrierung_laeuft
    kalibrierung_laeuft = False

    for slider in sliders.values():
        slider.reset()  # Slider.reset() → auf valinit (= 0) zurücksetzen

    for motor_id in ['1', '2', '3', '4']:
        sende_befehl(f"{motor_id}speed", 0)  # Stoppsignal an alle 4 Motoren senden

    log_hinzufuegen("NOT-STOPP ausgelöst!", ist_sticky=True)


# Not-Stopp Button
ax_btn_not_stopp = plt.axes([_KOPF_START_X + 0.35, 0.88, 0.14, 0.09])
btn_not_stopp = Button(ax_btn_not_stopp, 'NOT-STOPP', color='#c23616', hovercolor='#e84118')
btn_not_stopp.label.set_color('white')
btn_not_stopp.label.set_weight('bold')
btn_not_stopp.on_clicked(not_stopp)

# Auto-Kalibrierungs-Button
ax_btn_auto = plt.axes([_KOPF_START_X + 0.50, 0.88, 0.14, 0.04])
btn_auto = Button(ax_btn_auto, 'Auto_Kalibrierung', color="#8dcd0e", hovercolor="#2aa40c")
btn_auto.label.set_color('white')
btn_auto.label.set_weight('bold')
btn_auto.on_clicked(kalibrierung_umschalten)

# Streuer-Modus Buttons (4-Ring oder 2-Ring)
ax_btn_4ring = plt.axes([_KOPF_START_X + 0.50, 0.93, 0.069, 0.035])
btn_4ring = Button(ax_btn_4ring, '4-Ring', color="#2a2f37", hovercolor="#444")
btn_4ring.label.set_color('white')
btn_4ring.label.set_fontsize(9)

ax_btn_2ring = plt.axes([_KOPF_START_X + 0.571, 0.93, 0.069, 0.035])
btn_2ring = Button(ax_btn_2ring, '2-Ring', color="#dcdde1", hovercolor="#aaa")
btn_2ring.label.set_fontsize(9)

# Button zum Auswählen des Kalibrierungs-Bildordners (öffnet System-Explorer) – ganz rechts in der Kopfzeile
ax_btn_bildpfad = plt.axes([_KOPF_START_X + 0.65, 0.895, 0.145, 0.06])
btn_bildpfad = Button(ax_btn_bildpfad, '📁 Bildordner', color='#487eb0', hovercolor='#40739e')
btn_bildpfad.label.set_color('white')
btn_bildpfad.label.set_weight('bold')
btn_bildpfad.label.set_fontsize(9)
btn_bildpfad.on_clicked(bildpfad_waehlen)

# Anzeige des aktuell gewählten Kalibrierungs-Bildordners
lbl_bildpfad = fig.text(_KOPF_START_X + 0.65, 0.965, f"Bild: {kalibrierungs_bildpfad}", fontsize=8, color='#353b48')


def streuer_setzen(modus: int, event=None):
    """
    Wechselt den Streuer-Modus und speichert die Einstellung in config.json.
    Aktualisiert die Button-Farben: aktiver Button dunkel, inaktiver hell.

    Parameter
    ---------
    modus : 2 (Zwei-Ring) oder 4 (Vier-Ring)
    event : matplotlib Button-Event (wird nicht verwendet)
    """
    global streuer_modus
    streuer_modus        = modus
    config["streuer_modus"] = modus
    speichere_config(config, skript_ordner)  # Persistent speichern für nächsten Start

    # Aktiver Button: dunkel; inaktiver Button: hell
    btn_4ring.color = "#2a2f37" if modus == 4 else "#dcdde1"
    btn_4ring.label.set_color('white' if modus == 4 else '#333')
    btn_2ring.color = "#2a2f37" if modus == 2 else "#dcdde1"
    btn_2ring.label.set_color('white' if modus == 2 else '#333')

    log_hinzufuegen(f"Streuer: {modus}-Ring  →  {get_mm_pro_prozent():.3f} mm/%")
    fig.canvas.draw_idle()


# lambda e: → anonyme Funktion die den Button-Event ignoriert, nur modus wird übergeben
btn_4ring.on_clicked(lambda e: streuer_setzen(4))
btn_2ring.on_clicked(lambda e: streuer_setzen(2))
streuer_setzen(streuer_modus)  # Initialzustand aus Config setzen


# ──────────────────────────────────────────────────────────────
#  BEENDEN
# ──────────────────────────────────────────────────────────────

def beim_schliessen(event):
    """
    Wird aufgerufen wenn das matplotlib-Fenster geschlossen wird.
    Stoppt den Reader-Thread und schließt die serielle Verbindung.

    Parameter
    ---------
    event : matplotlib Close-Event (wird nicht verwendet)
    """
    global programm_laeuft
    programm_laeuft = False  # Reader-Thread signalisieren: Schleife beenden
    if serielle_verbindung:
        try:
            serielle_verbindung.close()  # Seriellen Port sauber schließen
        except:
            pass  # Fehler ignorieren – Fenster soll auf jeden Fall schließen


# ──────────────────────────────────────────────────────────────
#  RESPONSIVE SCHRIFTGRÖSSEN
# ──────────────────────────────────────────────────────────────
# Alle Achsen liegen in Figur-relativen Koordinaten (0–1) und skalieren daher
# schon automatisch mit der Fenstergröße. Nur die Schriftgrößen (in Punkt)
# tun das nicht von selbst – ohne diese Anpassung würde z.B. der Terminal-Text
# beim Verkleinern des Fensters aus seiner Box "herauswandern".
_FIGSIZE_BASIS = tuple(fig.get_size_inches())  # Referenzgröße beim Start (15, 9)
_skalierbare_texte = []  # [(Text-Objekt, Basis-Fontsize), ...]


def _texte_registrieren():
    """Sammelt einmalig alle Text-Objekte der Figur (Labels, Buttons, Log, ...)."""
    gesehen = set()
    kandidaten = list(fig.texts)
    for ax in fig.axes:
        kandidaten.extend(ax.texts)
        kandidaten.append(ax.title)

    for txt in kandidaten:
        if id(txt) in gesehen:
            continue
        gesehen.add(id(txt))
        _skalierbare_texte.append((txt, txt.get_fontsize()))


def _bei_groessenaenderung(event=None):
    """
    Wird bei jeder Fenstergrößenänderung aufgerufen (matplotlib 'resize_event').
    Skaliert alle registrierten Schriftgrößen proportional zur aktuellen
    Fenstergröße, damit Texte (v.a. das Log) innerhalb ihrer Boxen bleiben.
    """
    breite, hoehe = fig.get_size_inches()
    skala = min(breite / _FIGSIZE_BASIS[0], hoehe / _FIGSIZE_BASIS[1])
    skala = max(0.35, min(1.5, skala))  # Extreme abfedern (nicht unlesbar klein/riesig)

    for txt, basis_fontsize in _skalierbare_texte:
        txt.set_fontsize(basis_fontsize * skala)

    fig.canvas.draw_idle()


_texte_registrieren()
fig.canvas.mpl_connect('resize_event', _bei_groessenaenderung)


# fig.canvas.mpl_connect() → Event-Handler am Fenster registrieren
fig.canvas.mpl_connect('close_event', beim_schliessen)

verbindung_starten(None)  # Verbindung direkt beim Programmstart aufbauen

print("GUI gestartet.")
plt.show()  # Matplotlib-Ereignisschleife starten; blockiert bis das Fenster geschlossen wird
