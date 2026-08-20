"""
config_utils.py
===============
Gemeinsame Konfigurationslogik für alle Skripte im Projekt.

Enthält:
  - basis_ordner_ermitteln() → Skript-/EXE-Ordner bestimmen
  - lade_config()            → config.json laden (mit Default-Fallback)
  - speichere_config()       → config.json schreiben

Verwendung in einem Hauptskript
---------------------------------
    from config_utils import basis_ordner_ermitteln, lade_config, speichere_config

    BASIS_ORDNER = basis_ordner_ermitteln()

    DEFAULT_CONFIG = {
        "mein_wert": 42,
        "pfad":      str(BASIS_ORDNER),  # Pfade immer als str, nicht Path
    }

    config = lade_config(DEFAULT_CONFIG, BASIS_ORDNER)
    ...
    speichere_config(config, BASIS_ORDNER)

Pfad-Konvention
---------------
Alle Pfad-Werte in der Config werden als Strings gespeichert.
Backslashes (Windows) werden beim Speichern zu Forward-Slashes normiert,
damit config.json plattformunabhängig lesbar bleibt.
Welche Schlüssel Pfade enthalten, wird in PFAD_SCHLUESSEL festgelegt.
"""

import os
import sys
import json
from pathlib import Path

# Schlüssel in DEFAULT_CONFIG, deren Werte Dateipfade sind.
# Diese werden beim Laden/Speichern automatisch normiert (Backslash → Slash).
PFAD_SCHLUESSEL = {
    "bild_ordner",            # beam_analyzer: Ordner mit den Kamerabildern
    "kalibrierungs_bildpfad", # serial_slider: Ordner für Kalibrierungsbilder
}


def basis_ordner_ermitteln() -> Path:
    """
    Gibt den Ordner zurück, in dem das laufende Skript oder die EXE liegt.

    Im EXE-Modus (mit PyInstaller gebaut):
        → Ordner der .exe-Datei (sys.executable)
    Im normalen Python-Modus:
        → Ordner des aufgerufenen Skripts (sys.argv[0])

    Rückgabe
    --------
    Path : absoluter Pfad des Basis-Ordners
    """
    if getattr(sys, "frozen", False):
        # getattr(sys, "frozen", False) → True wenn PyInstaller die EXE gebaut hat
        # sys.executable → vollständiger Pfad zur .exe-Datei
        return Path(sys.executable).parent
    else:
        # sys.argv[0] → Pfad des aufgerufenen Python-Skripts
        # os.path.abspath() → macht den Pfad absolut (löst relative Pfade auf)
        return Path(os.path.abspath(sys.argv[0])).parent


def lade_config(standard_config: dict, basis_ordner: Path) -> dict:
    """
    Lädt die Konfiguration aus config.json im Basisordner.

    Verhalten
    ---------
    - Datei vorhanden & gültig  → geladene Werte; fehlende Schlüssel werden aus
                                   standard_config ergänzt.
    - Datei vorhanden & defekt  → Warnung im Terminal, standard_config wird genutzt.
    - Datei nicht vorhanden     → standard_config wird angelegt und gespeichert.
    - Nach dem Laden wird die Datei immer neu geschrieben (normiert Pfade,
      ergänzt neue Schlüssel nach Code-Updates).

    Parameter
    ---------
    standard_config : dict mit Standardwerten (wird nie verändert)
    basis_ordner    : Ordner, in dem config.json gesucht / angelegt wird

    Rückgabe
    --------
    dict : fertige Konfiguration (alle Schlüssel aus standard_config sind vorhanden)
    """
    basis_ordner = Path(basis_ordner)
    config_pfad  = basis_ordner / "config.json"  # vollständiger Dateipfad

    # Arbeitskopie anlegen – das Original-Dict wird nie verändert
    cfg = standard_config.copy()

    if config_pfad.exists():
        try:
            # Datei als Text einlesen; replace("\\", "/") → Backslashes vorab normieren,
            # damit json.loads() keine Probleme mit Windows-Pfaden bekommt
            rohdaten  = config_pfad.read_text(encoding="utf-8")
            geladene  = json.loads(rohdaten.replace("\\", "/"))

            # Geladene Werte in die Arbeitskopie übernehmen
            for schluessel, wert in geladene.items():
                cfg[schluessel] = wert

            print(f"[INFO] config.json geladen: {config_pfad}")

        except json.JSONDecodeError as e:
            print(f"[WARNUNG] config.json fehlerhaft ({e}) – Standardwerte werden genutzt.")

    else:
        print(f"[INFO] config.json nicht gefunden – Standardwerte werden angelegt: {config_pfad}")

    # Fehlende Schlüssel mit Standardwerten auffüllen.
    # dict.setdefault(key, wert) → setzt key nur dann, wenn er noch nicht vorhanden ist.
    # Das ist wichtig damit neue Config-Einträge nach Code-Updates automatisch auftauchen.
    for schluessel, standardwert in standard_config.items():
        cfg.setdefault(schluessel, standardwert)

    # Pfad-Werte bereinigen: os.path.normpath() behebt z.B. doppelte Slashes
    for schluessel in PFAD_SCHLUESSEL:
        if schluessel in cfg and isinstance(cfg[schluessel], str):
            cfg[schluessel] = os.path.normpath(cfg[schluessel])

    # Config sofort zurückschreiben: normiert Pfade und ergänzt fehlende Schlüssel
    speichere_config(cfg, basis_ordner)

    return cfg


def speichere_config(cfg: dict, basis_ordner: Path) -> None:
    """
    Schreibt die Konfiguration als config.json in den Basisordner.

    Pfad-Werte (Schlüssel aus PFAD_SCHLUESSEL) werden mit Forward-Slashes
    gespeichert, damit die Datei unter Windows und Linux gleich aussieht.

    Parameter
    ---------
    cfg          : Konfigurationsdict, das gespeichert werden soll
    basis_ordner : Zielordner für config.json
    """
    basis_ordner = Path(basis_ordner)
    config_pfad  = basis_ordner / "config.json"

    # Kopie anlegen, damit das Original-Dict nicht verändert wird
    ausgabe = cfg.copy()

    # Backslashes in Pfad-Werten durch Forward-Slashes ersetzen (JSON-Standard)
    for schluessel in PFAD_SCHLUESSEL:
        if schluessel in ausgabe and isinstance(ausgabe[schluessel], str):
            ausgabe[schluessel] = ausgabe[schluessel].replace("\\", "/")

    # json.dumps() → dict in JSON-String umwandeln
    # indent=4            → eingerückte, lesbare Formatierung
    # ensure_ascii=False  → Umlaute (ä, ö, ü) direkt speichern, nicht als \uXXXX
    config_pfad.write_text(
        json.dumps(ausgabe, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )
