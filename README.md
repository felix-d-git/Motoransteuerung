# Motoransteuerung für das Doppelstreusystem

Bachelorarbeit: Steuerung und Strahlprofil-Analyse eines Doppelstreusystems
über einen ESP32-Mikrocontroller.

## Struktur

- `py/` – Python-Programm (GUI-Steuerung + Bildanalyse)
  - `Motoransteuerung.py` – Hauptprogramm, matplotlib-GUI, serielle Kommunikation mit dem ESP32
  - `beam_analysis4.py` – Strahlprofil-Analyse (Zentrum, Homogenität, Symmetrie)
  - `config_utils.py` – Laden/Speichern der Konfiguration
  - `config.json` – Parameter (COM-Port, Kalibrierung, Pixel/mm, Zielwerte …)
- `AnsteuerungESP32_6.3.2/` – Arduino-Sketch (`.ino`) für den ESP32
- `dist/` – gebautes Standalone-Programm (PyInstaller)

## Nutzung

```
cd py
python Motoransteuerung.py
```

Vor dem Start `config.json` an die eigene Hardware anpassen (v.a. `serieller_port`
und die Bild-Ordner-Pfade).
