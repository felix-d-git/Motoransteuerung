
// Stepper Controller — ESP32 | 4-Achsen

#include "Basic-Stepper-Driver-SOLDERED.h"
#include "esp_bt.h"
#include <Arduino.h>
#include <WiFi.h>


// KONFIGURATION 1/8-Stepp
constexpr float  MAX_SPEED      = 28000.0f;
constexpr float  BESCHLEUNIGUNG   = 20000.0f;
constexpr int    MIN_SPEED      = 300;
constexpr int    BAUD_RATE      = 115200;
constexpr double MM_PER_STEP    = 0.0000028858; 
constexpr long   Overshoot_STEPS = 8700*4;   // angenommenes Getriebespiel


// PINS
constexpr int STEP_PINS[4]   = {18,  2, 21, 33};
constexpr int DIR_PINS[4]    = {19,  4, 22, 26};
constexpr int SLP_PINS[4]    = { 5, 15, 13, 27};
constexpr int S_UNTEN_PINS[4] = {17, 39, 25, 14};
constexpr int S_OBEN_PINS[4] = {16, 36, 35, 34};


// GLOBALE INSTANZEN
struct Axis {
    long logZiel   = 0;                // Soll-Position in Schritten
    int  speedMode = 0;                // 0 = Positions-Modus, !=0 = Speed-Modus
    long spiel     = Overshoot_STEPS;  // Spiel nach oben gefasst (0..Overshoot_STEPS)
    long letztePos = 0;                // fuer die Spiel-Mitzaehlung
    bool slpAn     = false;            // Treiber bestromt? (nur bei Wechsel schalten)
};

BasicStepper *mot[4];
Axis          ax[4];

// Serial-Parser
char cmdBuf[64];
int  cmdIdx = 0;

// Status-Timer
int statusMotor = 0;

bool statusReady  = false;
char statusBuf[10];
unsigned long letzterStatus = 0;




// Spielausgleich: logZiel wird immer von unten angefahren.
long fahrZiel(long logZiel, long ist, long spiel) {
    if (logZiel < ist)                                  return logZiel - Overshoot_STEPS; // Ziel drunter -> untergreifen
    if (spiel + (logZiel - ist) < Overshoot_STEPS)      return logZiel - Overshoot_STEPS; // Auflauf zu kurz -> untergreifen
    return logZiel;                                                                       // Spiel gefasst -> direkt hoch
}

// Ziel neu setzen; jederzeit waehrend der Fahrt aufrufbar (zieht nach).
void FahrZuZiel(int i) {
    mot[i]->setMaxSpeed(MAX_SPEED);
    mot[i]->moveTo(fahrZiel(ax[i].logZiel, mot[i]->currentPosition(), ax[i].spiel));
}

// Harter Stopp: setCurrentPosition nullt _speed/_stepInterval/_n -> Pulse sofort aus.
void stopMotor(int i, const char *reason) {
    mot[i]->setCurrentPosition(mot[i]->currentPosition());
    ax[i].speedMode = 0;
    ax[i].logZiel   = mot[i]->currentPosition();
    Serial.printf("STOP_Motor_%d%s\n", i + 1, reason);
    Serial.printf("SYNC_Motor_%d_Pos_gesetzt\n", i + 1);
}


// — ENDSCHALTER
void checkeEndschalter(int i) {
    float spd = mot[i]->speed();
    if      (spd > 0 && digitalRead(S_OBEN_PINS[i]) == HIGH) stopMotor(i, ":Anschlag_Oben");
    else if (spd < 0 && digitalRead(S_UNTEN_PINS[i]) == HIGH) stopMotor(i, ":Anschlag_Unten");
}


// — BEFEHLS-VERARBEITUNG
// Relativer Positions-Befehl in mm. Kein Puffer, Ziel wird sofort nachgezogen.
void Step(int i, double mm) {
    bool inFahrt = (mot[i]->distanceToGo() != 0);

    ax[i].speedMode = 0;
    ax[i].logZiel  += lround(mm / MM_PER_STEP);
    FahrZuZiel(i);

    if (inFahrt) Serial.printf("NACHZIEHEN_Motor_%d_Ziel_erweitert\n", i + 1);
}

// Kontinuierliche Fahrt mit Schritt/s.
void Speed(int i, int s) {
    ax[i].speedMode = s;
    ax[i].logZiel   = mot[i]->currentPosition();

    if (s == 0) { stopMotor(i, ":Befehl_0"); return; }

    int eff = (abs(s) < MIN_SPEED) ? (s > 0 ? MIN_SPEED : -MIN_SPEED) : s;
    if (eff != s)
        Serial.printf("WARN_Speed_Motor_%d_auf_%d_angehoben\n", i + 1, MIN_SPEED);

    mot[i]->setMaxSpeed(abs(eff));
    mot[i]->moveTo(eff > 0 ? 2000000000L : -2000000000L);
}


// — SERIAL-PARSER


void verarbeiteSerial() {
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\n' || c == '\r') { //|| = Oder 
            if (cmdIdx == 0) return;
            cmdBuf[cmdIdx] = '\0';
            cmdIdx = 0;

            int idx = cmdBuf[0] - '1';
            if (idx < 0 || idx > 3) return;

            if      (strncmp(cmdBuf + 1, "speed", 5) == 0) Speed(idx, atoi(cmdBuf + 6));
            else if (strncmp(cmdBuf + 1, "step",  4) == 0) Step (idx, atof(cmdBuf + 5));

        } else if (cmdIdx < 31) {
            cmdBuf[cmdIdx++] = c;
        }
    }
}


// — STATUS-AUSGABE
void sendeStatus() {
    if (!statusReady) {
        // Loop 1: Rechnen + Konvertieren
        if (millis() - letzterStatus < 100) return;
        letzterStatus = millis();
        long pos = mot[statusMotor]->currentPosition();
        dtostrf((double)pos * MM_PER_STEP, 1, 2, statusBuf);
        statusReady = true;
    } else {
        // Loop 2: Senden
        Serial.printf("%d%s\n", statusMotor + 1, statusBuf);
        statusMotor = (statusMotor + 1) % 4;
        statusReady = false;
    }
}


// — LOOP-LOGIK PRO ACHSE


void processAxis(int i) {
    static bool warInBewegung[4] = {false, false, false, false};

    mot[i]->run();
    long pos = mot[i]->currentPosition();

    long s = ax[i].spiel + (pos - ax[i].letztePos);       // hoch fasst, runter gibt frei
    ax[i].letztePos = pos;
    ax[i].spiel = constrain(s, 0L, Overshoot_STEPS);

    if (mot[i]->speed() != 0) checkeEndschalter(i);

    if (ax[i].speedMode == 0 && mot[i]->distanceToGo() == 0 && pos != ax[i].logZiel)
        FahrZuZiel(i);                                    // Untergriff erreicht -> auflaufen

    if (mot[i]->distanceToGo() == 0 && pos == ax[i].logZiel) {
        if (warInBewegung[i]) {
            Serial.printf("INFO_Motor_%d_Fahrt_abgeschlossen\n", i + 1);
            warInBewegung[i] = false;
        }
    } else if (mot[i]->distanceToGo() != 0) {
        warInBewegung[i] = true;
    }

    // Haltestrom nur waehrend der Fahrt (isRunning = speed!=0 oder Ziel noch nicht erreicht)
    bool laeuft = mot[i]->isRunning();
    if (laeuft != ax[i].slpAn) {
        digitalWrite(SLP_PINS[i], laeuft ? HIGH : LOW);
        ax[i].slpAn = laeuft;
    }
}


// SETUP & LOOP


void setup() {
    Serial.begin(BAUD_RATE);
    WiFi.mode(WIFI_OFF);
    btStop();

    for (int i = 0; i < 4; i++) {
        mot[i] = new BasicStepper(1, STEP_PINS[i], DIR_PINS[i]);
        mot[i]->setMaxSpeed(MAX_SPEED);
        mot[i]->setAcceleration(BESCHLEUNIGUNG);

        pinMode(S_UNTEN_PINS[i], INPUT);
        pinMode(S_OBEN_PINS[i], INPUT);
        pinMode(SLP_PINS[i], OUTPUT);
        digitalWrite(SLP_PINS[i], LOW);   // Haltestrom aus, wird je Fahrt geweckt
    }
    Serial.println("SYSTEM BEREIT");
}

void loop() {
    verarbeiteSerial();

    for (int i = 0; i < 4; i++)
        processAxis(i);

    sendeStatus();
}
