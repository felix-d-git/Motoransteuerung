
// Stepper Controller — ESP32 | 4-Achsen

#include "Basic-Stepper-Driver-SOLDERED.h"
#include "esp_bt.h"
#include <Arduino.h>
#include <WiFi.h>


// KONFIGURATION 1/8-Stepp
constexpr float  MAX_SPEED      = 28000.0f;//7000.0f;
constexpr float  BESCHLEUNIGUNG   = 20000.0f;
constexpr int    MIN_SPEED      = 300;
constexpr int    BAUD_RATE      = 115200;
constexpr double MM_PER_STEP    = 0.0000028858; //28697; //0.0000114789;
constexpr long   Overshoot_STEPS = 8700*4;


// PINS
constexpr int STEP_PINS[4]   = {18,  2, 21, 33};
constexpr int DIR_PINS[4]    = {19,  4, 22, 26};
constexpr int SLP_PINS[4]    = { 5, 15, 13, 27};
constexpr int S_UNTEN_PINS[4] = {17, 39, 25, 14};
constexpr int S_OBEN_PINS[4] = {16, 36, 35, 34};


// GLOBALE INSTANZEN
struct Axis {
    // Positionsverfolgung
    long   logZiel     = 0;
    // Betriebs-Modus
    int    speedMode     = 0;   // 0 = Positions-Modus, !=0 = Speed-Modus
    // Gegenfahrt-Status
    bool   gegenfahrt_ausstehend  = false;
    // Puffer: Befehl der während Gegenfahrt- ankam
    bool   gegenfahrt_aktiv    = false;
    double gegenfahrt_mm        = 0.0;
    // Stopp-Synchronisation
    bool   gegenfahrt_stop      = false;
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




// Harter Stopp: Soll = Ist, damit AccelStepper distanceToGo == 0 hat.
void stopMotor(int i, const char *reason) {
    mot[i]->stop();
    mot[i]->moveTo(mot[i]->currentPosition());
    ax[i].speedMode    = 0;
    ax[i].gegenfahrt_stop     = true;
    ax[i].gegenfahrt_ausstehend = false;
    Serial.printf("STOP_Motor_%d%s\n", i + 1, reason);
}

// Fährt zur absoluten Schritt-Position (kein Gegenfahrt-Status hier).
void FahrZuZiel(int i, long target) {
    mot[i]->setMaxSpeed(MAX_SPEED);
    mot[i]->moveTo(target);
}


// — ENDSCHALTER
void checkeEndschalter(int i) {
    if (ax[i].gegenfahrt_stop) return;   // ← Stopp läuft bereits

    float spd = mot[i]->speed();
    if      (spd > 0 && digitalRead(S_OBEN_PINS[i]) == HIGH) stopMotor(i, ":Anschlag_Oben");
    else if (spd < 0 && digitalRead(S_UNTEN_PINS[i]) == HIGH) stopMotor(i, ":Anschlag_Unten");
}


// — Gegenfahrt-Status-KOMPENSATION

// Startet eine Gegenfahrt-Status-Fahrt: erst 8700 Schritte über Ziel hinaus,
// dann zurück auf logZiel (wird in processGegenfahrt-Status erledigt).
void startfahrtmitOvershoot(int i) {
    ax[i].gegenfahrt_ausstehend = true;
    FahrZuZiel(i, ax[i].logZiel - Overshoot_STEPS);
}

// Wird aus dem Loop aufgerufen wenn distanceToGo == 0 und gegenfahrt_ausstehend gesetzt.
void finishGegenfahrt(int i) {
    ax[i].gegenfahrt_ausstehend = false;
    FahrZuZiel(i, ax[i].logZiel);
}


// — BEFEHLS-VERARBEITUNG


// Relativer Positions-Befehl in mm.
void Step(int i, double mm) {
    digitalWrite(SLP_PINS[i], HIGH);
    ax[i].speedMode = 0;

    // Befehl kam während laufender Gegenfahrt-Status-Fahrt → puffern
    if (ax[i].gegenfahrt_ausstehend) {
        ax[i].gegenfahrt_mm    += mm;
        ax[i].gegenfahrt_aktiv = true;
        Serial.printf("PUFFER_Motor_%d_warte_auf_Gegenfahrt-Status\n", i + 1);
        return;
    }

    ax[i].logZiel += mm / MM_PER_STEP;

    if (mm < 0) {
        startfahrtmitOvershoot(i);
    } else {
        FahrZuZiel(i, ax[i].logZiel);
    }
}

// Kontinuierliche Fahrt mit Schritt/s.
void Speed(int i, int s) {
    digitalWrite(SLP_PINS[i], HIGH);
    ax[i].speedMode     = s;
    ax[i].gegenfahrt_ausstehend  = false;
    ax[i].logZiel     = mot[i]->currentPosition();

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


// Stopp-Synchronisation: logZiel nach vollständigem Stillstand setzen.
void SynchronisationStop(int i) {
    if (!ax[i].gegenfahrt_stop) return;
    if (mot[i]->speed() != 0 || mot[i]->distanceToGo() != 0) return;

    ax[i].gegenfahrt_stop      = false;
    ax[i].logZiel     = mot[i]->currentPosition();
    Serial.printf("SYNC_Motor_%d_Pos_gesetzt\n", i + 1);
}

// Gegenfahrt-Status-Abschluss und gepufferten Befehl ausführen.
void processGegenfahrt(int i) {
    if (ax[i].speedMode != 0)        return;   // nur im Positions-Modus
    if (mot[i]->distanceToGo() != 0) return;   // noch in Fahrt

    if (ax[i].gegenfahrt_ausstehend) {
        finishGegenfahrt(i);
        return;
    }

    if (ax[i].gegenfahrt_aktiv) {
        double mm        = ax[i].gegenfahrt_mm;
        ax[i].gegenfahrt_mm     = 0.0;
        ax[i].gegenfahrt_aktiv = false;

        ax[i].logZiel += mm / MM_PER_STEP;

        if (mm < 0) {
            startfahrtmitOvershoot(i);
        } else {
            FahrZuZiel(i, ax[i].logZiel);
        }
        Serial.printf("PUFFER_Motor_%d_ausgefuehrt\n", i + 1);
    }
}

void processAxis(int i) {
    static bool warInBewegung[4] = {false, false, false, false};
    mot[i]->run();
    SynchronisationStop(i);
    processGegenfahrt(i);
    if (mot[i]->speed() != 0) checkeEndschalter(i);
    // Bedingung: Motor steht (speed 0) und keine Gegenfahrt ausstehend/aktiv
    if (mot[i]->distanceToGo() == 0 && !ax[i].gegenfahrt_ausstehend && !ax[i].gegenfahrt_aktiv) {
        // Hier sicherstellen, dass wir nicht bei jedem Loop-Durchlauf spammen:
        if (warInBewegung[i]) {
            Serial.printf("INFO_Motor_%d_Fahrt_abgeschlossen\n", i + 1);
            warInBewegung[i] = false;
            digitalWrite(SLP_PINS[i], LOW);
        }
    } else {
        // Wenn er sich bewegt, setzen wir den Flag
        if (mot[i]->distanceToGo() != 0) warInBewegung[i] = true;
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
        digitalWrite(SLP_PINS[i], LOW);  //  Haltestrom off
    }
    Serial.println("SYSTEM BEREIT");
}

void loop() {
    verarbeiteSerial();

    for (int i = 0; i < 4; i++)
        processAxis(i);

    sendeStatus();
}
