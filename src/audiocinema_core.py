#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AudioCinema Core — Analyzer estable y determinístico
"""

from __future__ import annotations
import os, json
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import yaml
import paho.mqtt.client as mqtt

# ======================================================
# CONSTANTES DEL SISTEMA
# ======================================================

BEEP_FREQ_HZ = 5000.0
BEEP_DURATION_S = 1.838
BEEP_SPACING_S = 4.594
NUM_CHANNELS = 6
GUARD_S = 0.08
DEAD_RMS_DB = -65.0

# ======================================================
# RUTAS
# ======================================================

APP_DIR = Path(__file__).resolve().parent
CONFIG_DIR = APP_DIR / "config"
DATA_DIR = APP_DIR / "data"
REPORTS_DIR = DATA_DIR / "reports"
ASSETS_DIR = APP_DIR / "assets"
CONFIG_PATH = CONFIG_DIR / "config.yaml"

for p in [CONFIG_DIR, DATA_DIR, REPORTS_DIR, ASSETS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ======================================================
# CONFIG
# ======================================================

DEFAULT_CONFIG = {
    "audio": {"fs": 48000, "duration_s": 30.0},
    "reference": {"path": str(ASSETS_DIR / "reference_master.wav")},
    "evaluation": {
        "level": "Medio",
        "tolerances": {
            "Bajo":  {"rms_db": 6, "crest_db": 6},
            "Medio": {"rms_db": 3, "crest_db": 4},
            "Alto":  {"rms_db": 1.5, "crest_db": 2},
        }
    },
    "thingsboard": {"host": "thingsboard.cloud", "port": 1883, "token": ""}
}

def load_config() -> Dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    return DEFAULT_CONFIG | cfg

# ======================================================
# AUDIO UTILS
# ======================================================

def normalize(x: np.ndarray) -> np.ndarray:
    m = np.max(np.abs(x)) + 1e-12
    return (x / m).astype(np.float32)

def rms_db(x: np.ndarray) -> float:
    return 20*np.log10(np.sqrt(np.mean(x**2)+1e-20)+1e-20)

def crest_db(x: np.ndarray) -> float:
    return 20*np.log10((np.max(np.abs(x))+1e-20) /
                       (np.sqrt(np.mean(x**2))+1e-20))

# ======================================================
# GOERTZEL — detección de 5 kHz
# ======================================================

def detect_5k_start(x: np.ndarray, fs: int) -> Optional[float]:
    win = int(0.15 * fs)
    step = int(0.05 * fs)
    k = int(0.5 + (win * BEEP_FREQ_HZ) / fs)
    w = 2 * np.pi * k / win
    coeff = 2 * np.cos(w)

    energies = []
    times = []

    for i in range(0, len(x) - win, step):
        s0 = s1 = 0.0
        for sample in x[i:i+win]:
            s2 = sample + coeff * s0 - s1
            s1 = s0
            s0 = s2
        power = s0*s0 + s1*s1 - coeff*s0*s1
        energies.append(power)
        times.append(i / fs)

    energies = np.array(energies)
    thr = np.median(energies) * 8.0

    idx = np.where(energies > thr)[0]
    if len(idx) == 0:
        return None

    return times[idx[0]]

# ======================================================
# RESULTADOS
# ======================================================

@dataclass
class ChannelResult:
    index: int
    evaluacion: str
    estado: str
    rms_ref_db: float
    rms_cin_db: float
    crest_ref_db: float
    crest_cin_db: float

@dataclass
class GlobalResult:
    overall: str
    channels: List[ChannelResult]

# ======================================================
# ANALYZER
# ======================================================

def analyze(x_ref: np.ndarray, x_cur: np.ndarray, fs: int, cfg: Dict) -> GlobalResult:
    tol = cfg["evaluation"]["tolerances"][cfg["evaluation"]["level"]]

    t0_ref = detect_5k_start(x_ref, fs)
    t0_cur = detect_5k_start(x_cur, fs)

    if t0_ref is None or t0_cur is None:
        return GlobalResult(
            overall="FAILED",
            channels=[
                ChannelResult(i+1, "FAILED", "MUERTO", 0, -120, 0, 0)
                for i in range(NUM_CHANNELS)
            ]
        )

    offset = t0_cur - t0_ref
    shift = int(offset * fs)
    if shift > 0:
        x_cur = x_cur[shift:]
    else:
        x_cur = np.pad(x_cur, (-shift, 0))

    L = min(len(x_ref), len(x_cur))
    x_ref = x_ref[:L]
    x_cur = x_cur[:L]

    channels = []
    any_failed = False

    for i in range(NUM_CHANNELS):
        t_start = t0_ref + i * BEEP_SPACING_S + BEEP_DURATION_S + GUARD_S
        t_end   = t0_ref + (i+1) * BEEP_SPACING_S - GUARD_S

        a = int(t_start * fs)
        b = int(t_end * fs)

        seg_ref = x_ref[a:b]
        seg_cur = x_cur[a:b]

        rms_r = rms_db(seg_ref)
        rms_c = rms_db(seg_cur)
        crest_r = crest_db(seg_ref)
        crest_c = crest_db(seg_cur)

        if rms_c < DEAD_RMS_DB:
            estado = "MUERTO"
            evaluacion = "FAILED"
        else:
            estado = "VIVO"
            evaluacion = "PASSED"
            if abs(rms_c - rms_r) > tol["rms_db"]:
                evaluacion = "FAILED"
            if abs(crest_c - crest_r) > tol["crest_db"]:
                evaluacion = "FAILED"

        if evaluacion == "FAILED":
            any_failed = True

        channels.append(ChannelResult(
            i+1, evaluacion, estado,
            round(rms_r,2), round(rms_c,2),
            round(crest_r,2), round(crest_c,2)
        ))

    overall = "FAILED" if any_failed else "PASSED"
    return GlobalResult(overall, channels)

# ======================================================
# THINGSBOARD
# ======================================================

def send_tb(payload: Dict, cfg: Dict) -> bool:
    token = cfg["thingsboard"]["token"]
    if not token:
        return False
    try:
        client = mqtt.Client()
        client.username_pw_set(token)
        client.connect(cfg["thingsboard"]["host"],
                       cfg["thingsboard"]["port"], 30)
        client.publish("v1/devices/me/telemetry", json.dumps(payload), qos=1)
        client.disconnect()
        return True
    except Exception:
        return False

# ======================================================
# MAIN MEASUREMENT
# ======================================================

def run_measurement(device_index: Optional[int] = None):
    cfg = load_config()
    fs = cfg["audio"]["fs"]
    dur = cfg["audio"]["duration_s"]

    x_ref, _ = sf.read(cfg["reference"]["path"], dtype="float32")
    x_ref = normalize(x_ref)

    x_cur = sd.rec(int(dur*fs), fs, 1, device=device_index)
    sd.wait()
    x_cur = normalize(x_cur[:,0])

    res = analyze(x_ref, x_cur, fs, cfg)

    payload = {
        f"Canal{c.index}": {
            "Evaluacion": c.evaluacion,
            "Estado": c.estado,
            "rms": {"ref": c.rms_ref_db, "cin": c.rms_cin_db},
            "crest": {"ref": c.crest_ref_db, "cin": c.crest_cin_db},
        }
        for c in res.channels
    }
    payload["Resumen"] = {"overall": res.overall}

    out = REPORTS_DIR / f"analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out,"w") as f:
        json.dump(payload,f,indent=2)

    send_tb(payload, cfg)
    return res, payload, out, x_ref, x_cur, fs
