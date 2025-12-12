#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GUI de AudioCinema

- Edición completa de configuración (config.yaml)
- Lanza mediciones del core
- Visualiza pistas y resultados
"""

from __future__ import annotations
import traceback
from datetime import datetime
from typing import Optional, List

import tkinter as tk
from tkinter import ttk, messagebox
import ttkbootstrap as tb
from ttkbootstrap.constants import *

import numpy as np
import sounddevice as sd
import soundfile as sf
import yaml
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ===============================
# IMPORTS DEL CORE (LIMPIOS)
# ===============================

from audiocinema_core import (
    APP_DIR,
    ASSETS_DIR,
    load_config,
    run_measurement,
)

APP_NAME = "AudioCinema"
CONFIG_PATH = APP_DIR / "config" / "config.yaml"

# ===============================
# CONFIG IO (GUI SIDE)
# ===============================

def save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

# ===============================
# UTILIDADES MIC
# ===============================

def list_input_devices() -> List[str]:
    try:
        devices = sd.query_devices()
    except Exception:
        return []
    out = []
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            out.append(f"{idx} - {d['name']}")
    return out

def pick_device_index_from_label(label: str) -> Optional[int]:
    try:
        return int(label.split(" - ", 1)[0])
    except Exception:
        return None

# ===============================
# DECORADOR DE ERRORES UI
# ===============================

def ui_action(fn):
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except Exception:
            tb_str = traceback.format_exc()
            print(tb_str)
            messagebox.showerror(APP_NAME, tb_str)
            return None
    return wrapper

# ===============================
# CLASE PRINCIPAL GUI
# ===============================

class AudioCinemaGUI:
    def __init__(self, root: tb.Window):
        self.root = root
        self.root.title(APP_NAME)
        tb.Style(theme="flatly")

        self.cfg = load_config()

        self.test_name = tk.StringVar(value="—")
        self.eval_text = tk.StringVar(value="—")

        self.last_ref = None
        self.last_cur = None
        self.last_fs = int(self.cfg["audio"]["fs"])

        self.input_device_label = tk.StringVar(value="(auto)")

        self._build_ui()

    # ================= UI =================

    def _build_ui(self):
        root_frame = ttk.Frame(self.root, padding=8)
        root_frame.pack(fill=BOTH, expand=True)

        paned = ttk.Panedwindow(root_frame, orient=HORIZONTAL)
        paned.pack(fill=BOTH, expand=True)

        # -------- IZQUIERDA --------
        left = ttk.Frame(paned, padding=6)
        paned.add(left, weight=1)

        ttk.Label(left, text="AudioCinema", font=("Segoe UI", 18, "bold")).pack(pady=6)

        tb.Button(left, text="Configuración", bootstyle=PRIMARY,
                  command=self._popup_config).pack(pady=6)

        tb.Button(left, text="Prueba ahora", bootstyle=SUCCESS,
                  command=self._run_once).pack(pady=6)

        # -------- DERECHA --------
        right = ttk.Frame(paned, padding=6)
        paned.add(right, weight=4)

        header = ttk.Frame(right)
        header.pack(fill=X)

        ttk.Label(header, text="PRUEBA:").grid(row=0, column=0, sticky="w")
        ttk.Entry(header, textvariable=self.test_name,
                  state="readonly", width=30).grid(row=0, column=1, sticky="w")

        ttk.Label(header, text="RESULTADO:").grid(row=1, column=0, sticky="w", pady=4)
        self.eval_lbl = ttk.Label(header, textvariable=self.eval_text,
                                  font=("Segoe UI", 11, "bold"))
        self.eval_lbl.grid(row=1, column=1, sticky="w")

        fig_frame = ttk.Frame(right)
        fig_frame.pack(fill=BOTH, expand=True)

        self.fig = Figure(figsize=(5, 4))
        self.ax_ref = self.fig.add_subplot(2, 1, 1)
        self.ax_cur = self.fig.add_subplot(2, 1, 2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)

        self._clear_waves()

    # ================= AUX =================

    def _clear_waves(self):
        for ax, title in (
            (self.ax_ref, "Pista de referencia"),
            (self.ax_cur, "Pista de prueba"),
        ):
            ax.clear()
            ax.set_title(title)
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Amplitud")
            ax.grid(True)
        self.canvas.draw_idle()

    def _plot_wave(self, ax, x, fs):
        t = np.arange(len(x)) / fs
        ax.plot(t, x, linewidth=0.8)

    def _set_eval(self, overall: str):
        self.eval_text.set(overall)
        if overall == "PASSED":
            self.eval_lbl.configure(foreground="green")
        else:
            self.eval_lbl.configure(foreground="red")

    # ================= POPUP CONFIG =================

    @ui_action
    def _popup_config(self):
        w = tk.Toplevel(self.root)
        w.title("Configuración")

        nb = ttk.Notebook(w)
        nb.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # ---- AUDIO ----
        tab_audio = ttk.Frame(nb)
        nb.add(tab_audio, text="Grabación")

        fs_var = tk.IntVar(value=self.cfg["audio"]["fs"])
        dur_var = tk.DoubleVar(value=self.cfg["audio"]["duration_s"])

        ttk.Label(tab_audio, text="Sample rate (Hz):").grid(row=0, column=0, sticky="w")
        ttk.Entry(tab_audio, textvariable=fs_var).grid(row=0, column=1)

        ttk.Label(tab_audio, text="Duración (s):").grid(row=1, column=0, sticky="w")
        ttk.Entry(tab_audio, textvariable=dur_var).grid(row=1, column=1)

        # ---- REFERENCIA ----
        tab_ref = ttk.Frame(nb)
        nb.add(tab_ref, text="Referencia")

        ref_var = tk.StringVar(value=self.cfg["reference"]["path"])
        ttk.Entry(tab_ref, textvariable=ref_var, width=60).pack(anchor="w")

        # ---- CRITERIOS ----
        tab_eval = ttk.Frame(nb)
        nb.add(tab_eval, text="Evaluación")

        level_var = tk.StringVar(value=self.cfg["evaluation"]["level"])
        ttk.Combobox(tab_eval, textvariable=level_var,
                     values=["Bajo", "Medio", "Alto"],
                     state="readonly").pack(anchor="w")

        # ---- TB ----
        tab_tb = ttk.Frame(nb)
        nb.add(tab_tb, text="ThingsBoard")

        host_var = tk.StringVar(value=self.cfg["thingsboard"]["host"])
        port_var = tk.IntVar(value=self.cfg["thingsboard"]["port"])
        token_var = tk.StringVar(value=self.cfg["thingsboard"]["token"])

        ttk.Entry(tab_tb, textvariable=host_var).pack(anchor="w")
        ttk.Entry(tab_tb, textvariable=port_var).pack(anchor="w")
        ttk.Entry(tab_tb, textvariable=token_var, width=50).pack(anchor="w")

        def on_save():
            self.cfg["audio"]["fs"] = fs_var.get()
            self.cfg["audio"]["duration_s"] = dur_var.get()
            self.cfg["reference"]["path"] = ref_var.get()
            self.cfg["evaluation"]["level"] = level_var.get()
            self.cfg["thingsboard"]["host"] = host_var.get()
            self.cfg["thingsboard"]["port"] = port_var.get()
            self.cfg["thingsboard"]["token"] = token_var.get()
            save_config(self.cfg)
            messagebox.showinfo(APP_NAME, "Configuración guardada")
            w.destroy()

        ttk.Button(w, text="Guardar", command=on_save).pack(pady=6)

    # ================= EJECUCIÓN =================

    @ui_action
    def _run_once(self):
        dev_idx = pick_device_index_from_label(self.input_device_label.get())

        res, payload, out_path, x_ref, x_cur, fs = run_measurement(
            device_index=dev_idx
        )

        self.test_name.set(datetime.now().strftime("Test_%Y-%m-%d_%H-%M-%S"))
        self._set_eval(res.overall)

        self._clear_waves()
        self._plot_wave(self.ax_ref, x_ref, fs)
        self._plot_wave(self.ax_cur, x_cur, fs)
        self.canvas.draw_idle()

        messagebox.showinfo(
            APP_NAME,
            f"Resultado: {res.overall}\n\nJSON:\n{out_path}",
        )

# ===============================
# MAIN
# ===============================

def main():
    root = tb.Window(themename="flatly")
    app = AudioCinemaGUI(root)
    root.geometry("1000x620")
    root.mainloop()

if __name__ == "__main__":
    main()
