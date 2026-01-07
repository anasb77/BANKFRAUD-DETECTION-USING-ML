"""
FraudShield Pro - Bank Fraud Detection System
Complete Desktop Application with AI Integration
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import pandas as pd
import numpy as np
import joblib
import json
import os
import threading
import time
import winsound
from datetime import datetime
import csv
import base64
import hashlib
from cryptography.fernet import Fernet

# Matplotlib
try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MATPLOTLIB = True
except:
    HAS_MATPLOTLIB = False

# Gemini AI
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except:
    HAS_GEMINI = False

# App Info
APP_NAME = "FraudShield Pro"
VERSION = "3.0.0"

SOUND_CLICK = "Button-Click.wav"
SOUND_ERROR = "Button-Error.wav"
SOUND_AI = "AI-response.wav"
LOGO_PATH = "icon.png"
ICON_PATH = "icon.ico"
API_KEY_FILE = ".gemini_key.enc"
ENCRYPTION_KEY_FILE = ".enc_key"

# Theme
THEME = {
    'bg': '#0f0f14',
    'sidebar': '#161620',
    'card': '#1c1c28',
    'card_hover': '#252535',
    'input': '#22222e',
    'border': '#2d2d3d',
    'accent': '#6366f1',
    'accent_hover': '#818cf8',
    'green': '#22c55e',
    'green_dark': '#16a34a',
    'red': '#ef4444',
    'red_dark': '#dc2626',
    'yellow': '#eab308',
    'purple': '#a855f7',
    'text': '#f1f5f9',
    'text_dim': '#94a3b8',
    'text_muted': '#64748b',
}

ctk.set_appearance_mode("dark")


class SoundPlayer:
    @staticmethod
    def play(sound_file):
        def _p():
            try:
                if os.path.exists(sound_file):
                    winsound.PlaySound(sound_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
            except:
                pass
        threading.Thread(target=_p, daemon=True).start()

    @staticmethod
    def click(): SoundPlayer.play(SOUND_CLICK)
    @staticmethod
    def error(): SoundPlayer.play(SOUND_ERROR)
    @staticmethod
    def success(): SoundPlayer.play(SOUND_AI)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1400x900")
        self.minsize(1200, 750)
        self.configure(fg_color=THEME['bg'])
        if os.path.exists(ICON_PATH):
            self.iconbitmap(ICON_PATH)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # State
        self.model = None
        self.scaler = None
        self.encoders = {}
        self.features = []
        self.metrics = None
        self.sim_running = False
        self.batch_results = None
        self.gemini_model = None

        self._build_sidebar()
        self._build_main()
        self._load_model()
        self._init_gemini()
        self.show_dashboard()

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=260, corner_radius=0, fg_color=THEME['sidebar'])
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        # Logo
        logo_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        logo_frame.pack(fill='x', pady=(30, 20))
        if os.path.exists(LOGO_PATH):
            try:
                img = Image.open(LOGO_PATH)
                logo_img = ctk.CTkImage(light_image=img, dark_image=img, size=(60, 60))
                ctk.CTkLabel(logo_frame, image=logo_img, text="").pack()
            except:
                ctk.CTkLabel(logo_frame, text="🛡️", font=ctk.CTkFont(size=48)).pack()
        else:
            ctk.CTkLabel(logo_frame, text="🛡️", font=ctk.CTkFont(size=48)).pack()
        ctk.CTkLabel(logo_frame, text=APP_NAME, font=ctk.CTkFont(size=20, weight="bold"),
                    text_color=THEME['text']).pack(pady=(8, 2))
        ctk.CTkLabel(logo_frame, text="AI Fraud Detection", font=ctk.CTkFont(size=11),
                    text_color=THEME['text_muted']).pack()

        ctk.CTkFrame(self.sidebar, height=1, fg_color=THEME['border']).pack(fill='x', padx=20, pady=20)

        # Nav
        self.nav_btns = {}
        nav_items = [
            ("dashboard", "Dashboard", "📊"),
            ("analyze", "Batch Analyze", "📁"),
            ("simulate", "Simulate", "⚡"),
            ("generate", "Generate Data", "📝"),
            ("ai", "AI Overview", "🤖"),
            ("info", "Model Info", "ℹ️"),
        ]
        for key, label, icon in nav_items:
            btn = ctk.CTkButton(self.sidebar, text=f"  {icon}   {label}", height=44, corner_radius=8,
                               font=ctk.CTkFont(size=14), fg_color="transparent",
                               text_color=THEME['text_dim'], hover_color=THEME['card'], anchor="w",
                               command=lambda k=key: self._nav_click(k))
            btn.pack(fill='x', padx=15, pady=2)
            self.nav_btns[key] = btn

        # Status
        bottom = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        bottom.pack(side='bottom', fill='x', padx=15, pady=15)
        status_card = ctk.CTkFrame(bottom, fg_color=THEME['card'], corner_radius=10)
        status_card.pack(fill='x')
        self.status_dot = ctk.CTkLabel(status_card, text="●", text_color=THEME['yellow'], font=ctk.CTkFont(size=10))
        self.status_dot.pack(side='left', padx=(12, 6), pady=10)
        self.status_text = ctk.CTkLabel(status_card, text="Loading...", font=ctk.CTkFont(size=12), text_color=THEME['text_dim'])
        self.status_text.pack(side='left', pady=10)
        ctk.CTkLabel(status_card, text=f"v{VERSION}", font=ctk.CTkFont(size=10), text_color=THEME['text_muted']).pack(side='right', padx=12, pady=10)

    def _set_active_nav(self, key):
        for k, btn in self.nav_btns.items():
            if k == key:
                btn.configure(fg_color=THEME['accent'], text_color=THEME['text'])
            else:
                btn.configure(fg_color="transparent", text_color=THEME['text_dim'])

    def _nav_click(self, key):
        SoundPlayer.click()
        views = {
            "dashboard": self.show_dashboard,
            "analyze": self.show_analyze,
            "simulate": self.show_simulate,
            "generate": self.show_generate,
            "ai": self.show_ai,
            "info": self.show_info
        }
        views[key]()

    def _build_main(self):
        self.main = ctk.CTkFrame(self, corner_radius=0, fg_color=THEME['bg'])
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self.main, height=60, fg_color=THEME['sidebar'], corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        self.page_title = ctk.CTkLabel(header, text="Dashboard", font=ctk.CTkFont(size=18, weight="bold"), text_color=THEME['text'])
        self.page_title.pack(side='left', padx=25, pady=18)
        self.date_label = ctk.CTkLabel(header, text=datetime.now().strftime("%B %d, %Y"), font=ctk.CTkFont(size=12), text_color=THEME['text_muted'])
        self.date_label.pack(side='right', padx=25, pady=18)

        self.content = ctk.CTkScrollableFrame(self.main, fg_color="transparent")
        self.content.grid(row=1, column=0, sticky="nsew", padx=25, pady=25)
        self.content.grid_columnconfigure(0, weight=1)

    def _clear(self):
        for w in self.content.winfo_children():
            w.destroy()

    def _card(self, parent, **kw):
        return ctk.CTkFrame(parent, fg_color=THEME['card'], corner_radius=12, **kw)

    def _load_model(self):
        def _do():
            try:
                if os.path.exists('models/best_model.pkl'):
                    self.model = joblib.load('models/best_model.pkl')
                if os.path.exists('models/scaler.pkl'):
                    self.scaler = joblib.load('models/scaler.pkl')
                if os.path.exists('models/label_encoders.pkl'):
                    self.encoders = joblib.load('models/label_encoders.pkl')
                if os.path.exists('models/feature_names.json'):
                    with open('models/feature_names.json') as f:
                        self.features = json.load(f)
                if os.path.exists('models/model_metrics.json'):
                    with open('models/model_metrics.json') as f:
                        self.metrics = json.load(f)
                self.status_dot.configure(text_color=THEME['green'])
                self.status_text.configure(text="Model Ready")
            except Exception as e:
                self.status_dot.configure(text_color=THEME['red'])
                self.status_text.configure(text="Error")
        threading.Thread(target=_do, daemon=True).start()

    def _init_gemini(self):
        """Initialize Gemini with encrypted key if available"""
        if not HAS_GEMINI:
            return
        try:
            if self._has_stored_key():
                api_key = self._decrypt_key()
                if api_key:
                    genai.configure(api_key=api_key)
                    self.gemini_model = genai.GenerativeModel('gemini-pro')
        except:
            pass

    # =========================================
    # DASHBOARD
    # =========================================
    def show_dashboard(self):
        self._set_active_nav("dashboard")
        self.page_title.configure(text="📊 Dashboard")
        self._clear()

        # Stats
        if self.metrics:
            stats_frame = ctk.CTkFrame(self.content, fg_color="transparent")
            stats_frame.pack(fill='x', pady=(0, 20))
            stats_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

            best = self.metrics.get('best_model', 'N/A')
            m = self.metrics.get('results', {}).get(best, {})
            stats = [
                ("🎯", "AUC-ROC", f"{m.get('auc_roc', 0)*100:.1f}%", THEME['accent']),
                ("📊", "Precision", f"{m.get('precision', 0)*100:.1f}%", THEME['green']),
                ("🔍", "Recall", f"{m.get('recall', 0)*100:.1f}%", THEME['yellow']),
                ("⚡", "F1-Score", f"{m.get('f1', 0)*100:.1f}%", THEME['purple']),
            ]
            for i, (icon, name, val, color) in enumerate(stats):
                card = self._card(stats_frame)
                card.grid(row=0, column=i, padx=6, pady=5, sticky="ew")
                inner = ctk.CTkFrame(card, fg_color="transparent")
                inner.pack(fill='x', padx=18, pady=18)
                ctk.CTkLabel(inner, text=icon, font=ctk.CTkFont(size=28)).pack(anchor='w')
                ctk.CTkLabel(inner, text=val, font=ctk.CTkFont(size=24, weight="bold"), text_color=color).pack(anchor='w', pady=(8, 2))
                ctk.CTkLabel(inner, text=name, font=ctk.CTkFont(size=12), text_color=THEME['text_muted']).pack(anchor='w')

        # Quick actions
        actions = self._card(self.content)
        actions.pack(fill='x', pady=(0, 20))
        ctk.CTkLabel(actions, text="Quick Actions", font=ctk.CTkFont(size=16, weight="bold"), text_color=THEME['text']).pack(anchor='w', padx=20, pady=(20, 15))
        
        btn_frame = ctk.CTkFrame(actions, fg_color="transparent")
        btn_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        ctk.CTkButton(btn_frame, text="📁 Analyze CSV", height=50, width=180, corner_radius=10,
                     fg_color=THEME['accent'], hover_color=THEME['accent_hover'],
                     font=ctk.CTkFont(size=14, weight="bold"),
                     command=lambda: self._nav_click("analyze")).pack(side='left', padx=(0, 10))
        
        ctk.CTkButton(btn_frame, text="⚡ Run Simulation", height=50, width=180, corner_radius=10,
                     fg_color=THEME['green_dark'], hover_color=THEME['green'],
                     font=ctk.CTkFont(size=14, weight="bold"),
                     command=lambda: self._nav_click("simulate")).pack(side='left', padx=(0, 10))
        
        ctk.CTkButton(btn_frame, text="📝 Generate Test Data", height=50, width=180, corner_radius=10,
                     fg_color=THEME['purple'], hover_color="#9333ea",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     command=lambda: self._nav_click("generate")).pack(side='left', padx=(0, 10))

        if HAS_GEMINI and self.gemini_model:
            ctk.CTkButton(btn_frame, text="🤖 AI Analysis", height=50, width=180, corner_radius=10,
                         fg_color=THEME['red_dark'], hover_color=THEME['red'],
                         font=ctk.CTkFont(size=14, weight="bold"),
                         command=lambda: self._nav_click("ai")).pack(side='left')

    # =========================================
    # BATCH ANALYZE (CSV Upload)
    # =========================================
    def show_analyze(self):
        self._set_active_nav("analyze")
        self.page_title.configure(text="📁 Batch CSV Analysis")
        self._clear()

        # Instructions
        info = self._card(self.content)
        info.pack(fill='x', pady=(0, 15))
        ctk.CTkLabel(info, text="Upload a CSV file with transaction data. The system will analyze EVERY transaction and predict fraud probability.",
                    font=ctk.CTkFont(size=13), text_color=THEME['text_dim'], wraplength=800).pack(padx=20, pady=15)

        # Upload section
        upload = self._card(self.content)
        upload.pack(fill='x', pady=(0, 15))
        ctk.CTkLabel(upload, text="Upload Transactions", font=ctk.CTkFont(size=16, weight="bold"), text_color=THEME['text']).pack(anchor='w', padx=20, pady=(20, 15))

        btn_row = ctk.CTkFrame(upload, fg_color="transparent")
        btn_row.pack(fill='x', padx=20, pady=(0, 20))

        ctk.CTkButton(btn_row, text="📂 Select CSV File", height=50, width=200, corner_radius=10,
                     fg_color=THEME['accent'], hover_color=THEME['accent_hover'],
                     font=ctk.CTkFont(size=14, weight="bold"),
                     command=self._load_and_analyze_csv).pack(side='left', padx=(0, 15))

        self.file_label = ctk.CTkLabel(btn_row, text="No file selected", font=ctk.CTkFont(size=13), text_color=THEME['text_muted'])
        self.file_label.pack(side='left')

        # Progress
        self.progress_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        self.progress_frame.pack(fill='x', pady=(0, 15))

        # Results
        self.results_card = self._card(self.content)
        self.results_card.pack(fill='both', expand=True)
        ctk.CTkLabel(self.results_card, text="Results will appear here after analysis...",
                    font=ctk.CTkFont(size=14), text_color=THEME['text_muted']).pack(padx=20, pady=50)

    def _load_and_analyze_csv(self):
        SoundPlayer.click()
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path:
            return

        if not self.model:
            SoundPlayer.error()
            messagebox.showerror("Error", "Model not loaded!")
            return

        self.file_label.configure(text=os.path.basename(path))

        def analyze():
            try:
                df = pd.read_csv(path)
                total = len(df)

                # Clear progress frame
                for w in self.progress_frame.winfo_children():
                    w.destroy()

                prog_card = self._card(self.progress_frame)
                prog_card.pack(fill='x')
                ctk.CTkLabel(prog_card, text="Analyzing transactions...", font=ctk.CTkFont(size=14, weight="bold"), text_color=THEME['text']).pack(padx=20, pady=(15, 10))
                progress = ctk.CTkProgressBar(prog_card, width=400, height=12)
                progress.pack(padx=20, pady=(0, 5))
                progress.set(0)
                status_lbl = ctk.CTkLabel(prog_card, text="0%", font=ctk.CTkFont(size=12), text_color=THEME['text_dim'])
                status_lbl.pack(padx=20, pady=(0, 15))

                # Prepare data
                data = df.copy()
                cat_cols = ['TransactionType', 'Location', 'Channel', 'MerchantCategory', 'CustomerOccupation']
                for col in cat_cols:
                    if col in data.columns and col in self.encoders:
                        try:
                            data[col] = self.encoders[col].transform(data[col])
                        except:
                            data[col] = 0

                # Filter to required features
                if self.features:
                    available = [f for f in self.features if f in data.columns]
                    data = data[available]

                # Scale
                if self.scaler:
                    data_scaled = self.scaler.transform(data)
                else:
                    data_scaled = data.values

                # Predict in batches
                predictions = []
                probabilities = []
                batch_size = 100

                for i in range(0, total, batch_size):
                    batch = data_scaled[i:i+batch_size]
                    preds = self.model.predict(batch)
                    probs = self.model.predict_proba(batch)[:, 1]
                    predictions.extend(preds)
                    probabilities.extend(probs)

                    pct = min((i + batch_size) / total, 1.0)
                    progress.set(pct)
                    status_lbl.configure(text=f"{int(pct*100)}% ({min(i+batch_size, total)}/{total})")
                    self.update()

                # Add results to original df
                df['Prediction'] = predictions
                df['FraudProbability'] = probabilities
                df['RiskLevel'] = df['FraudProbability'].apply(lambda x: 'CRITICAL' if x > 0.8 else 'HIGH' if x > 0.6 else 'MEDIUM' if x > 0.3 else 'LOW')

                self.batch_results = df

                # Show results
                for w in self.results_card.winfo_children():
                    w.destroy()

                # Summary stats
                fraud_count = sum(predictions)
                fraud_pct = (fraud_count / total) * 100

                ctk.CTkLabel(self.results_card, text="Analysis Complete ✅", font=ctk.CTkFont(size=18, weight="bold"), text_color=THEME['green']).pack(anchor='w', padx=20, pady=(20, 15))

                stats = ctk.CTkFrame(self.results_card, fg_color="transparent")
                stats.pack(fill='x', padx=20)
                stats.grid_columnconfigure((0, 1, 2, 3), weight=1)

                stat_data = [
                    ("Total", str(total), THEME['text']),
                    ("Fraudulent", str(fraud_count), THEME['red']),
                    ("Legitimate", str(total - fraud_count), THEME['green']),
                    ("Fraud Rate", f"{fraud_pct:.2f}%", THEME['yellow']),
                ]
                for i, (lbl, val, col) in enumerate(stat_data):
                    s = ctk.CTkFrame(stats, fg_color=THEME['input'], corner_radius=10)
                    s.grid(row=0, column=i, padx=5, pady=5, sticky="ew")
                    ctk.CTkLabel(s, text=val, font=ctk.CTkFont(size=24, weight="bold"), text_color=col).pack(padx=15, pady=(15, 5))
                    ctk.CTkLabel(s, text=lbl, font=ctk.CTkFont(size=11), text_color=THEME['text_muted']).pack(padx=15, pady=(0, 15))

                # High risk transactions
                high_risk = df[df['FraudProbability'] > 0.6].head(10)
                if len(high_risk) > 0:
                    ctk.CTkLabel(self.results_card, text=f"⚠️ High Risk Transactions ({len(df[df['FraudProbability'] > 0.6])} total)",
                                font=ctk.CTkFont(size=15, weight="bold"), text_color=THEME['red']).pack(anchor='w', padx=20, pady=(20, 10))

                    for _, row in high_risk.iterrows():
                        item = ctk.CTkFrame(self.results_card, fg_color=THEME['input'], corner_radius=8, height=45)
                        item.pack(fill='x', padx=20, pady=3)
                        item.pack_propagate(False)
                        inner = ctk.CTkFrame(item, fg_color="transparent")
                        inner.pack(fill='both', expand=True, padx=15)
                        
                        amt = f"${row.get('TransactionAmount', 0):,.2f}" if 'TransactionAmount' in row else "N/A"
                        ctk.CTkLabel(inner, text=amt, font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME['text']).pack(side='left', pady=12)
                        
                        prob = f"{row['FraudProbability']*100:.1f}%"
                        ctk.CTkLabel(inner, text=prob, font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME['red']).pack(side='right', pady=12)
                        ctk.CTkLabel(inner, text=row.get('RiskLevel', ''), font=ctk.CTkFont(size=11), text_color=THEME['yellow']).pack(side='right', padx=15, pady=12)

                # Export button
                ctk.CTkButton(self.results_card, text="💾 Export Results to CSV", height=45, corner_radius=10,
                             fg_color=THEME['accent'], hover_color=THEME['accent_hover'],
                             font=ctk.CTkFont(size=14, weight="bold"),
                             command=self._export_results).pack(padx=20, pady=20)

                SoundPlayer.success()

            except Exception as e:
                SoundPlayer.error()
                messagebox.showerror("Error", str(e))

        threading.Thread(target=analyze, daemon=True).start()

    def _export_results(self):
        if self.batch_results is None:
            return
        SoundPlayer.click()
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            self.batch_results.to_csv(path, index=False)
            messagebox.showinfo("Success", f"Results exported to {path}")

    # =========================================
    # SIMULATION
    # =========================================
    def show_simulate(self):
        self._set_active_nav("simulate")
        self.page_title.configure(text="⚡ Transaction Simulation")
        self._clear()

        # Controls
        ctrl = self._card(self.content)
        ctrl.pack(fill='x', pady=(0, 15))
        ctk.CTkLabel(ctrl, text="Simulation Controls", font=ctk.CTkFont(size=16, weight="bold"), text_color=THEME['text']).pack(anchor='w', padx=20, pady=(20, 15))

        # Speed
        row1 = ctk.CTkFrame(ctrl, fg_color="transparent")
        row1.pack(fill='x', padx=20, pady=5)
        ctk.CTkLabel(row1, text="Transaction Speed (per second)", font=ctk.CTkFont(size=12), text_color=THEME['text_dim']).pack(side='left')
        self.speed_lbl = ctk.CTkLabel(row1, text="1.0", font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME['accent'])
        self.speed_lbl.pack(side='right')
        self.speed = ctk.CTkSlider(ctrl, from_=0.5, to=10, number_of_steps=19, fg_color=THEME['input'],
                                  progress_color=THEME['accent'], button_color=THEME['accent'],
                                  command=lambda v: self.speed_lbl.configure(text=f"{v:.1f}"))
        self.speed.set(1)
        self.speed.pack(fill='x', padx=20, pady=(5, 15))

        # Fraud probability
        row2 = ctk.CTkFrame(ctrl, fg_color="transparent")
        row2.pack(fill='x', padx=20, pady=5)
        ctk.CTkLabel(row2, text="Fraud Probability (%)", font=ctk.CTkFont(size=12), text_color=THEME['text_dim']).pack(side='left')
        self.fraud_pct_lbl = ctk.CTkLabel(row2, text="15%", font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME['red'])
        self.fraud_pct_lbl.pack(side='right')
        self.fraud_pct = ctk.CTkSlider(ctrl, from_=1, to=50, number_of_steps=49, fg_color=THEME['input'],
                                       progress_color=THEME['red'], button_color=THEME['red'],
                                       command=lambda v: self.fraud_pct_lbl.configure(text=f"{int(v)}%"))
        self.fraud_pct.set(15)
        self.fraud_pct.pack(fill='x', padx=20, pady=(5, 20))

        # Buttons
        btn_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        btn_row.pack(fill='x', padx=20, pady=(0, 20))
        self.start_btn = ctk.CTkButton(btn_row, text="▶ Start", width=120, height=45, corner_radius=10,
                                       fg_color=THEME['green_dark'], hover_color=THEME['green'],
                                       font=ctk.CTkFont(size=14, weight="bold"), command=self._start_sim)
        self.start_btn.pack(side='left', padx=(0, 10))
        self.stop_btn = ctk.CTkButton(btn_row, text="■ Stop", width=120, height=45, corner_radius=10,
                                      fg_color=THEME['red_dark'], hover_color=THEME['red'],
                                      font=ctk.CTkFont(size=14, weight="bold"), command=self._stop_sim, state="disabled")
        self.stop_btn.pack(side='left')

        # Stats display
        self.sim_stats = ctk.CTkFrame(ctrl, fg_color="transparent")
        self.sim_stats.pack(fill='x', padx=20, pady=(0, 20))
        self.sim_stats.grid_columnconfigure((0, 1, 2), weight=1)

        # Create frames first, then labels inside them
        total_frame = ctk.CTkFrame(self.sim_stats, fg_color=THEME['input'], corner_radius=10)
        total_frame.grid(row=0, column=0, padx=5, sticky="ew")
        self.sim_total = ctk.CTkLabel(total_frame, text="0", font=ctk.CTkFont(size=20, weight="bold"), text_color=THEME['text'])
        self.sim_total.pack(pady=(12, 3))
        ctk.CTkLabel(total_frame, text="Total", font=ctk.CTkFont(size=11), text_color=THEME['text_muted']).pack(pady=(0, 12))

        fraud_frame = ctk.CTkFrame(self.sim_stats, fg_color=THEME['input'], corner_radius=10)
        fraud_frame.grid(row=0, column=1, padx=5, sticky="ew")
        self.sim_fraud = ctk.CTkLabel(fraud_frame, text="0", font=ctk.CTkFont(size=20, weight="bold"), text_color=THEME['red'])
        self.sim_fraud.pack(pady=(12, 3))
        ctk.CTkLabel(fraud_frame, text="Fraud", font=ctk.CTkFont(size=11), text_color=THEME['text_muted']).pack(pady=(0, 12))

        legit_frame = ctk.CTkFrame(self.sim_stats, fg_color=THEME['input'], corner_radius=10)
        legit_frame.grid(row=0, column=2, padx=5, sticky="ew")
        self.sim_legit = ctk.CTkLabel(legit_frame, text="0", font=ctk.CTkFont(size=20, weight="bold"), text_color=THEME['green'])
        self.sim_legit.pack(pady=(12, 3))
        ctk.CTkLabel(legit_frame, text="Legit", font=ctk.CTkFont(size=11), text_color=THEME['text_muted']).pack(pady=(0, 12))

        # Log
        log_card = self._card(self.content)
        log_card.pack(fill='both', expand=True)
        ctk.CTkLabel(log_card, text="Transaction Log", font=ctk.CTkFont(size=14, weight="bold"), text_color=THEME['text']).pack(anchor='w', padx=20, pady=(18, 10))
        self.log = ctk.CTkTextbox(log_card, font=ctk.CTkFont(family="Consolas", size=11), fg_color=THEME['input'], corner_radius=8)
        self.log.pack(fill='both', expand=True, padx=20, pady=(0, 18))
        self.log.configure(state="disabled")

    def _start_sim(self):
        if not self.model:
            SoundPlayer.error()
            return
        SoundPlayer.click()
        self.sim_running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.sim_count = [0, 0, 0]  # total, fraud, legit

        def run():
            while self.sim_running:
                fraud_prob = self.fraud_pct.get() / 100
                is_fraud = np.random.random() < fraud_prob
                tx = self._make_tx(is_fraud)

                df = pd.DataFrame([tx])
                for c in ['TransactionType', 'Location', 'Channel', 'MerchantCategory', 'CustomerOccupation']:
                    if c in self.encoders:
                        try:
                            df[c] = self.encoders[c].transform(df[c])
                        except:
                            df[c] = 0
                df = df[self.features]
                scaled = self.scaler.transform(df)
                pred = self.model.predict(scaled)[0]
                prob = self.model.predict_proba(scaled)[0][1]

                self.sim_count[0] += 1
                if pred == 1:
                    self.sim_count[1] += 1
                else:
                    self.sim_count[2] += 1

                self.sim_total.configure(text=str(self.sim_count[0]))
                self.sim_fraud.configure(text=str(self.sim_count[1]))
                self.sim_legit.configure(text=str(self.sim_count[2]))

                ts = datetime.now().strftime("%H:%M:%S")
                status = "🚨 FRAUD" if pred == 1 else "✓ OK"
                line = f"[{ts}] #{self.sim_count[0]:04} | ${tx['TransactionAmount']:>7.2f} | {prob*100:>5.1f}% | {status}\n"
                self.log.configure(state="normal")
                self.log.insert("end", line)
                self.log.see("end")
                self.log.configure(state="disabled")
                time.sleep(1 / self.speed.get())
        threading.Thread(target=run, daemon=True).start()

    def _make_tx(self, fraud):
        locs = ['New York', 'Chicago', 'LA', 'Houston', 'Dallas']
        if fraud:
            return {'TransactionAmount': round(np.random.uniform(2500, 8000), 2), 'hour': np.random.choice([0, 1, 2, 3]),
                    'TransactionType': 'Transfer', 'Location': np.random.choice(locs), 'Channel': 'Online',
                    'MerchantCategory': 'Electronics', 'CustomerAge': 25, 'CustomerOccupation': 'Student',
                    'TransactionDuration': np.random.randint(3, 15), 'LoginAttempts': np.random.choice([4, 5]),
                    'AccountBalance': round(np.random.uniform(200, 1500), 2)}
        return {'TransactionAmount': round(np.random.uniform(20, 400), 2), 'hour': np.random.randint(9, 18),
                'TransactionType': 'Debit', 'Location': np.random.choice(locs), 'Channel': np.random.choice(['ATM', 'POS']),
                'MerchantCategory': 'Grocery', 'CustomerAge': np.random.randint(28, 55), 'CustomerOccupation': 'Engineer',
                'TransactionDuration': np.random.randint(30, 100), 'LoginAttempts': 1,
                'AccountBalance': round(np.random.uniform(4000, 12000), 2)}

    def _stop_sim(self):
        SoundPlayer.click()
        self.sim_running = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    # =========================================
    # GENERATE UNLABELED DATA
    # =========================================
    def show_generate(self):
        self._set_active_nav("generate")
        self.page_title.configure(text="📝 Generate Test Data")
        self._clear()

        card = self._card(self.content)
        card.pack(fill='x')
        ctk.CTkLabel(card, text="Generate Unlabeled Transaction Data", font=ctk.CTkFont(size=16, weight="bold"), text_color=THEME['text']).pack(anchor='w', padx=20, pady=(20, 10))
        ctk.CTkLabel(card, text="Create a CSV file with transactions (NO labels) that you can use to test the model.", font=ctk.CTkFont(size=13), text_color=THEME['text_dim']).pack(anchor='w', padx=20, pady=(0, 20))

        # Row count
        row1 = ctk.CTkFrame(card, fg_color="transparent")
        row1.pack(fill='x', padx=20, pady=5)
        ctk.CTkLabel(row1, text="Number of Transactions", font=ctk.CTkFont(size=12), text_color=THEME['text_dim']).pack(side='left')
        self.gen_count = ctk.CTkEntry(row1, width=100, font=ctk.CTkFont(size=13), fg_color=THEME['input'])
        self.gen_count.insert(0, "1000")
        self.gen_count.pack(side='right')

        # Fraud %
        row2 = ctk.CTkFrame(card, fg_color="transparent")
        row2.pack(fill='x', padx=20, pady=10)
        ctk.CTkLabel(row2, text="Approximate Fraud % (hidden)", font=ctk.CTkFont(size=12), text_color=THEME['text_dim']).pack(side='left')
        self.gen_fraud_lbl = ctk.CTkLabel(row2, text="10%", font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME['red'])
        self.gen_fraud_lbl.pack(side='right')
        self.gen_fraud = ctk.CTkSlider(card, from_=1, to=30, number_of_steps=29, fg_color=THEME['input'],
                                       progress_color=THEME['red'], button_color=THEME['red'],
                                       command=lambda v: self.gen_fraud_lbl.configure(text=f"{int(v)}%"))
        self.gen_fraud.set(10)
        self.gen_fraud.pack(fill='x', padx=20, pady=(0, 20))

        ctk.CTkButton(card, text="📥 Generate & Save CSV", height=50, corner_radius=10,
                     fg_color=THEME['accent'], hover_color=THEME['accent_hover'],
                     font=ctk.CTkFont(size=14, weight="bold"),
                     command=self._generate_data).pack(fill='x', padx=20, pady=(0, 20))

    def _generate_data(self):
        SoundPlayer.click()
        try:
            count = int(self.gen_count.get())
            fraud_pct = self.gen_fraud.get() / 100
        except:
            SoundPlayer.error()
            return

        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], initialfile="test_transactions.csv")
        if not path:
            return

        locs = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Dallas', 'Phoenix', 'San Antonio', 'San Diego']
        chans = ['Online', 'ATM', 'Branch', 'Mobile', 'POS']
        types = ['Debit', 'Credit', 'Transfer', 'Withdrawal', 'Deposit']
        cats = ['Grocery', 'Restaurant', 'Electronics', 'Travel', 'Gas Station', 'Clothing', 'Entertainment', 'Healthcare']
        jobs = ['Engineer', 'Doctor', 'Student', 'Retired', 'Teacher', 'Lawyer', 'Artist', 'Nurse', 'Businessman']

        rows = []
        for i in range(count):
            is_fraud = np.random.random() < fraud_pct
            if is_fraud:
                row = {
                    'TransactionID': f'TX{i+1:06}',
                    'TransactionAmount': round(np.random.uniform(2000, 9000), 2),
                    'hour': np.random.choice([0, 1, 2, 3, 4, 22, 23]),
                    'TransactionType': np.random.choice(types),
                    'Location': np.random.choice(locs),
                    'Channel': 'Online',
                    'MerchantCategory': np.random.choice(cats),
                    'CustomerAge': np.random.randint(18, 75),
                    'CustomerOccupation': np.random.choice(jobs),
                    'TransactionDuration': np.random.randint(1, 20),
                    'LoginAttempts': np.random.choice([3, 4, 5, 6]),
                    'AccountBalance': round(np.random.uniform(50, 3000), 2)
                }
            else:
                row = {
                    'TransactionID': f'TX{i+1:06}',
                    'TransactionAmount': round(np.random.uniform(5, 800), 2),
                    'hour': np.random.randint(7, 22),
                    'TransactionType': np.random.choice(types),
                    'Location': np.random.choice(locs),
                    'Channel': np.random.choice(chans),
                    'MerchantCategory': np.random.choice(cats),
                    'CustomerAge': np.random.randint(18, 75),
                    'CustomerOccupation': np.random.choice(jobs),
                    'TransactionDuration': np.random.randint(15, 180),
                    'LoginAttempts': 1,
                    'AccountBalance': round(np.random.uniform(500, 25000), 2)
                }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        SoundPlayer.success()
        messagebox.showinfo("Success", f"Generated {count} transactions and saved to:\n{path}")

    # =========================================
    # AI OVERVIEW (Gemini)
    # =========================================
    def show_ai(self):
        self._set_active_nav("ai")
        self.page_title.configure(text="🤖 AI Overview")
        self._clear()

        if not HAS_GEMINI:
            card = self._card(self.content)
            card.pack(fill='x')
            ctk.CTkLabel(card, text="Gemini AI not available", font=ctk.CTkFont(size=16, weight="bold"), text_color=THEME['yellow']).pack(padx=20, pady=(20, 10))
            ctk.CTkLabel(card, text="Install: pip install google-generativeai cryptography", font=ctk.CTkFont(size=13), text_color=THEME['text_dim']).pack(padx=20, pady=(0, 20))
            return

        # API Key Management Card
        key_card = self._card(self.content)
        key_card.pack(fill='x', pady=(0, 15))
        ctk.CTkLabel(key_card, text="🔐 Gemini API Key", font=ctk.CTkFont(size=16, weight="bold"), text_color=THEME['text']).pack(anchor='w', padx=20, pady=(20, 10))
        
        # Check if key exists
        has_key = self._has_stored_key()
        
        if has_key:
            # Key exists - show status and remove option
            status_frame = ctk.CTkFrame(key_card, fg_color=THEME['input'], corner_radius=10)
            status_frame.pack(fill='x', padx=20, pady=(0, 15))
            
            status_inner = ctk.CTkFrame(status_frame, fg_color="transparent")
            status_inner.pack(fill='x', padx=15, pady=15)
            
            ctk.CTkLabel(status_inner, text="✅ API Key Configured", font=ctk.CTkFont(size=14, weight="bold"), text_color=THEME['green']).pack(side='left')
            ctk.CTkLabel(status_inner, text="Key: ****" + "*"*20, font=ctk.CTkFont(size=12), text_color=THEME['text_muted']).pack(side='left', padx=20)
            
            ctk.CTkButton(status_inner, text="🗑️ Remove Key", width=120, height=35, corner_radius=8,
                         fg_color=THEME['red_dark'], hover_color=THEME['red'],
                         font=ctk.CTkFont(size=12, weight="bold"),
                         command=self._remove_api_key).pack(side='right')
            
            conn_status = "✅ Connected" if self.gemini_model else "❌ Connection failed"
            conn_color = THEME['green'] if self.gemini_model else THEME['red']
            ctk.CTkLabel(key_card, text=conn_status, font=ctk.CTkFont(size=12), text_color=conn_color).pack(anchor='w', padx=20, pady=(0, 15))
        else:
            # No key - show entry form
            ctk.CTkLabel(key_card, text="Enter your Gemini API key to enable AI features.", font=ctk.CTkFont(size=12), text_color=THEME['text_dim']).pack(anchor='w', padx=20, pady=(0, 10))
            ctk.CTkLabel(key_card, text="Get your key from: https://makersuite.google.com/app/apikey", font=ctk.CTkFont(size=11), text_color=THEME['accent']).pack(anchor='w', padx=20, pady=(0, 15))
            
            key_row = ctk.CTkFrame(key_card, fg_color="transparent")
            key_row.pack(fill='x', padx=20, pady=(0, 15))
            
            self.api_key_entry = ctk.CTkEntry(key_row, width=450, height=40, font=ctk.CTkFont(size=13), 
                                             fg_color=THEME['input'], placeholder_text="Paste your API key here...")
            self.api_key_entry.pack(side='left', padx=(0, 10))
            
            ctk.CTkButton(key_row, text="✓ Save & Validate", width=140, height=40, corner_radius=8,
                         fg_color=THEME['green_dark'], hover_color=THEME['green'],
                         font=ctk.CTkFont(size=13, weight="bold"),
                         command=self._save_api_key).pack(side='left')
            
            # Security note
            ctk.CTkLabel(key_card, text="🔒 Your key is encrypted locally and never shared.", font=ctk.CTkFont(size=11), text_color=THEME['text_muted']).pack(anchor='w', padx=20, pady=(0, 20))

        # AI Chat (only show if key is configured)
        if has_key and self.gemini_model:
            chat_card = self._card(self.content)
            chat_card.pack(fill='both', expand=True)
            ctk.CTkLabel(chat_card, text="Ask AI About Your Data", font=ctk.CTkFont(size=16, weight="bold"), text_color=THEME['text']).pack(anchor='w', padx=20, pady=(20, 10))

            self.ai_output = ctk.CTkTextbox(chat_card, font=ctk.CTkFont(size=12), fg_color=THEME['input'], corner_radius=8, wrap="word")
            self.ai_output.pack(fill='both', expand=True, padx=20, pady=(0, 10))
            self.ai_output.insert("1.0", "Ask me anything about fraud detection, your model performance, or transaction patterns...")
            self.ai_output.configure(state="disabled")

            input_row = ctk.CTkFrame(chat_card, fg_color="transparent")
            input_row.pack(fill='x', padx=20, pady=(0, 15))
            
            self.ai_input = ctk.CTkEntry(input_row, height=40, font=ctk.CTkFont(size=13), fg_color=THEME['input'], placeholder_text="Type your question...")
            self.ai_input.pack(side='left', fill='x', expand=True, padx=(0, 10))
            self.ai_input.bind("<Return>", lambda e: self._ask_ai())
            
            ctk.CTkButton(input_row, text="Ask AI", width=100, height=40, corner_radius=8,
                         fg_color=THEME['accent'], hover_color=THEME['accent_hover'],
                         font=ctk.CTkFont(size=13, weight="bold"),
                         command=self._ask_ai).pack(side='right')

            # Quick prompts
            prompts_frame = ctk.CTkFrame(chat_card, fg_color="transparent")
            prompts_frame.pack(fill='x', padx=20, pady=(0, 20))
            
            prompts = ["Explain the model metrics", "What makes a transaction fraudulent?", "How accurate is this model?"]
            for p in prompts:
                ctk.CTkButton(prompts_frame, text=p, height=32, corner_radius=6,
                             fg_color=THEME['input'], hover_color=THEME['card_hover'],
                             font=ctk.CTkFont(size=11),
                             command=lambda q=p: self._quick_prompt(q)).pack(side='left', padx=(0, 8))

    def _get_encryption_key(self):
        """Get or create encryption key"""
        if os.path.exists(ENCRYPTION_KEY_FILE):
            with open(ENCRYPTION_KEY_FILE, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(ENCRYPTION_KEY_FILE, 'wb') as f:
                f.write(key)
            return key

    def _encrypt_key(self, api_key):
        """Encrypt API key"""
        fernet = Fernet(self._get_encryption_key())
        return fernet.encrypt(api_key.encode())

    def _decrypt_key(self):
        """Decrypt stored API key"""
        try:
            fernet = Fernet(self._get_encryption_key())
            with open(API_KEY_FILE, 'rb') as f:
                encrypted = f.read()
            return fernet.decrypt(encrypted).decode()
        except:
            return None

    def _has_stored_key(self):
        """Check if encrypted key exists"""
        return os.path.exists(API_KEY_FILE) and os.path.exists(ENCRYPTION_KEY_FILE)

    def _save_api_key(self):
        """Validate and save API key with encryption"""
        if not hasattr(self, 'api_key_entry'):
            return
        
        key = self.api_key_entry.get().strip()
        if not key:
            SoundPlayer.error()
            messagebox.showwarning("Warning", "Please enter an API key.")
            return
        
        if len(key) < 20:
            SoundPlayer.error()
            messagebox.showerror("Invalid Key", "API key seems too short. Please check and try again.")
            return

        # Test the key
        def validate():
            try:
                genai.configure(api_key=key)
                model = genai.GenerativeModel('gemini-pro')
                # Quick test
                response = model.generate_content("Say 'OK' if you can read this.")
                if response.text:
                    # Key works! Encrypt and save
                    encrypted = self._encrypt_key(key)
                    with open(API_KEY_FILE, 'wb') as f:
                        f.write(encrypted)
                    
                    self.gemini_model = model
                    SoundPlayer.success()
                    messagebox.showinfo("Success", "✅ API key validated and saved!\n\nYour key is encrypted locally.")
                    self.show_ai()
            except Exception as e:
                SoundPlayer.error()
                messagebox.showerror("Validation Failed", f"Could not validate API key:\n{str(e)[:100]}")

        # Show loading
        self.api_key_entry.configure(state="disabled")
        threading.Thread(target=validate, daemon=True).start()

    def _remove_api_key(self):
        """Remove stored API key"""
        SoundPlayer.click()
        if messagebox.askyesno("Confirm Removal", "Are you sure you want to remove your API key?\n\nYou will need to enter it again to use AI features."):
            try:
                if os.path.exists(API_KEY_FILE):
                    os.remove(API_KEY_FILE)
                if os.path.exists(ENCRYPTION_KEY_FILE):
                    os.remove(ENCRYPTION_KEY_FILE)
                self.gemini_model = None
                SoundPlayer.success()
                messagebox.showinfo("Removed", "API key has been removed.")
                self.show_ai()
            except Exception as e:
                SoundPlayer.error()
                messagebox.showerror("Error", f"Could not remove key: {e}")

    def _quick_prompt(self, prompt):
        self.ai_input.delete(0, 'end')
        self.ai_input.insert(0, prompt)
        self._ask_ai()

    def _ask_ai(self):
        if not self.gemini_model:
            SoundPlayer.error()
            messagebox.showwarning("Warning", "Please enter and save your Gemini API key first.")
            return

        question = self.ai_input.get().strip()
        if not question:
            return

        SoundPlayer.click()
        self.ai_input.delete(0, 'end')

        # Build context with safe metric access
        if self.metrics and 'results' in self.metrics and self.metrics.get('best_model') in self.metrics['results']:
            best = self.metrics['best_model']
            m = self.metrics['results'][best]
            model_info = f"""- Best Model: {best}
- AUC-ROC: {m.get('auc_roc', 0)*100:.2f}%
- Precision: {m.get('precision', 0)*100:.2f}%
- Recall: {m.get('recall', 0)*100:.2f}%
- F1-Score: {m.get('f1', 0)*100:.2f}%
- Dataset Size: {self.metrics.get('dataset_size', 'Unknown')}
- Fraud Rate: {self.metrics.get('fraud_rate', 'Unknown')}%"""
        else:
            model_info = "Model metrics not available."
        
        features_info = ', '.join(self.features) if self.features else 'Unknown'
        
        context = f"""You are an AI assistant for a bank fraud detection system called FraudShield Pro.

Model Information:
{model_info}

Features used: {features_info}

User Question: {question}

Provide a helpful, concise answer about fraud detection or the model."""

        def ask():
            try:
                response = self.gemini_model.generate_content(context)
                answer = response.text

                self.ai_output.configure(state="normal")
                self.ai_output.delete("1.0", "end")
                self.ai_output.insert("1.0", f"Q: {question}\n\n{answer}")
                self.ai_output.configure(state="disabled")
                SoundPlayer.success()
            except Exception as e:
                self.ai_output.configure(state="normal")
                self.ai_output.delete("1.0", "end")
                self.ai_output.insert("1.0", f"Error: {str(e)}")
                self.ai_output.configure(state="disabled")
                SoundPlayer.error()

        threading.Thread(target=ask, daemon=True).start()

    # =========================================
    # MODEL INFO
    # =========================================
    def show_info(self):
        self._set_active_nav("info")
        self.page_title.configure(text="ℹ️ Model Information")
        self._clear()

        if not self.metrics:
            card = self._card(self.content)
            card.pack(fill='x')
            ctk.CTkLabel(card, text="No model loaded", font=ctk.CTkFont(size=16), text_color=THEME['text_muted']).pack(padx=20, pady=50)
            return

        # Hero
        card = self._card(self.content)
        card.pack(fill='x', pady=(0, 15))
        hero = ctk.CTkFrame(card, fg_color="transparent")
        hero.pack(fill='x', padx=25, pady=25)
        ctk.CTkLabel(hero, text="🏆", font=ctk.CTkFont(size=48)).pack(side='left', padx=(0, 18))
        info = ctk.CTkFrame(hero, fg_color="transparent")
        info.pack(side='left')
        ctk.CTkLabel(info, text=self.metrics.get('best_model', 'N/A'), font=ctk.CTkFont(size=26, weight="bold"), text_color=THEME['text']).pack(anchor='w')
        ctk.CTkLabel(info, text="Best Performing Model", font=ctk.CTkFont(size=12), text_color=THEME['text_muted']).pack(anchor='w')

        # Metrics
        grid = ctk.CTkFrame(card, fg_color="transparent")
        grid.pack(fill='x', padx=25, pady=(0, 25))
        grid.grid_columnconfigure((0, 1, 2), weight=1)

        best = self.metrics['best_model']
        m = self.metrics['results'][best]
        metrics = [
            ("Accuracy", f"{m['accuracy']*100:.2f}%", THEME['accent']),
            ("Precision", f"{m['precision']*100:.2f}%", THEME['green']),
            ("Recall", f"{m['recall']*100:.2f}%", THEME['yellow']),
            ("F1-Score", f"{m['f1']*100:.2f}%", THEME['purple']),
            ("AUC-ROC", f"{m['auc_roc']*100:.2f}%", THEME['red']),
            ("Avg Precision", f"{m['avg_precision']*100:.2f}%", "#ec4899"),
        ]
        for i, (name, val, col) in enumerate(metrics):
            cell = ctk.CTkFrame(grid, fg_color=THEME['input'], corner_radius=10)
            cell.grid(row=i//3, column=i%3, padx=4, pady=4, sticky="ew")
            ctk.CTkLabel(cell, text=name, font=ctk.CTkFont(size=11), text_color=THEME['text_muted']).pack(padx=18, pady=(14, 3))
            ctk.CTkLabel(cell, text=val, font=ctk.CTkFont(size=20, weight="bold"), text_color=col).pack(padx=18, pady=(0, 14))


if __name__ == "__main__":
    app = App()
    app.mainloop()
