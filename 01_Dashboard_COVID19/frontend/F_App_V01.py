# frontend.py - Version Finale Complète
import ttkbootstrap as tb
from ttkbootstrap import ttk
import tkinter as tk
import requests
from pandas import to_datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os, time, threading, webbrowser, tempfile, logging
from urllib.parse import quote
from datetime import datetime, timedelta
from tkinter import filedialog, messagebox, scrolledtext
import json
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter


# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('covid_dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path for ml_analysis import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

try:
    from ml_analysis import enhanced_predict_cases_df
    ML_AVAILABLE = True
    logger.info("Module ML chargé avec succès")
except ImportError as e:
    ML_AVAILABLE = False
    logger.warning(f"Module ML non disponible: {e}")

class COVID19Dashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("🦠 Dashboard COVID-19 - Analyse et Prédictions Avancées")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Configuration des couleurs personnalisées
        self.colors = {
            'total_cases': '#ff6b6b',
            'total_deaths': '#4a4a4a',
            'new_cases': '#ffa726',
            'new_deaths': '#ab47bc',
            'predictions': '#26c6da'
        }
        
        # Variables dynamiques
        self.api_url_var = tk.StringVar(value="http://127.0.0.1:8000")
        self.REQUEST_TIMEOUT = 15
        self.cache = {}
        self.cache_timeout = 300  # secondes
        self.countries = []
        self.current_data = None
        self.current_df = None
        self.current_predictions = None
        self.logs = []
        
        # Interface
        self.create_widgets()
        self.test_api_connection()
        self.load_countries()
        
        # Protocole de fermeture
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def log_message(self, level: str, message: str):
        """Ajoute un message aux logs internes"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {level.upper()}: {message}"
        self.logs.append(log_entry)
        
        # Garder seulement les 100 derniers logs
        if len(self.logs) > 100:
            self.logs = self.logs[-100:]
        
        # Log aussi avec le logger Python
        if level == 'error':
            logger.error(message)
        elif level == 'warning':
            logger.warning(message)
        else:
            logger.info(message)

    def create_widgets(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(2, weight=1)

        self.create_left_panel(main)
        self.create_right_panel(main)
        self.create_status_bar()

    def create_left_panel(self, parent):
        left = ttk.Labelframe(parent, text="🛠️ Configuration", padding=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0,10))

        # URL API dynamique : Label, Entry et Bouton sur la même ligne
        api_frame = ttk.Labelframe(left, text="🔗 Configuration API", padding=5)
        api_frame.pack(fill="x", pady=(0,10))

        row = ttk.Frame(api_frame)
        row.pack(fill="x", pady=2)

        ttk.Label(row, text="URL API:").pack(side="left", padx=(0,5))
        self.api_entry = ttk.Entry(row, textvariable=self.api_url_var, width=15)
        self.api_entry.pack(side="left", fill="x", expand=True)
        self.api_entry.bind('<Return>', lambda e: self.test_api_connection())

        ttk.Button(row, text="▶️ Tester", command=self.test_api_connection).pack(side="left", padx=(5,0))


        # État API amélioré
        status_api = ttk.Labelframe(
            left,
            text="🔧 État de l'API",
            padding=5,
            bootstyle="danger"
        )
        status_api.pack(fill="x", pady=(0,10))

        self.api_status_label = ttk.Label(
            status_api,
            text="❌ Non testé",
            bootstyle="danger"
        )
        self.api_status_label.pack(anchor="w", padx=5, pady=2)
        
        self.api_response_time_label = ttk.Label(
            status_api,
            text="Temps de réponse: -",
            bootstyle="secondary"
        )
        self.api_response_time_label.pack(anchor="w", padx=5, pady=2)

        # Sélection pays améliorée
        cf = ttk.Labelframe(
            left,
            text="🌍 Sélection des Pays",
            padding=5,
            bootstyle="info"
        )
        cf.pack(fill="x", pady=(0,10))

        # Ligne 1: Statut et rechargement
        country_controls = ttk.Frame(cf)
        country_controls.pack(fill="x", pady=2)
        
        ttk.Button(
            country_controls,
            text="🔄 Recharger",
            command=self.load_countries,
            bootstyle="info-outline"
        ).pack(side="left", padx=(0,5))

        self.countries_label = ttk.Label(
            country_controls,
            text="Chargement...",
            bootstyle="info"
        )
        self.countries_label.pack(side="left")

        # Ligne 2 : Combobox avec recherche en temps réel
        line_frame = ttk.Frame(cf)
        line_frame.pack(fill='x', pady=(5,2))

        ttk.Label(line_frame, text="Pays:").pack(side='left', padx=(0,5))

        self.country_var = tk.StringVar()
        self.country_combo = ttk.Combobox(
            line_frame,
            textvariable=self.country_var,
            state='normal',      # passe en mode éditable pour taper/rechercher
            width=30,
            bootstyle="info"
        )
        self.country_combo.pack(side='left', fill='x', expand=True)

        # Filtre dès que l'utilisateur tape
        self.country_combo.bind('<KeyRelease>', self._on_country_search)
        # Chargement des données quand il sélectionne un élément
        self.country_combo.bind('<<ComboboxSelected>>', lambda e: self.load_country_data())


       # Filtres de date et limite sur une seule ligne
        df_frame = ttk.Labelframe(left, text="📅 Filtres de Date", padding=5)
        df_frame.pack(fill="x", pady=(0,10))

        # Ligne 1 : Date de début et Entry côte à côte
        row1 = ttk.Frame(df_frame)
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="Date de début:").pack(side="left", padx=(0,5))
        self.start_date_var = tk.StringVar()
        start_entry = ttk.Entry(row1, textvariable=self.start_date_var, width=20)
        start_entry.pack(side="left", padx=(0,5))
        start_entry.insert(0, "Format: 2020-01-01")
        start_entry.bind('<FocusIn>', lambda e, ph="Format: 2020-01-01": self.clear_placeholder(e, ph))

        # Ligne 2 : Date de fin et Entry côte à côte
        row2 = ttk.Frame(df_frame)
        row2.pack(fill="x", pady=2)
        ttk.Label(row2, text="Date de fin:").pack(side="left", padx=(0,17))
        self.end_date_var = tk.StringVar()
        end_entry = ttk.Entry(row2, textvariable=self.end_date_var, width=20)
        end_entry.pack(side="left", padx=(0,5))
        end_entry.insert(0, "Format: 2023-12-31")
        end_entry.bind('<FocusIn>', lambda e, ph="Format: 2023-12-31": self.clear_placeholder(e, ph))

        # Ligne 3 : Limite d'enregistrements et Spinbox côte à côte
        row3 = ttk.Frame(df_frame)
        row3.pack(fill="x", pady=2)
        ttk.Label(row3, text="Limite d'enregistrements:").pack(side="left", padx=(0,5))
        self.limit_var = tk.StringVar(value="0")
        ttk.Spinbox(
            row3,
            from_=0,
            to=10000,
            increment=100,
            textvariable=self.limit_var,
            width=8
        ).pack(side="left", padx=(0,5))
        ttk.Label(row3, text="(0 = aucune limite)").pack(side="left")

        # Ligne 4 : Bouton Charger les Données
        ttk.Button(
            df_frame,
            text="📊 Charger les Données",
            command=self.load_country_data,
            bootstyle="success"
        ).pack(fill="x", pady=(10,0))


        # Prédictions ML améliorées
        if ML_AVAILABLE:
            mlf = ttk.Labelframe(left, text="🔮 Prédictions ML", padding=5, bootstyle="warning")
            mlf.pack(fill="x", pady=(0,10))
            
            # ttk.Label(mlf, text="✅ Module ML disponible", bootstyle="success").pack(anchor="w")
            
            # Modèle : Label et Combobox sur la même ligne
            row = ttk.Frame(mlf)
            row.pack(fill='x', pady=(5,2))
            ttk.Label(row, text="Modèle:").pack(side='left', padx=(0,5))
            self.model_type_var = tk.StringVar(value="auto")
            ttk.Combobox(
                row,
                values=["auto", "linear", "polynomial", "ridge", "lasso"],
                textvariable=self.model_type_var,
                state="readonly",
                width=20
            ).pack(side='left', fill='x', expand=True)

            
            # Jours à prédire : label et spinbox sur la même ligne
            row = ttk.Frame(mlf)
            row.pack(fill='x', pady=(5,2))
            ttk.Label(row, text="Jours à prédire:").pack(side='left', padx=(0,5))
            self.days_var = tk.StringVar(value="7")
            ttk.Spinbox(
                row,
                from_=1,
                to=90,
                textvariable=self.days_var,
                width=10
            ).pack(side='left')

            
            # Options avancées
            self.show_confidence_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                mlf, 
                text="Afficher intervalles de confiance",
                variable=self.show_confidence_var
            ).pack(anchor="w", pady=2)
            
            ttk.Button(
                mlf, 
                text="🚀 Générer Prédictions", 
                command=self.generate_predictions,
                bootstyle="warning"
            ).pack(fill="x", pady=5)
        else:
            mlf = ttk.Labelframe(left, text="🔮 Prédictions ML", padding=5, bootstyle="danger")
            mlf.pack(fill="x", pady=(0,10))
            ttk.Label(mlf, text="❌ Module ML non disponible", bootstyle="danger").pack()
            ttk.Label(mlf, text="Installez scikit-learn pour activer", font=('Arial', 8)).pack()

        # Export amélioré
        exf = ttk.Labelframe(left, text="💾 Export de Données", padding=5, bootstyle="secondary")
        exf.pack(fill="x", pady=(0,10))
        
        export_row1 = ttk.Frame(exf)
        export_row1.pack(fill="x", pady=2)
        ttk.Button(export_row1, text="📥 CSV", command=self.export_csv).pack(side="left", padx=(0,5))
        ttk.Button(export_row1, text="📥 JSON", command=self.export_json).pack(side="left", padx=5)
        ttk.Button(export_row1, text="📥 Excel", command=self.export_excel).pack(side="left", padx=5)
        
        export_row2 = ttk.Frame(exf)
        export_row2.pack(fill="x", pady=(5,0))
        ttk.Button(export_row2, text="🖼️ Graphique", command=self.export_chart).pack(side="left", padx=(0,5))
        if ML_AVAILABLE:
            ttk.Button(export_row2, text="🔮 Prédictions", command=self.export_predictions).pack(side="left", padx=5)

        # Debug et maintenance
        dbg = ttk.Labelframe(
            left,
            text="🔧 Debug & Maintenance",
            padding=5,
            bootstyle="dark"
        )
        dbg.pack(fill="x")

        debug_row1 = ttk.Frame(dbg)
        debug_row1.pack(fill="x", pady=2)
        
        ttk.Button(
            debug_row1,
            text="🔄 Vider Cache",
            command=self.clear_cache,
            bootstyle="dark-outline"
        ).pack(side="left", padx=(0,5))

        ttk.Button(
            debug_row1,
            text="📝 Voir Logs",
            command=self.show_logs,
            bootstyle="dark-outline"
        ).pack(side="left", padx=5)
        
        # Statistiques du cache
        self.cache_stats_label = ttk.Label(
            dbg,
            text="Cache: 0 entrées",
            font=('Arial', 8)
        )
        self.cache_stats_label.pack(anchor="w", pady=(5,0))

    def _on_country_search(self, event):
        """Filtre la liste des pays au fur et à mesure de la frappe."""
        text = self.country_var.get().lower()
        # self.countries doit contenir la liste complète des pays chargés
        filtered = [c for c in self.countries if text in c.lower()]
        # Met à jour les valeurs disponibles dans la combobox
        self.country_combo['values'] = filtered
        # Ouvre automatiquement la liste pour voir les résultats
        if filtered:
            self.country_combo.event_generate('<Down>')

    def clear_placeholder(self, event, placeholder):
        """Efface le placeholder lors du focus"""
        if event.widget.get() == placeholder:
            event.widget.delete(0, tk.END)

    def create_right_panel(self, parent):
        self.nb = ttk.Notebook(parent)
        self.nb.grid(row=0, column=1, rowspan=3, sticky="nsew")
        
        # Onglets
        self.data_tab = ttk.Frame(self.nb)
        self.chart_tab = ttk.Frame(self.nb)
        self.pred_tab = ttk.Frame(self.nb)
        self.compare_tab = ttk.Frame(self.nb)
        
        self.nb.add(self.data_tab, text="📊 Données")
        self.nb.add(self.chart_tab, text="📈 Graphiques")
        self.nb.add(self.pred_tab, text="🔮 Prédictions")
        self.nb.add(self.compare_tab, text="⚖️ Comparaison")
        
        self.create_data_display()
        self.create_charts_display()
        self.create_predictions_display()
        self.create_comparison_display()

    def create_data_display(self):
        # Résumé statistique amélioré
        sumf = ttk.Labelframe(self.data_tab, text="📈 Résumé Statistique", padding=10)
        sumf.pack(fill="x", padx=10, pady=5)
        
        # Première ligne
        row1 = ttk.Frame(sumf)
        row1.pack(fill="x")
        self.records_label = ttk.Label(row1, text="Enregistrements: -", font=('Arial', 10, 'bold'))
        self.period_label = ttk.Label(row1, text="Période: -", font=('Arial', 10))
        self.records_label.pack(side="left", padx=(0,20))
        self.period_label.pack(side="left")
        
        # Deuxième ligne
        row2 = ttk.Frame(sumf)
        row2.pack(fill="x", pady=(5,0))
        self.cases_label = ttk.Label(row2, text="Derniers cas: -", bootstyle="danger")
        self.deaths_label = ttk.Label(row2, text="Derniers décès: -", bootstyle="dark")
        self.cases_label.pack(side="left", padx=(0,20))
        self.deaths_label.pack(side="left")
        
        # Nouvelles statistiques
        row3 = ttk.Frame(sumf)
        row3.pack(fill="x", pady=(5,0))
        self.max_cases_label = ttk.Label(row3, text="Pic quotidien: -", bootstyle="warning")
        self.avg_cases_label = ttk.Label(row3, text="Moyenne/jour: -", bootstyle="info")
        self.max_cases_label.pack(side="left", padx=(0,20))
        self.avg_cases_label.pack(side="left")

        # Aperçu des données avec recherche
        prevf = ttk.Labelframe(self.data_tab, text="🔍 Aperçu des Données", padding=10)
        prevf.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Barre d'outils
        toolbar = ttk.Frame(prevf)
        toolbar.pack(fill="x", pady=(0,5))
        
        ttk.Label(toolbar, text="Recherche:").pack(side="left")
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(toolbar, textvariable=self.search_var, width=20)
        search_entry.pack(side="left", padx=5)
        search_entry.bind('<KeyRelease>', self.filter_data)
        
        ttk.Button(toolbar, text="🔄 Actualiser", command=self.refresh_data_display).pack(side="right")
        
        # Treeview amélioré
        tree_frame = ttk.Frame(prevf)
        tree_frame.pack(fill="both", expand=True)
        
        cols = ['Date', 'Cas Totaux', 'Décès Totaux', 'Nouveaux Cas', 'Nouveaux Décès', 'Taux Mortalité (%)']
        self.data_tree = ttk.Treeview(tree_frame, columns=cols, show='headings', height=15)
        
        # Configuration des colonnes
        col_widths = [100, 120, 120, 120, 120, 140]
        for i, (col, width) in enumerate(zip(cols, col_widths)):
            self.data_tree.heading(col, text=col, command=lambda c=i: self.sort_treeview(c))
            self.data_tree.column(col, width=width, anchor='center')
        
        # Scrollbars
        v_scroll = ttk.Scrollbar(tree_frame, orient='vertical', command=self.data_tree.yview)
        h_scroll = ttk.Scrollbar(tree_frame, orient='horizontal', command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        # Placement
        self.data_tree.grid(row=0, column=0, sticky='nsew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        h_scroll.grid(row=1, column=0, sticky='ew')
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

    def create_charts_display(self):
        # Panneau de contrôle des graphiques
        control_frame = ttk.Frame(self.chart_tab)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # Ligne 1: Type et style
        row1 = ttk.Frame(control_frame)
        row1.pack(fill='x', pady=2)
        
        ttk.Label(row1, text="Type:").pack(side='left')
        self.chart_type = tk.StringVar(value='line')
        ttk.Combobox(
            row1, 
            textvariable=self.chart_type, 
            values=['line', 'bar', 'scatter', 'area'], 
            state='readonly', 
            width=10
        ).pack(side='left', padx=5)
        
        ttk.Label(row1, text="Style:").pack(side='left', padx=(20,0))
        self.chart_style = tk.StringVar(value='seaborn-v0_8')
        style_combo = ttk.Combobox(
            row1, 
            textvariable=self.chart_style,
            values=['seaborn-v0_8', 'ggplot', 'classic', 'dark_background'],
            state='readonly',
            width=15
        )
        style_combo.pack(side='left', padx=5)
        
        # Ligne 2: Options d'affichage
        row2 = ttk.Frame(control_frame)
        row2.pack(fill='x', pady=2)
        
        self.show_grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row2, text="Grille", variable=self.show_grid_var).pack(side='left')
        
        self.show_legend_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row2, text="Légende", variable=self.show_legend_var).pack(side='left', padx=10)
        
        self.log_scale_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text="Échelle log", variable=self.log_scale_var).pack(side='left', padx=10)
        
        # Boutons d'action
        row3 = ttk.Frame(control_frame)
        row3.pack(fill='x', pady=5)
        
        ttk.Button(row3, text="📈 Actualiser", command=self.update_charts, bootstyle="primary").pack(side='left')
        ttk.Button(row3, text="🌐 Plotly Interactif", command=self.open_plotly_chart, bootstyle="info").pack(side='left', padx=5)
        ttk.Button(row3, text="🔍 Zoom", command=self.toggle_chart_zoom).pack(side='left', padx=5)
        
        # Zone d'affichage des graphiques
        self.chart_frame = ttk.Frame(self.chart_tab)
        self.chart_frame.pack(fill='both', expand=True, padx=10, pady=5)

    def create_predictions_display(self):
        # Informations sur le modèle
        model_frame = ttk.Labelframe(self.pred_tab, text="🤖 Informations du Modèle", padding=10)
        model_frame.pack(fill='x', padx=10, pady=5)
        
        self.model_info_label = ttk.Label(model_frame, text="Modèle: - | R²: - | MAE: -", font=('Arial', 10, 'bold'))
        self.model_info_label.pack(anchor='w')
        
        self.model_details_label = ttk.Label(model_frame, text="", font=('Arial', 9))
        self.model_details_label.pack(anchor='w', pady=(5,0))
        
        # Résultats des prédictions
        pred_frame = ttk.Labelframe(self.pred_tab, text="📊 Résultats des Prédictions", padding=10)
        pred_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Tableau des prédictions
        tree_container = ttk.Frame(pred_frame)
        tree_container.pack(fill='both', expand=True)
        
        cols = ['Date', 'Cas Prédits', 'Borne Inf.', 'Borne Sup.', 'Confiance (%)']
        self.pred_tree = ttk.Treeview(tree_container, columns=cols, show='headings', height=12)
        
        for col in cols:
            self.pred_tree.heading(col, text=col)
            self.pred_tree.column(col, width=120, anchor='center')
        
        pred_scroll = ttk.Scrollbar(tree_container, orient='vertical', command=self.pred_tree.yview)
        self.pred_tree.configure(yscrollcommand=pred_scroll.set)
        
        self.pred_tree.pack(side='left', fill='both', expand=True)
        pred_scroll.pack(side='right', fill='y')
        
        # Graphique des prédictions
        self.pred_chart_frame = ttk.Frame(pred_frame)
        self.pred_chart_frame.pack(fill='x', pady=(10,0))












































    def generate_comparison(self):
        """Trace statiquement la métrique pour le pays origine + pays sélectionnés."""
        base = self.country_var.get().strip()
        chosen = [self.comparison_listbox.get(i) for i in self.comparison_listbox.curselection()]
        if not base or not chosen:
            messagebox.showwarning("Comparaison", "Sélectionnez un pays origine et au moins un comparatif.")
            return

        metric = self.comp_metric_var.get()
        fig = Figure(figsize=(12,6), dpi=100)
        ax = fig.add_subplot(111)

        for country in [base] + chosen:
            df = self.get_country_df(country)
            if df.empty:
                continue

            # Cas prédits
            if metric == 'predicted_cases':
                days_list, preds, _, low, high, _, _ = enhanced_predict_cases_df(
                    df,
                    days_ahead=int(self.days_var.get()),
                    model_type=self.model_type_var.get()
                )
                dates = pd.to_datetime(days_list)  # <-- conversion essentielle
                ax.plot(dates, preds, label=f"{country} (prédits)")
                if self.show_confidence_var.get():
                    ax.fill_between(dates, low, high, alpha=0.2)
                continue

            # Cas bruts (total_cases ou new_cases)
            if metric in df.columns:
                ax.plot(df['date'], df[metric], label=country)

        ax.set_title(f"Comparaison: {metric.replace('_',' ')}")
        ax.set_xlabel("Date")
        ax.set_ylabel(metric.replace('_',' ').capitalize())

        # **Formatter les dates proprement**
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()  # rotation automatique

        ax.legend()
        ax.grid(True)

        for w in self.comparison_chart_frame.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, self.comparison_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def open_plotly_comparison(self):
        """Ouvre un graphique Plotly interactif de la comparaison."""
        base = self.country_var.get().strip()
        chosen = [self.comparison_listbox.get(i) for i in self.comparison_listbox.curselection()]
        if not base or not chosen:
            messagebox.showwarning("Comparaison", "Sélectionnez un pays origine et au moins un comparatif.")
            return

        metric = self.comp_metric_var.get()
        fig = make_subplots(rows=1, cols=1, specs=[[{'type':'scatter'}]])

        for country in [base] + chosen:
            df = self.get_country_df(country)
            if df.empty:
                continue

            if metric == 'predicted_cases':
                days_list, preds, _, low, high, _, _ = enhanced_predict_cases_df(
                    df,
                    days_ahead=int(self.days_var.get()),
                    model_type=self.model_type_var.get()
                )
                dates = pd.to_datetime(days_list)
                fig.add_trace(go.Scatter(
                    x=dates, y=preds, mode='lines+markers', name=f"{country} (prédits)"
                ))
                if self.show_confidence_var.get():
                    fig.add_trace(go.Scatter(x=dates, y=low, mode='lines', showlegend=False, line={'dash':'dash'}))
                    fig.add_trace(go.Scatter(x=dates, y=high, mode='lines', showlegend=False, line={'dash':'dash'}))
                continue

            if metric in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['date'], y=df[metric], mode='lines+markers', name=country
                ))

        fig.update_layout(
            title=f"Comparaison interactive: {metric.replace('_',' ')}",
            xaxis_title="Date",
            yaxis_title=metric.replace('_',' ').capitalize(),
            template='plotly_white',
            hovermode='x unified'
        )

        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False)
        fig.write_html(tmp.name, include_plotlyjs='cdn')
        tmp.close()
        webbrowser.open(f'file://{tmp.name}')
        self.update_status("Comparaison Plotly ouverte")

    def get_country_df(self, country: str) -> pd.DataFrame:
        """Récupère et met en cache les données d’un pays, retourne un DataFrame trié et enrichi."""
        key = f"{country}_[]"
        if key not in self.cache:
            try:
                url = self.api_url_var.get().rstrip('/') + f"/stats/{quote(country)}"
                response = requests.get(url, timeout=self.REQUEST_TIMEOUT)
                response.raise_for_status()
                data = response.json()
                self.cache[key] = (time.time(), data)
            except Exception:
                return pd.DataFrame()

        df = pd.DataFrame(self.cache[key][1].get('data', []))
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.sort_values('date', inplace=True)

        # **Calcul des nouveaux cas et nouveaux décès à partir des totaux**
        if 'total_cases' in df.columns:
            df['new_cases'] = df['total_cases'].diff().fillna(0).astype(int)
        if 'total_deaths' in df.columns:
            df['new_deaths'] = df['total_deaths'].diff().fillna(0).astype(int)

        return df

    def create_comparison_display(self):
        """Onglet pour comparer plusieurs pays avec choix de métrique."""
        comp_frame = ttk.Labelframe(self.compare_tab, text="🌍 Comparaison Multi-Pays", padding=10)
        comp_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # --- Sélection métrique + pays ---
        sel_frame = ttk.Frame(comp_frame)
        sel_frame.pack(fill='x', pady=(0,10))

        ttk.Label(sel_frame, text="Métrique:").pack(side='left')
        self.comp_metric_var = tk.StringVar(value='total_cases')
        ttk.Combobox(
            sel_frame,
            textvariable=self.comp_metric_var,
            values=['total_cases', 'new_cases', 'predicted_cases'],
            state='readonly',
            width=18
        ).pack(side='left', padx=(5,20))

        ttk.Label(sel_frame, text="Pays à comparer:").pack(anchor='w')
        self.comparison_listbox = tk.Listbox(sel_frame, height=6, selectmode='extended')
        self.comparison_listbox.pack(fill='x', pady=5)
        for country in sorted(self.countries):
            self.comparison_listbox.insert(tk.END, country)

        self.selected_label = ttk.Label(sel_frame, text="Sélection: Aucun")
        self.selected_label.pack(anchor='w', pady=(2,5))
        self.comparison_listbox.bind('<<ListboxSelect>>', self._update_selected_label)

        btn_frame = ttk.Frame(sel_frame)
        btn_frame.pack(fill='x')
        ttk.Button(btn_frame, text="📊 Comparer", command=self.generate_comparison).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="🌐 Interactif", command=self.open_plotly_comparison, bootstyle="info").pack(side='left', padx=5)

        # --- Zone d’affichage ---
        self.comparison_chart_frame = ttk.Frame(comp_frame)
        self.comparison_chart_frame.pack(fill='both', expand=True, pady=(10,0))

    def _update_selected_label(self, event=None):
        """Met à jour l’étiquette des pays sélectionnés."""
        sel = self.comparison_listbox.curselection()
        if not sel:
            text = "Aucun"
        else:
            pays = [self.comparison_listbox.get(i) for i in sel]
            text = ", ".join(pays)
        self.selected_label.config(text=f"Sélection: {text}")

    def open_plotly_comparison(self):
        """Ouvre un graphique Plotly interactif de la comparaison."""
        base = self.country_var.get().strip()
        chosen = [self.comparison_listbox.get(i) for i in self.comparison_listbox.curselection()]
        if not base or not chosen:
            messagebox.showwarning("Comparaison", "Sélectionnez un pays origine et au moins un comparatif.")
            return

        metric = self.comp_metric_var.get()
        fig = make_subplots(rows=1, cols=1, specs=[[{'type':'scatter'}]])

        for country in [base] + chosen:
            if metric == 'predicted_cases' and country == base and self.current_predictions is not None:
                dfp = self.current_predictions.copy()
                dfp['date'] = pd.to_datetime(dfp['date'])
                fig.add_trace(go.Scatter(
                    x=dfp['date'], y=dfp['predicted_cases'],
                    mode='lines+markers', name=f"{country} (prédits)"
                ))
                continue

            df = self.get_country_df(country)
            if metric in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['date'], y=df[metric],
                    mode='lines+markers', name=country
                ))

        fig.update_layout(
            title=f"Comparaison interactive: {metric.replace('_',' ')}",
            xaxis_title="Date",
            yaxis_title=metric.replace('_',' ').capitalize(),
            template='plotly_white',
            hovermode='x unified'
        )

        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False)
        fig.write_html(tmp.name, include_plotlyjs='cdn')
        tmp.close()
        webbrowser.open(f'file://{tmp.name}')
        self.update_status("Comparaison Plotly ouverte")









































    def create_status_bar(self):
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill='x', side='bottom')
        
        self.status_bar = ttk.Label(status_frame, text="Prêt", relief='sunken', anchor='w')
        self.status_bar.pack(side='left', fill='x', expand=True)
        
        # Indicateur de progression
        self.progress_bar = ttk.Progressbar(status_frame, mode='indeterminate', length=100)
        self.progress_bar.pack(side='right', padx=5)

    def update_status(self, msg: str, show_progress: bool = False):
        """Met à jour la barre de statut avec horodatage"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.status_bar.config(text=f"{timestamp} - {msg}")
        
        if show_progress:
            self.progress_bar.start()
        else:
            self.progress_bar.stop()
            
        self.root.update_idletasks()
        self.log_message('info', msg)

    def test_api_connection(self):
        """Test de connexion API amélioré avec métriques"""
        def task():
            self.update_status("Test de connexion API...", True)
            url = self.api_url_var.get().rstrip('/') + '/health'
            
            start_time = time.time()
            try:
                response = requests.get(url, timeout=self.REQUEST_TIMEOUT)
                response_time = round((time.time() - start_time) * 1000, 2)
                
                if response.status_code == 200:
                    self.api_status_label.config(text="✅ API Connectée", bootstyle="success")
                    self.api_response_time_label.config(text=f"Temps: {response_time}ms")
                    self.update_status("API accessible et fonctionnelle")
                    self.log_message('info', f"API OK - Temps de réponse: {response_time}ms")
                else:
                    self.api_status_label.config(text=f"⚠️ Erreur {response.status_code}", bootstyle="warning")
                    self.api_response_time_label.config(text=f"Temps: {response_time}ms")
                    self.update_status(f"API répond mais erreur {response.status_code}")
                    self.log_message('warning', f"API erreur {response.status_code}")
                    
            except requests.exceptions.Timeout:
                self.api_status_label.config(text="⏰ Timeout", bootstyle="danger")
                self.api_response_time_label.config(text="Temps: >15s")
                self.update_status("Timeout de connexion API")
                self.log_message('error', "Timeout de connexion API")
                
            except requests.exceptions.ConnectionError:
                self.api_status_label.config(text="❌ Connexion impossible", bootstyle="danger")
                self.api_response_time_label.config(text="Temps: -")
                self.update_status("Impossible de se connecter à l'API")
                self.log_message('error', "Connexion API impossible")
                
            except Exception as e:
                self.api_status_label.config(text="❌ Erreur inconnue", bootstyle="danger")
                self.api_response_time_label.config(text="Temps: -")
                self.update_status(f"Erreur API: {str(e)}")
                self.log_message('error', f"Erreur API: {str(e)}")
                
        threading.Thread(target=task, daemon=True).start()

    def load_countries(self):
        """Chargement des pays avec gestion d'erreurs améliorée"""
        def task():
            self.update_status("Chargement de la liste des pays...", True)
            key = 'countries'
            
            # Vérifier le cache
            if key in self.cache and time.time() - self.cache[key][0] < self.cache_timeout:
                self.countries = self.cache[key][1]
                self.log_message('info', f"Pays chargés depuis le cache: {len(self.countries)}")
            else:
                try:
                    url = self.api_url_var.get().rstrip('/') + '/'
                    response = requests.get(url, timeout=self.REQUEST_TIMEOUT)
                    response.raise_for_status()
                    
                    data = response.json()
                    self.countries = data.get('countries', data if isinstance(data, list) else [])
                    self.cache[key] = (time.time(), self.countries)
                    self.log_message('info', f"Pays chargés depuis l'API: {len(self.countries)}")
                    
                except requests.exceptions.RequestException as e:
                    self.log_message('error', f"Erreur réseau lors du chargement des pays: {e}")
                    messagebox.showerror("Erreur Réseau", f"Impossible de charger les pays:\n{str(e)}")
                    return
                except json.JSONDecodeError as e:
                    self.log_message('error', f"Erreur JSON lors du chargement des pays: {e}")
                    messagebox.showerror("Erreur de Format", "Réponse API invalide")
                    return
                except Exception as e:
                    self.log_message('error', f"Erreur inconnue: {e}")
                    messagebox.showerror("Erreur", f"Erreur inattendue: {str(e)}")
                    return
            
            self.root.after(0, self._update_countries_display)
            self.update_status(f"{len(self.countries)} pays disponibles")
            self._update_cache_stats()
            
        threading.Thread(target=task, daemon=True).start()

    def refresh_comparison_listbox(self):
        """Met à jour la listbox des pays disponibles."""
        self.comparison_listbox.delete(0, tk.END)
        for country in sorted(self.countries):
            self.comparison_listbox.insert(tk.END, country)

    def _update_countries_display(self):
        """Met à jour l'affichage de la liste des pays"""
        if self.countries:
            self.countries_label.config(text=f"✅ {len(self.countries)} pays disponibles")
            sorted_countries = [''] + sorted(self.countries)
            self.country_combo['values'] = sorted_countries
            
            # Mise à jour de la liste de comparaison
            self.comparison_listbox.delete(0, tk.END)
            for country in sorted(self.countries):
                self.comparison_listbox.insert(tk.END, country)
                
            self.country_combo.set('')
            self.refresh_comparison_listbox()
        else:
            self.countries_label.config(text="❌ Aucun pays disponible")

    def load_country_data(self):
        """Chargement des données d'un pays avec validation"""
        country = self.country_var.get().strip()
        if not country:
            messagebox.showwarning("Attention", "Veuillez sélectionner un pays")
            return
            
        def task():
            self.update_status(f"Chargement des données pour {country}...", True)
            
            # Préparation des paramètres
            params = {}
            
            # Validation et ajout des dates
            start_date = self.start_date_var.get().strip()
            end_date = self.end_date_var.get().strip()
            
            if start_date and not start_date.startswith('Format:'):
                try:
                    datetime.strptime(start_date, '%Y-%m-%d')
                    params['start_date'] = start_date
                except ValueError:
                    self.log_message('warning', f"Format de date de début invalide: {start_date}")
                    
            if end_date and not end_date.startswith('Format:'):
                try:
                    datetime.strptime(end_date, '%Y-%m-%d')
                    params['end_date'] = end_date
                except ValueError:
                    self.log_message('warning', f"Format de date de fin invalide: {end_date}")
            
            # Validation et ajout de la limite
            try:
                limit = int(self.limit_var.get().strip())
                if limit > 0:
                    params['limit'] = limit
            except ValueError:
                self.log_message('warning', "Limite non valide, ignorée")
            
            # Vérification du cache
            cache_key = f"{country}_{str(sorted(params.items()))}"
            if cache_key in self.cache and time.time() - self.cache[cache_key][0] < self.cache_timeout:
                data = self.cache[cache_key][1]
                self.log_message('info', f"Données de {country} chargées depuis le cache")
            else:
                try:
                    url = self.api_url_var.get().rstrip('/') + f"/stats/{quote(country)}"
                    response = requests.get(url, params=params, timeout=self.REQUEST_TIMEOUT)
                    
                    if response.status_code == 404:
                        messagebox.showerror("Pays Non Trouvé", f"Le pays '{country}' n'existe pas dans l'API")
                        self.update_status(f"Pays {country} non trouvé")
                        return
                    elif response.status_code != 200:
                        messagebox.showerror("Erreur API", f"Erreur {response.status_code}: {response.reason}")
                        self.update_status(f"Erreur API {response.status_code}")
                        return
                    
                    data = response.json()
                    self.cache[cache_key] = (time.time(), data)
                    self.log_message('info', f"Données de {country} chargées depuis l'API")
                    
                except requests.exceptions.RequestException as e:
                    self.log_message('error', f"Erreur réseau: {e}")
                    messagebox.showerror("Erreur Réseau", f"Impossible de charger les données:\n{str(e)}")
                    return
                except json.JSONDecodeError:
                    self.log_message('error', "Réponse API invalide")
                    messagebox.showerror("Erreur", "Réponse API invalide")
                    return
            
            self.current_data = data
            self.root.after(0, self._update_data_display)
            self.update_status(f"Données de {country} chargées avec succès")
            self._update_cache_stats()
            
        threading.Thread(target=task, daemon=True).start()

    def _update_data_display(self):
        """Met à jour l'affichage des données avec calculs statistiques"""
        try:
            data = self.current_data.get('data', self.current_data if isinstance(self.current_data, list) else [])
            
            if not data:
                messagebox.showwarning("Attention", "Aucune donnée disponible pour ce pays")
                return
            
            # Création du DataFrame
            df = pd.DataFrame(data)
            
            # Conversion et tri des dates
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
            
            # Calculs statistiques
            df = self._calculate_statistics(df)
            self.current_df = df
            
            # Mise à jour des labels de résumé
            self._update_summary_labels(df)
            
            # Mise à jour du tableau
            self._populate_data_tree(df)
            
            # Mise à jour automatique des graphiques
            self.update_charts()
            
        except Exception as e:
            self.log_message('error', f"Erreur lors de la mise à jour des données: {e}")
            messagebox.showerror("Erreur", f"Erreur lors du traitement des données:\n{str(e)}")

    def _calculate_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les statistiques dérivées"""
        # Calcul du taux de mortalité
        if 'total_cases' in df.columns and 'total_deaths' in df.columns:
            df['mortality_rate'] = np.where(
                df['total_cases'] > 0,
                (df['total_deaths'] / df['total_cases']) * 100,
                0
            )
        
        # Calcul de la moyenne mobile sur 7 jours
        if 'new_cases' in df.columns:
            df['new_cases_ma7'] = df['new_cases'].rolling(window=7, min_periods=1).mean()
        
        if 'new_deaths' in df.columns:
            df['new_deaths_ma7'] = df['new_deaths'].rolling(window=7, min_periods=1).mean()
        
        return df

    def _update_summary_labels(self, df: pd.DataFrame):
        """Met à jour les labels de résumé statistique"""
        self.records_label.config(text=f"Enregistrements: {len(df):,}")
        
        if 'date' in df.columns and len(df) > 0:
            period_days = (df['date'].max() - df['date'].min()).days
            self.period_label.config(text=f"Période: {period_days} jours")
        
        if 'total_cases' in df.columns and len(df) > 0:
            latest_cases = int(df['total_cases'].iloc[-1])
            self.cases_label.config(text=f"Derniers cas: {latest_cases:,}")
        
        if 'total_deaths' in df.columns and len(df) > 0:
            latest_deaths = int(df['total_deaths'].iloc[-1])
            self.deaths_label.config(text=f"Derniers décès: {latest_deaths:,}")
        
        if 'new_cases' in df.columns and len(df) > 0:
            max_daily = int(df['new_cases'].max())
            avg_daily = int(df['new_cases'].mean())
            self.max_cases_label.config(text=f"Pic quotidien: {max_daily:,}")
            self.avg_cases_label.config(text=f"Moyenne/jour: {avg_daily:,}")

    def _populate_data_tree(self, df: pd.DataFrame):
        """Remplit le tableau de données"""
        # Vider le tableau
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        # Remplir avec les nouvelles données
        for _, row in df.iterrows():
            values = [
                row.get('date').strftime('%Y-%m-%d') if pd.notna(row.get('date')) else '',
                f"{int(row.get('total_cases', 0)):,}",
                f"{int(row.get('total_deaths', 0)):,}",
                f"{int(row.get('new_cases', 0)):,}",
                f"{int(row.get('new_deaths', 0)):,}",
                f"{row.get('mortality_rate', 0):.2f}" if 'mortality_rate' in row else '0.00'
            ]
            self.data_tree.insert('', tk.END, values=values)

    def filter_data(self, event=None):
        """Filtre les données du tableau selon la recherche"""
        if self.current_df is None:
            return
        
        search_term = self.search_var.get().lower()
        
        # Vider le tableau
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        # Filtrer et afficher
        for _, row in self.current_df.iterrows():
            date_str = row.get('date').strftime('%Y-%m-%d') if pd.notna(row.get('date')) else ''
            
            if search_term in date_str.lower():
                values = [
                    date_str,
                    f"{int(row.get('total_cases', 0)):,}",
                    f"{int(row.get('total_deaths', 0)):,}",
                    f"{int(row.get('new_cases', 0)):,}",
                    f"{int(row.get('new_deaths', 0)):,}",
                    f"{row.get('mortality_rate', 0):.2f}" if 'mortality_rate' in row else '0.00'
                ]
                self.data_tree.insert('', tk.END, values=values)

    def sort_treeview(self, col_index: int):
        """Trie le tableau par colonne"""
        if self.current_df is None:
            return
        
        # Colonnes correspondantes dans le DataFrame
        df_columns = ['date', 'total_cases', 'total_deaths', 'new_cases', 'new_deaths', 'mortality_rate']
        
        if col_index < len(df_columns):
            col_name = df_columns[col_index]
            if col_name in self.current_df.columns:
                # Tri du DataFrame
                self.current_df = self.current_df.sort_values(col_name, ascending=True)
                # Mise à jour de l'affichage
                self._populate_data_tree(self.current_df)

    def refresh_data_display(self):
        """Actualise l'affichage des données"""
        if self.current_df is not None:
            self._populate_data_tree(self.current_df)
            self.update_status("Affichage des données actualisé")

    def update_charts(self):
        """Génère les graphiques avec style personnalisé"""
        if self.current_df is None or 'date' not in self.current_df.columns:
            return
        
        # Vider le frame des graphiques
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        try:
            # Application du style
            plt.style.use(self.chart_style.get())
            
            # Configuration de la figure
            fig = Figure(figsize=(14, 10), dpi=100)
            fig.suptitle(f'COVID-19 - {self.country_var.get()}', fontsize=16, fontweight='bold')
            
            # Création des sous-graphiques
            gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
            
            df = self.current_df
            metrics = [
                ('total_cases', 'Cas Totaux Cumulés', self.colors['total_cases']),
                ('total_deaths', 'Décès Totaux Cumulés', self.colors['total_deaths']),
                ('new_cases', 'Nouveaux Cas Quotidiens', self.colors['new_cases']),
                ('new_deaths', 'Nouveaux Décès Quotidiens', self.colors['new_deaths'])
            ]
            
            for i, (col, title, color) in enumerate(metrics):
                if col not in df.columns:
                    continue
                
                ax = fig.add_subplot(gs[i//2, i%2])
                
                # Type de graphique
                if self.chart_type.get() == 'line':
                    ax.plot(df['date'], df[col], color=color, linewidth=2, marker='o', markersize=3)
                elif self.chart_type.get() == 'bar':
                    ax.bar(df['date'], df[col], color=color, alpha=0.7)
                elif self.chart_type.get() == 'scatter':
                    ax.scatter(df['date'], df[col], color=color, alpha=0.6, s=20)
                elif self.chart_type.get() == 'area':
                    ax.fill_between(df['date'], df[col], color=color, alpha=0.3)
                    ax.plot(df['date'], df[col], color=color, linewidth=2)
                
                # Ajout de la moyenne mobile si disponible
                ma_col = f"{col.replace('total_', 'new_')}_ma7"
                if ma_col in df.columns and 'new_' in col:
                    ax.plot(df['date'], df[ma_col], color='red', linewidth=2, 
                           linestyle='--', alpha=0.8, label='Moyenne mobile 7j')
                
                # Configuration de l'axe
                ax.set_title(title, fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
                
                # Échelle logarithmique si demandée
                if self.log_scale_var.get() and df[col].max() > 0:
                    ax.set_yscale('log')
                
                # Grille
                if self.show_grid_var.get():
                    ax.grid(True, alpha=0.3)
                
                # Légende
                if self.show_legend_var.get() and ma_col in df.columns and 'new_' in col:
                    ax.legend()
                
                # Formatage des nombres sur l'axe Y
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
            
            # Canvas pour Tkinter
            canvas = FigureCanvasTkAgg(fig, self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
            # Barre d'outils de navigation
            toolbar = ttk.Frame(self.chart_frame)
            toolbar.pack(fill='x', pady=(5,0))
            
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            nav_toolbar = NavigationToolbar2Tk(canvas, toolbar)
            nav_toolbar.update()
            
            self.log_message('info', 'Graphiques mis à jour avec succès')
            
        except Exception as e:
            self.log_message('error', f'Erreur lors de la génération des graphiques: {e}')
            messagebox.showerror("Erreur Graphique", f"Impossible de générer les graphiques:\n{str(e)}")

    def toggle_chart_zoom(self):
        """Active/désactive le zoom sur les graphiques"""
        # Cette fonctionnalité est gérée par la barre d'outils matplotlib
        messagebox.showinfo("Zoom", "Utilisez la barre d'outils en bas des graphiques pour zoomer/dézoomer")

    def open_plotly_chart(self):
        """Ouvre un graphique interactif Plotly"""
        if self.current_df is None:
            messagebox.showwarning("Attention", "Aucune donnée disponible")
            return
        
        try:
            df = self.current_df
            
            # Création de sous-graphiques
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Cas Totaux', 'Décès Totaux', 'Nouveaux Cas', 'Nouveaux Décès'),
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # Mapping des métriques
            metrics = [
                ('total_cases', 'Cas Totaux', self.colors['total_cases'], 1, 1),
                ('total_deaths', 'Décès Totaux', self.colors['total_deaths'], 1, 2),
                ('new_cases', 'Nouveaux Cas', self.colors['new_cases'], 2, 1),
                ('new_deaths', 'Nouveaux Décès', self.colors['new_deaths'], 2, 2)
            ]
            
            for col, name, color, row, col_pos in metrics:
                if col in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['date'],
                            y=df[col],
                            mode='lines+markers',
                            name=name,
                            line=dict(color=color, width=2),
                            marker=dict(size=4),
                            hovertemplate=f"<b>{name}</b><br>" +
                                        "Date: %{x}<br>" +
                                        "Valeur: %{y:,}<br>" +
                                        "<extra></extra>"
                        ),
                        row=row, col=col_pos
                    )
                    
                    # Ajout de la moyenne mobile si disponible
                    ma_col = f"{col.replace('total_', 'new_')}_ma7"
                    if ma_col in df.columns and 'new_' in col:
                        fig.add_trace(
                            go.Scatter(
                                x=df['date'],
                                y=df[ma_col],
                                mode='lines',
                                name=f'{name} (Moy. 7j)',
                                line=dict(color='red', width=2, dash='dash'),
                                hovertemplate=f"<b>{name} - Moyenne 7j</b><br>" +
                                            "Date: %{x}<br>" +
                                            "Valeur: %{y:,.1f}<br>" +
                                            "<extra></extra>"
                            ),
                            row=row, col=col_pos
                        )
            
            # Configuration du layout
            fig.update_layout(
                title=f"COVID-19 - {self.country_var.get()} (Graphique Interactif)",
                height=800,
                showlegend=True,
                hovermode='x unified',
                template='plotly_white'
            )
            
            # Mise à jour des axes
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Nombre de cas")
            
            # Sauvegarde temporaire et ouverture
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False)
            fig.write_html(temp_file.name, include_plotlyjs='cdn')
            temp_file.close()
            
            webbrowser.open(f'file://{temp_file.name}')
            self.update_status("Graphique Plotly ouvert dans le navigateur")
            self.log_message('info', 'Graphique Plotly généré avec succès')
            
        except Exception as e:
            self.log_message('error', f'Erreur lors de la génération du graphique Plotly: {e}')
            messagebox.showerror("Erreur Plotly", f"Impossible de générer le graphique interactif:\n{str(e)}")

    def generate_predictions(self):
        """Génère les prédictions ML avec graphique"""
        if not ML_AVAILABLE:
            messagebox.showwarning("ML Non Disponible", "Le module de machine learning n'est pas installé")
            return
        
        if self.current_df is None or len(self.current_df) < 10:
            messagebox.showwarning("Données Insuffisantes", 
                                 "Au moins 10 points de données sont nécessaires pour les prédictions")
            return
        
        def task():
            try:
                self.update_status("Génération des prédictions ML...", True)
                
                days = int(self.days_var.get())
                model_type = self.model_type_var.get()
                
                # Appel de la fonction de prédiction
                result = enhanced_predict_cases_df(
                    self.current_df, 
                    days_ahead=days, 
                    model_type=model_type
                )
                
                days_list, predictions, mae, confidence_low, confidence_high, r2, model_name = result
                
                self.root.after(0, lambda: self._update_predictions_display(
                    days_list, predictions, mae, confidence_low, confidence_high, r2, model_name
                ))
                
                self.update_status("Prédictions ML générées avec succès")
                self.log_message('info', f'Prédictions générées: {model_name}, R²={r2:.3f}')
                
            except Exception as e:
                self.log_message('error', f'Erreur lors de la génération des prédictions: {e}')
                self.root.after(0, lambda: messagebox.showerror(
                    "Erreur ML", f"Erreur lors des prédictions:\n{str(e)}"
                ))
                self.update_status("Erreur lors des prédictions")
        
        threading.Thread(target=task, daemon=True).start()









    def _update_predictions_display(self, days_list, predictions, mae, conf_low, conf_high, r2, model_name):
        """Met à jour l'affichage des prédictions"""
        try:
            # Mise à jour des informations du modèle
            self.model_info_label.config(
                text=f"Modèle: {model_name} | R²: {r2:.3f} | MAE: {mae:.0f}"
            )
            
            # Calcul de la précision
            accuracy = max(0, min(100, r2 * 100))
            details = f"Précision: {accuracy:.1f}% | Données d'entraînement: {len(self.current_df)} points"
            self.model_details_label.config(text=details)
            
            # Vider le tableau des prédictions
            for item in self.pred_tree.get_children():
                self.pred_tree.delete(item)
            
            # Remplir le tableau
            for i, (day, pred, low, high) in enumerate(zip(days_list, predictions, conf_low, conf_high)):
                confidence = max(0, min(100, 100 - (abs(high - low) / pred * 50)))

                # ✅ Convertir l'entier en date si nécessaire
                if not hasattr(day, "strftime"):
                    day = pd.to_datetime(day)

                values = [
                    day.strftime('%Y-%m-%d'),
                    f"{int(pred):,}",
                    f"{int(low):,}",
                    f"{int(high):,}",
                    f"{confidence:.1f}%"
                ]
                self.pred_tree.insert('', tk.END, values=values)
            
            # Sauvegarde des prédictions
            self.current_predictions = pd.DataFrame({
                'date': days_list,
                'predicted_cases': predictions.astype(int),
                'confidence_low': conf_low.astype(int),
                'confidence_high': conf_high.astype(int)
            })
            
            # Génération du graphique des prédictions
            self._create_predictions_chart(days_list, predictions, conf_low, conf_high)
            
            # Basculer vers l'onglet des prédictions
            self.nb.select(self.pred_tab)
            
        except Exception as e:
            self.log_message('error', f'Erreur lors de la mise à jour des prédictions: {e}')
            messagebox.showerror("Erreur", f"Erreur lors de l'affichage des prédictions:\n{str(e)}")

    def _create_predictions_chart(self, days_list, predictions, conf_low, conf_high):
        """Crée le graphique des prédictions"""
        try:
            # Vider le frame
            for widget in self.pred_chart_frame.winfo_children():
                widget.destroy()
            
            # Création de la figure
            fig = Figure(figsize=(12, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            # Données historiques
            if 'total_cases' in self.current_df.columns:
                ax.plot(self.current_df['date'], self.current_df['total_cases'], 
                       color=self.colors['total_cases'], linewidth=2, label='Données historiques', marker='o', markersize=3)
            
            # Prédictions
            ax.plot(days_list, predictions, color=self.colors['predictions'], 
                   linewidth=3, label='Prédictions', marker='s', markersize=4)
            
            # Intervalles de confiance
            if self.show_confidence_var.get():
                ax.fill_between(days_list, conf_low, conf_high, 
                               color=self.colors['predictions'], alpha=0.2, label='Intervalle de confiance')
                ax.plot(days_list, conf_low, color=self.colors['predictions'], 
                       linewidth=1, linestyle='--', alpha=0.7)
                ax.plot(days_list, conf_high, color=self.colors['predictions'], 
                       linewidth=1, linestyle='--', alpha=0.7)
            
            # Configuration
            ax.set_title('Prédictions COVID-19', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Nombre de cas')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            # Formatage des nombres
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
            
            fig.tight_layout()
            
            # Canvas
            canvas = FigureCanvasTkAgg(fig, self.pred_chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except Exception as e:
            self.log_message('error', f'Erreur lors de la création du graphique de prédictions: {e}')

    def remove_comparison_country(self):
        """Retire un pays sélectionné de la comparaison"""
        selection = self.comparison_listbox.curselection()
        if selection:
            for i in reversed(selection):
                country_str = self.comparison_listbox.get(i)
                country_name = country_str.split(" ")[0]
                if country_name in self.comparison_countries:
                    self.comparison_countries.remove(country_name)
                self.comparison_listbox.delete(i)
            self.update_status("Pays retiré de la comparaison")

    def export_csv(self):
        if self.current_df is not None:
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if path:
                self.current_df.to_csv(path, index=False)
                self.update_status("Export CSV réussi")

    def export_json(self):
        if self.current_df is not None:
            path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
            if path:
                self.current_df.to_json(path, orient="records", date_format="iso")
                self.update_status("Export JSON réussi")

    def export_excel(self):
        if self.current_df is not None:
            path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
            if path:
                self.current_df.to_excel(path, index=False)
                self.update_status("Export Excel réussi")

    def export_chart(self):
        if self.current_df is not None:
            path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("Image PNG", "*.png")])
            if path:
                fig = plt.figure(figsize=(10, 6))
                plt.plot(self.current_df['date'], self.current_df['total_cases'], label="Cas Totaux", color='blue')
                plt.title("Courbe des cas totaux")
                plt.xlabel("Date")
                plt.ylabel("Nombre de cas")
                plt.grid(True)
                plt.savefig(path)
                self.update_status("Export graphique réussi")

    def export_predictions(self):
        if self.current_predictions is not None:
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if path:
                self.current_predictions.to_csv(path, index=False)
                self.update_status("Export des prédictions réussi")

    def show_logs(self):
        log_win = tk.Toplevel(self.root)
        log_win.title("Logs de l'application")
        log_win.geometry("800x400")
        text = scrolledtext.ScrolledText(log_win, wrap=tk.WORD)
        text.pack(expand=True, fill='both')
        text.insert(tk.END, "\n".join(self.logs))
        text.configure(state='disabled')

    def clear_cache(self):
        self.cache.clear()
        self.update_status("Cache vidé")
        self._update_cache_stats()

    def _update_cache_stats(self):
        self.cache_stats_label.config(text=f"Cache: {len(self.cache)} entrées")

    def on_closing(self):
        """Confirmation avant de quitter l'application"""
        if messagebox.askokcancel("Quitter", "Voulez-vous vraiment quitter le dashboard ?"):
            self.root.destroy()


def main():
    # Choisissez un thème dark : "darkly", "cyborg", "vapor", "solar"
    root = tb.Window(themename="solar")
    app = COVID19Dashboard(root)
    root.mainloop()

if __name__ == "__main__":
    main()