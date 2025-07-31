import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
from urllib.parse import quote
import time

# Add parent directory to path for ml_analysis import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

try:
    from ml_analysis import predict_cases_df
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Configuration
API_URL = "http://127.0.0.1:5000"
REQUEST_TIMEOUT = 10

# Page configuration
st.set_page_config(
    page_title="COVID-19 Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e1e5e9;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🦠 Dashboard COVID-19 - Analyse et Prédictions")
st.markdown("---")

# Helper Functions
def test_api_connection():
    """Test if the API is accessible"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            return True, "OK", response.json()
        else:
            # Fallback to root endpoint
            response = requests.get(f"{API_URL}/", timeout=REQUEST_TIMEOUT)
            return response.status_code == 200, response.status_code, None
    except requests.exceptions.ConnectionError:
        return False, "Connection Error", None
    except requests.exceptions.Timeout:
        return False, "Timeout", None
    except Exception as e:
        return False, str(e), None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_countries():
    """Fetch list of available countries from API"""
    try:
        response = requests.get(f"{API_URL}/", timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict) and 'countries' in data:
                return data['countries'], None
            elif isinstance(data, list):
                return data, None
            else:
                return [], f"Format de réponse inattendu: {type(data)}"
        else:
            return [], f"Erreur API: {response.status_code} - {response.text[:200]}"
    except requests.exceptions.ConnectionError:
        return [], "Impossible de se connecter à l'API"
    except requests.exceptions.Timeout:
        return [], "Timeout lors de la connexion à l'API"
    except Exception as e:
        return [], f"Erreur: {str(e)}"

def fetch_country_stats(country, start_date=None, end_date=None, limit=None):
    """Fetch statistics for a specific country"""
    try:
        encoded_country = quote(country.strip())
        url = f"{API_URL}/stats/{encoded_country}"
        
        # Add query parameters
        params = {}
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        if limit:
            params['limit'] = limit
        
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        
        if response.status_code == 200:
            return response.json(), None
        elif response.status_code == 404:
            return None, f"Pays '{country}' non trouvé dans la base de données"
        else:
            return None, f"Erreur API: {response.status_code} - {response.text[:200]}"
            
    except requests.exceptions.ConnectionError:
        return None, "Impossible de se connecter à l'API"
    except requests.exceptions.Timeout:
        return None, "Timeout lors de la requête"
    except Exception as e:
        return None, f"Erreur: {str(e)}"

def create_plotly_chart(df, title, y_columns):
    """Create an interactive Plotly chart"""
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, col in enumerate(y_columns):
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df[col],
                mode='lines+markers',
                name=col.replace('_', ' ').title(),
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=4),
                hovertemplate=f'<b>{col}</b><br>Date: %{{x}}<br>Valeur: %{{y:,}}<extra></extra>'
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Nombre',
        hovermode='x unified',
        height=500,
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

# Sidebar
st.sidebar.header("🛠️ Configuration")

# API Status Check
with st.sidebar:
    st.subheader("🔧 État de l'API")
    api_ok, status, health_data = test_api_connection()
    
    if api_ok:
        st.success("✅ API accessible")
        if health_data:
            st.json(health_data)
    else:
        st.error(f"❌ API non accessible: {status}")
        st.info("💡 Vérifiez que votre serveur Flask est démarré sur http://127.0.0.1:5000")
        if not api_ok:
            st.stop()

# Main Content
if api_ok:
    # Fetch countries
    st.header("🌍 Sélection du pays")
    
    with st.spinner("Chargement de la liste des pays..."):
        countries, error = fetch_countries()
    
    if error:
        st.error(f"❌ {error}")
        st.stop()
    
    if not countries:
        st.error("Aucun pays disponible. Vérifiez votre API.")
        st.stop()
    
    # Display countries info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.success(f"✅ {len(countries)} pays disponibles")
    with col2:
        show_countries = st.checkbox("Voir la liste")
    
    if show_countries:
        st.write("**Pays disponibles:**")
        countries_df = pd.DataFrame(countries, columns=['Pays'])
        st.dataframe(countries_df, use_container_width=True)
    
    # Country selection
    selected_country = st.selectbox(
        "Sélectionnez un pays", 
        [""] + sorted(countries),
        index=0
    )
    
    # Date filters
    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("Date de début (optionnel)", value=None)
    with col2:
        end_date = st.date_input("Date de fin (optionnel)", value=None)
    with col3:
        limit = st.number_input("Limite d'enregistrements", min_value=0, value=0, help="0 = pas de limite")
    
    # Process selected country
    if selected_country:
        st.markdown("---")
        st.header(f"📈 Statistiques COVID pour {selected_country}")
        
        # Prepare parameters
        start_str = start_date.strftime('%Y-%m-%d') if start_date else None
        end_str = end_date.strftime('%Y-%m-%d') if end_date else None
        limit_val = limit if limit > 0 else None
        
        with st.spinner("Récupération des données..."):
            data, error = fetch_country_stats(selected_country, start_str, end_str, limit_val)
        
        if error:
            st.error(f"❌ {error}")
            
            # Suggestions
            with st.expander("💡 Suggestions de dépannage"):
                st.info("• Vérifiez l'orthographe du nom du pays")
                st.info("• Essayez un autre pays de la liste")
                st.info("• Vérifiez que votre base de données contient des données pour ce pays")
                st.info("• Consultez les logs de votre API Flask")
                
        elif data:
            try:
                # Extract data from API response
                if isinstance(data, dict) and 'data' in data:
                    records = data['data']
                    country_info = {
                        'country': data.get('country', selected_country),
                        'total_records': data.get('total_records', len(records)),
                        'date_range': data.get('date_range', {})
                    }
                elif isinstance(data, list):
                    records = data
                    country_info = {'country': selected_country, 'total_records': len(records)}
                else:
                    st.error("Format de données non reconnu")
                    st.json(data)
                    st.stop()
                
                if not records:
                    st.warning("⚠️ Aucune donnée disponible pour ce pays")
                    st.stop()
                
                # Create DataFrame
                df = pd.DataFrame(records)
                
                # Data processing
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date').reset_index(drop=True)
                
                # Display summary metrics
                st.subheader("📊 Résumé")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🗓️ Enregistrements", f"{len(df):,}")
                
                if 'total_cases' in df.columns and len(df) > 0:
                    latest_cases = df['total_cases'].iloc[-1]
                    with col2:
                        st.metric("📈 Derniers cas totaux", f"{latest_cases:,}")
                
                if 'total_deaths' in df.columns and len(df) > 0:
                    latest_deaths = df['total_deaths'].iloc[-1]
                    with col3:
                        st.metric("💀 Derniers décès totaux", f"{latest_deaths:,}")
                
                if 'date' in df.columns and len(df) > 1:
                    date_range = (df['date'].max() - df['date'].min()).days
                    with col4:
                        st.metric("📅 Période (jours)", f"{date_range:,}")
                
                # Date range info
                if country_info.get('date_range'):
                    date_info = country_info['date_range']
                    st.info(f"📅 **Période des données:** {date_info.get('start')} → {date_info.get('end')}")
                
                # Data preview
                with st.expander("🔍 Aperçu des données"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Premières lignes:**")
                        st.dataframe(df.head())
                    with col2:
                        st.write("**Dernières lignes:**")
                        st.dataframe(df.tail())
                    
                    st.write("**Informations sur les colonnes:**")
                    info_df = pd.DataFrame({
                        'Colonne': df.columns,
                        'Type': [str(dtype) for dtype in df.dtypes],
                        'Non-null': [df[col].notna().sum() for col in df.columns],
                        'Null': [df[col].isna().sum() for col in df.columns]
                    })
                    st.dataframe(info_df)
                
                # Charts
                if 'date' in df.columns:
                    numeric_columns = [col for col in df.columns if col != 'date' and pd.api.types.is_numeric_dtype(df[col])]
                    
                    if numeric_columns:
                        st.subheader("📈 Visualisations")
                        
                        # Main chart with all data
                        chart_columns = [col for col in ['total_cases', 'total_deaths'] if col in df.columns]
                        if chart_columns:
                            fig = create_plotly_chart(df, f"Évolution COVID-19 - {selected_country}", chart_columns)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Individual charts
                            if len(chart_columns) >= 2:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if 'total_cases' in df.columns:
                                        fig_cases = create_plotly_chart(df, "Cas totaux", ['total_cases'])
                                        st.plotly_chart(fig_cases, use_container_width=True)
                                
                                with col2:
                                    if 'total_deaths' in df.columns:
                                        fig_deaths = create_plotly_chart(df, "Décès totaux", ['total_deaths'])
                                        st.plotly_chart(fig_deaths, use_container_width=True)
                        
                        # Additional numeric columns
                        other_columns = [col for col in numeric_columns if col not in ['total_cases', 'total_deaths']]
                        if other_columns:
                            selected_cols = st.multiselect(
                                "Sélectionnez d'autres colonnes à visualiser:",
                                other_columns,
                                default=other_columns[:2]
                            )
                            if selected_cols:
                                fig_other = create_plotly_chart(df, "Autres métriques", selected_cols)
                                st.plotly_chart(fig_other, use_container_width=True)
                
                # Predictions
                if ML_AVAILABLE and len(df) > 10:  # Need minimum data for predictions
                    st.subheader("🔮 Prédictions ML")
                    
                    days_ahead = st.slider("Nombre de jours à prédire", 1, 30, 7)
                    
                    if st.button("🚀 Générer les prédictions"):
                        try:
                            with st.spinner("Calcul des prédictions..."):
                                days, preds = predict_cases_df(df, days_ahead=days_ahead)
                                
                            pred_df = pd.DataFrame({
                                'Jour': days,
                                'Cas prédits': preds.astype(int)
                            })
                            
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                st.write("**Prédictions:**")
                                st.dataframe(pred_df)
                            
                            with col2:
                                # Prediction chart
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.plot(pred_df['Jour'], pred_df['Cas prédits'], 
                                       marker='o', linewidth=2, markersize=6, color='red', alpha=0.8)
                                ax.set_xlabel('Jour (depuis début des données)')
                                ax.set_ylabel('Cas confirmés prédits')
                                ax.set_title(f'Prédictions COVID-19 - {selected_country}')
                                ax.grid(True, alpha=0.3)
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                        except Exception as e:
                            st.error(f"❌ Erreur de prédiction: {e}")
                            with st.expander("Détails de l'erreur"):
                                import traceback
                                st.code(traceback.format_exc())
                elif ML_AVAILABLE:
                    st.info("🔮 **Prédictions:** Pas assez de données (minimum 10 points requis)")
                else:
                    st.warning("⚠️ **Module de prédiction non disponible**")
                
                # Export data
                st.subheader("💾 Export des données")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Télécharger CSV",
                        data=csv,
                        file_name=f"covid_data_{selected_country}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    json_data = df.to_json(orient='records', indent=2)
                    st.download_button(
                        label="📥 Télécharger JSON",
                        data=json_data,
                        file_name=f"covid_data_{selected_country}_{pd.Timestamp.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
                
            except Exception as e:
                st.error(f"❌ Erreur lors du traitement des données: {e}")
                with st.expander("Données brutes et erreur détaillée"):
                    st.write("**Données reçues:**")
                    st.json(data)
                    st.write("**Erreur détaillée:**")
                    import traceback
                    st.code(traceback.format_exc())

# Sidebar additional info
with st.sidebar:
    st.markdown("---")
    st.subheader("ℹ️ Informations")
    st.info(f"**URL API:** {API_URL}")
    st.info(f"**ML disponible:** {'✅' if ML_AVAILABLE else '❌'}")
    
    if countries:
        st.info(f"**Pays disponibles:** {len(countries)}")
    
    st.markdown("---")
    st.subheader("🔧 Outils de debug")
    
    if st.button("🔄 Vider le cache"):
        st.cache_data.clear()
        st.success("Cache vidé!")
        st.rerun()
    
    if st.button("🔍 Test API Health"):
        with st.spinner("Test en cours..."):
            ok, status, health = test_api_connection()
            if ok and health:
                st.json(health)
            else:
                st.error(f"Erreur: {status}")