# backend.py
import logging
import re
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from cassandra.cluster import Cluster
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseConnection:
    def __init__(self, hosts: List[str] = ['127.0.0.1'], keyspace: str = 'covid'):
        self.session = None
        self.cluster = None
        self.hosts = hosts
        self.keyspace = keyspace
        self.connect()

    def connect(self) -> None:
        try:
            self.cluster = Cluster(self.hosts)
            self.session = self.cluster.connect()
            self.session.set_keyspace(self.keyspace)
            logger.info("Connexion à Cassandra établie")
        except Exception as e:
            logger.error(f"Erreur de connexion à Cassandra: {e}")
            raise

    def get_session(self):
        if not self.session:
            self.connect()
        return self.session

    def close(self) -> None:
        if self.cluster:
            self.cluster.shutdown()


def validate_country_name(country: str) -> Optional[str]:
    if not country:
        return None
    cleaned = re.sub(r'[^\w\s\-\.]', '', country).strip()
    if len(cleaned) < 2 or len(cleaned) > 100:
        return None
    return cleaned


def validate_date_range(
    start_date: Optional[str] = None, end_date: Optional[str] = None
) -> (Optional[date], Optional[date], Optional[str]):
    try:
        sd = datetime.strptime(start_date, '%Y-%m-%d').date() if start_date else None
        ed = datetime.strptime(end_date, '%Y-%m-%d').date() if end_date else None
        if sd and ed and sd > ed:
            return None, None, "Date de début postérieure à la date de fin"
        return sd, ed, None
    except ValueError:
        return None, None, "Format de date invalide. Utilisez YYYY-MM-DD"


# Initialize FastAPI app
app = FastAPI(
    title="COVID Stats API",
    description="API pour récupérer des statistiques COVID depuis Cassandra",
    version="1.0.0"
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection instance
db = DatabaseConnection()


@app.get("/", response_model=Dict[str, Any])
async def get_countries():
    try:
        session = db.get_session()
        rows = session.execute("SELECT DISTINCT country FROM stats")
        countries = sorted([row.country for row in rows if row.country])
        logger.info(f"Récupération de {len(countries)} pays")
        return {"countries": countries, "total": len(countries)}
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des pays: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des pays")


@app.get("/stats/{country}", response_model=Dict[str, Any])
async def get_stats(
    country: str,
    start_date: Optional[str] = Query(None, alias="start_date"),
    end_date: Optional[str] = Query(None, alias="end_date"),
    limit: Optional[int] = Query(None)
):
    # Validate country
    clean_country = validate_country_name(country)
    if not clean_country:
        raise HTTPException(status_code=400, detail="Nom de pays invalide")

    # Validate dates
    sd, ed, date_error = validate_date_range(start_date, end_date)
    if date_error:
        raise HTTPException(status_code=400, detail=date_error)

    # Build and execute query
    try:
        session = db.get_session()
        query = "SELECT date, total_cases, total_deaths FROM stats WHERE country=%s"
        params: List = [clean_country]

        if sd and ed:
            query += " AND date >= %s AND date <= %s"
            params.extend([sd, ed])

        query += " ORDER BY date ASC"

        if limit and limit > 0:
            query += " LIMIT %s"
            params.append(min(limit, 10000))

        rows = session.execute(query, params)
        data = [
            {
                "date": str(row.date),
                "total_cases": row.total_cases or 0,
                "total_deaths": row.total_deaths or 0
            }
            for row in rows
        ]

        if not data:
            raise HTTPException(status_code=404, detail=f"Aucune donnée trouvée pour {clean_country}")

        logger.info(f"Récupération de {len(data)} enregistrements pour {clean_country}")
        return {
            "country": clean_country,
            "data": data,
            "total_records": len(data),
            "date_range": {"start": data[0]["date"], "end": data[-1]["date"]}
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des stats pour {country}: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des données")


@app.get("/stats/{country}/summary", response_model=Dict[str, Any])
async def get_country_summary(country: str):
    clean_country = validate_country_name(country)
    if not clean_country:
        raise HTTPException(status_code=400, detail="Nom de pays invalide")

    try:
        session = db.get_session()
        summary_query = """
            SELECT MAX(total_cases) as max_cases,
                   MAX(total_deaths) as max_deaths,
                   MIN(date) as first_date,
                   MAX(date) as last_date
            FROM stats WHERE country=%s
        """
        result = session.execute(summary_query, [clean_country]).one()
        if not result or result.max_cases is None:
            raise HTTPException(status_code=404, detail=f"Aucune donnée trouvée pour {clean_country}")

        recent_query = (
            "SELECT date, total_cases, total_deaths FROM stats "
            "WHERE country=%s ORDER BY date DESC LIMIT 30"
        )
        recent_rows = session.execute(recent_query, [clean_country])
        recent_data = [
            {"date": str(r.date), "total_cases": r.total_cases or 0, "total_deaths": r.total_deaths or 0}
            for r in recent_rows
        ]

        return {
            "country": clean_country,
            "max_cases": result.max_cases or 0,
            "max_deaths": result.max_deaths or 0,
            "first_date": str(result.first_date) if result.first_date else None,
            "last_date": str(result.last_date) if result.last_date else None,
            "recent_data": recent_data[:10]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du résumé pour {country}: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération du résumé")


@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    try:
        session = db.get_session()
        session.execute("SELECT COUNT(*) FROM stats LIMIT 1")
        return {"status": "healthy", "timestamp": datetime.now().isoformat(), "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"status": "unhealthy", "timestamp": datetime.now().isoformat(), "database": "disconnected", "error": str(e)}
        )

# Shutdown event to close DB connection cleanly
@app.on_event("shutdown")
async def shutdown_event():
    db.close()

# To run with: uvicorn B_App:app --host 127.0.0.1 --port 8000
