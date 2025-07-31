=== Projet COVID-19 avec Cassandra et Streamlit ===

📌 Étapes à suivre :

1. Créer le keyspace et la table dans cqlsh :
---------------------------------------------
CREATE KEYSPACE covid
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

USE covid;

CREATE TABLE IF NOT EXISTS stats (
    country TEXT,
    date DATE,
    total_cases INT,
    total_deaths INT,
    PRIMARY KEY ((country), date)
);


2. Installer les dépendances :
-----------------------------
pip install -r requirements.txt

3. Insérer les données dans Cassandra :
--------------------------------------
python insert_data.py

4. Lancer le dashboard Streamlit :
----------------------------------
streamlit run app.py
