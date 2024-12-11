import happybase
import pandas as pd

"""
Ce programme : 
    - Connecte à la base Hbase
    - Scan toute la table avec un filter sur la ville de Nantes 
    - Calculer la meilleure commande sur l'année 2020 
    - Exporter le résultat sur un fichier CSV 
"""


# Fonction, qui à partir d'une entrée dans hbase, on peut extraire les couple (Row_Qualifier, Value)
def get_hbase_data(datas):
    for key, value in scanner:
        decoded_data = {k.decode('utf-8'): v.decode('utf-8') for k, v in value.items()}
        yield {'key': key.decode('utf-8'), **decoded_data}


# Connection à la base
connection = happybase.Connection('node175998-env-1839015-etudiant27.sh1.hidora.com', 11669)  # 9090
connection.open()

# Selectionner la table sur laquelle chercher les données
table_name = b'datafromagerie'
table = connection.table(table_name)

# Charger la table avec un filter sur la ville de NANTES
scanner = table.scan(filter="SingleColumnValueFilter('cf', 'villecli',=,'binary:NANTES')")

# Convertir le résultat chargé de la table en un DataFrame Pandas
data = list(get_hbase_data(scanner))
df = pd.DataFrame(data)

# Caster le type de la colonne (Date Commande) en type DataTime
df['cf:datcde'] = pd.to_datetime(df['cf:datcde'])
# Considérer les quantités égales à NULL à 1
df['cf:qte'] = [int(qte) if qte != "NULL" else 1 for qte in df['cf:qte']]
# Faire une premier Filtre sur l'année 2020
filtered_df = df[df['cf:datcde'].dt.year == 2020]

# Faire un premier regroupement sur le Code-Commande et Timbre-Commande avec une aggrégation SUM sur Quantité
# pour en garder qu'une seule valeur Timbre-Commande par Commande
groupedby_data = filtered_df.groupby(['cf:codcde', 'cf:villecli', 'cf:timbrecde']).agg({'cf:qte': 'sum'}).reset_index()
# Faire un tri décroissant sur la Quantité et TimbreCode pour en choisir la meilleure commande
sorted_data = groupedby_data.sort_values(by=['cf:qte', 'cf:timbrecde'], ascending=[False, False]).head(1)

# Renommer les colonnes pour l'export
sorted_data = sorted_data.rename(columns={'cf:codcde': 'Code commande',
                                          'cf:villecli': 'Ville client',
                                          'cf:timbrecde': 'Timbre commande',
                                          'cf:qte': 'Quantite'})

# Exporter le résultat en fichier Csv
sorted_data.to_csv('resultats/lot3/meilleur_commande_nantes_2020.csv', index=False)

# Fermer la connexion
connection.close()
