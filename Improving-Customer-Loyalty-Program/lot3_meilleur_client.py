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
    for key, data in datas:
        decoded_data = {k.decode('utf-8'): v.decode('utf-8') for k, v in data.items()}
        yield {'key': key.decode('utf-8'), **decoded_data}


# Connection à la base
connection = happybase.Connection('node175998-env-1839015-etudiant27.sh1.hidora.com', 11669)  # 9090
connection.open()

# Selectionner la table sur laquelle chercher les données
table_name = b'datafromagerie'
table = connection.table(table_name)

# List de colonnes à charger de la table
row_to_fetch = [b'cf:codcli', b'cf:prenomcli', b'cf:nomcli', b'cf:qte', b'cf:timbrecde', b'cf:codcde']

# Charger toute la table
scanner = table.scan(columns=row_to_fetch)

# Convertir le résultat chargé de la table en un DataFrame Pandas
data = list(get_hbase_data(scanner))
df = pd.DataFrame(data)

# Considérer les quantités égales à NULL comme égale à 1
df['cf:qte'] = [int(qte) if qte != "NULL" else 1 for qte in df['cf:qte']]

# Considérer les timbres-commande égales à NULL comme égale à 0
df['cf:timbrecde'] = [float(timbre_code) if timbre_code != "NULL" else 0 for timbre_code in df['cf:timbrecde']]

# Faire un group-by sur codcli et pas sur nom et prénom
# Faire un distinct sur le codcde, pour enlever les doublons commande
grouped_df = df.groupby(['cf:codcli', 'cf:prenomcli', 'cf:nomcli', 'cf:codcde', 'cf:timbrecde']).agg(
    {'cf:qte': 'sum'}).reset_index()

# faire une groupby sur codecli pour faire une somme sur les quantités commandées
# la somme des timbre commande, et le nombre de commande
grouped_df = grouped_df.groupby(['cf:codcli', 'cf:prenomcli', 'cf:nomcli']).agg(
    {'cf:timbrecde': 'sum', 'cf:qte': 'sum', 'cf:codcde': 'size'}).reset_index()
# Faire le sort pour en garder le meilleur
sorted_df = grouped_df.sort_values(by='cf:timbrecde', ascending=False).head(1)

# Changer les noms de colonnes pour l'export
sorted_df = sorted_df.rename(columns={'cf:codcli': 'code client',
                                      'cf:prenomcli': 'prenom client',
                                      'cf:nomcli': 'nom client',
                                      'cf:qte': 'quantite',
                                      'cf:timbrecde': 'timbre code',
                                      'cf:codcde': 'nombre de commande'})

# Exporter en Excel
sorted_df.to_excel('resultats/lot3/meilleur_client_timbre_code.xlsx', index=False)

# Fermer la connection
connection.close()
