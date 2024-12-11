import happybase
import pandas as pd
import matplotlib.pyplot as plt

"""
Ce programme : 
    - Connecte à la base Hbase
    - Scan toute la table avec un filter sur les années 2010 et 2015
    - Extraire le total de données pour chacune des années entre 2010 et 2015
    - Exporter le résulat sous forme d'un 'Graphe Pie' 
    - Enregistrer le visule dans un fichier PDF   
"""

# Définir les deux dates limites pour les filtres
start_date = 2010
end_date = 2015


# Fonction, qui à partir d'une entrée dans hbase, on peut extraire les couple (Row_Qualifier, Value)
def get_hbase_data(datas):
    for key, value in datas:
        decoded_data = {k.decode('utf-8'): v.decode('utf-8') for k, v in value.items()}
        yield {'key': key.decode('utf-8'), **decoded_data}


# Connection en base
connection = happybase.Connection('node175998-env-1839015-etudiant27.sh1.hidora.com', 11669)  # 9090
connection.open()

# Connection à la tabe
table_name = b'datafromagerie'
table = connection.table(table_name)

# Scan de la table avec un filter sur les deux années 2010 <= Date Commande <= 2015
scanner = table.scan(filter=f"SingleColumnValueFilter('cf', 'datcde', >=, 'binary:{start_date}', true, false) AND SingleColumnValueFilter('cf', 'datcde', <=, 'binary:{end_date + 1}', true, false)")

# Transformer les données récupérées de la base en DataFrame PANDAS
data = list(get_hbase_data(scanner))
df = pd.DataFrame(data)

# Créer une colonne yearcde à partir de datcde pour faire les filtres
df['cf:yearcde'] = pd.to_datetime(df['cf:datcde']).dt.year

# Grouper par l'année de commande et code commande pour faire un DISTINCT(codcde)
grouped_df = df.groupby(['cf:yearcde', 'cf:codcde']).size()
# Calculer le nombre de commandes sur chaque année
grouped_df = grouped_df.groupby('cf:yearcde').size().reset_index(name='total')

connection.close()

# Export des données en barplot dans un Fichier PDF
# Création du barplot
plt.bar(grouped_df['cf:yearcde'], grouped_df['cf:total'])

# Ajout de titres et d'étiquettes
plt.title('Nombre de commandes de l\'année 2010 à 2015')
plt.xlabel('Années')
plt.ylabel('Total commandes')

# Export du barplot
plt.savefig("resultats/lot3/BarPlot_total_commandes_2010_2015.pdf")
