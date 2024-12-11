import sys
from io import StringIO
import pandas as pd

# Lire l'entrée du stdin (cat fichier.csv)
data = sys.stdin.buffer.read()

# Transformer data en un DataFrame Pandas
df = pd.read_csv(StringIO(data.decode('utf-8')), sep=',', header=None,
                 names=['code_commande', 'ville_client', 'timbre_commande', 'quantite'], na_values=['NULL'])

# Grouper par code commande, pour effacer les doublons de commande et une aggrégation sur la quantité pour calculer la quantité total de chaque commmande
groupedby_data = df.groupby(['code_commande', 'ville_client', 'timbre_commande']).agg({'quantite': 'sum'}).reset_index()
# Faire un tri sur la quantité et timbre commande pour avoir un classement décroissant des commandes
sorted_data = groupedby_data.sort_values(by=['quantite', 'timbre_commande'], ascending=[False, False])

# Faire une limit à 100 et exporter les résultats dans un fichier xlsx
sorted_data.head(100).to_excel('resultats/lot1/projet_hadoop_bigdata_lot1.xlsx', index=True)
