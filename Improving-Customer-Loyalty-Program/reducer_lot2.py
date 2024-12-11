import sys
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt

# Lire l'entrée du stdin (cat fichier.csv)
data = sys.stdin.buffer.read()

# Transformer data en un DataFrame Pandas
df = pd.read_csv(StringIO(data.decode('utf-8')), sep=',', header=None,
                 names=['code_commande', 'ville_client', 'timbre_commande', 'quantite', 'libobj'], na_values=['NULL'])


# Grouper par code commande, pour effacer les doublons de commande
groupedby_data = df.groupby(['code_commande', 'ville_client', 'timbre_commande']).agg(
    {'quantite': ['sum', 'mean']}).reset_index()

# Faire un tri sur la quantité et timbre commande pour avoir un classement décroissant des commandes
sorted_data = groupedby_data.sort_values(by=[('quantite', 'sum'), 'timbre_commande'], ascending=[False, False])
# Garder que les 100 meilleures commandes
sample_data = sorted_data.head(100)
# Extraire un échantillon aléatoire qui représente 5%
random_data = sample_data.sample(frac=0.05)

# Exporter dans un fichier Xlsx
random_data.to_excel('resultats/lot2projet_hadoop_bigdata_lot2.xlsx', index=True)


# Données nécessaires pour le graphe en PIE
labels = random_data['ville_client']
sizes = random_data[('quantite', 'sum')]

# Tracer le diagramme circulaire
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Distribution aléatoire des 5 meilleurs commandes')
plt.savefig('resultats/lot2/Pie_Diagram_Lot2.pdf')
