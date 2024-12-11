import csv
from datetime import datetime
import happybase

"""
Ce programme : 
   - Se connecte à la base de données Hbase sur la machine distante 
   - Crée une table 'datafromagerie' en prendant le soin de la supprimer si elle était déjà présente 
   - Charge les données présentes dans un fichier CSV
   - Transforme les lignes du fichier csv en une entreé en hbase, Row_Key  
"""

# Etablir la connexion a HBASE
connection = happybase.Connection('node175998-env-1839015-etudiant27.sh1.hidora.com', 11669)  # 9090
connection.open()

# Fichier à charger
csv_file_path = "datas/dataw_fro03.csv"

# Information concernant la table à créer
table_name = b'datafromagerie'
families = {
    'cf': dict()
}

# Un compteur, utilisé pour le Row_Key de HBASE
row_key_counter = 1

"""
    Fonction, qui à partir de la la première ligne du csv (liste des Row_Qualifier) et la ligne en cours (Value) 
    On pourra retourner un dictionnaire de {'Column-Family:Column-Qualifier' : Value}
"""
def extract_data(headers, row):
    dict = {}

    for header, value in zip(headers,row):
        # Traitement des données égales à NULL
        if value == 'NULL':
            # On remplace par 1 la quantité, car on considère que si elle existe cette commande, donc au moins un article a été commandé
            if header == 'qte':
                value = '1'
            # Ignorer les lignes contenant un date à NULL
            elif header == 'datcde':
                return None
            # Pour la taille des objets, on a décidé de remplacar par une chaine de caractère vide
            elif header == 'Tailleobj':
                value = ''
            # Toutes les autres valeurs sont remplacés par zéro 0
            else:
                value = '0'

        # Formatage de ma données selon le type
        if value.isdigit():
            value = b'%d'%int(value)
        elif header == 'datcde':
            try:
                value = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                value = value.strftime("%Y-%m-%d %H:%M:%S")  # Par exemple, YYYY-MM-DD HH:mm:ss
                value = value.encode('utf-8')
            except ValueError:
                # Ignorer les lignes contenant une date mal formaté
                print('ERROR DATE', value)
                return None
        else:
            value = bytes(value,'utf-8')
        dict[bytes("cf:"+header.lower(), 'utf-8')] = value
    return dict


# Tester l'existence de la table à créer, et l'effacer, pour repartir sur table vide
try:
    if table_name in connection.tables():
        # Disable de la Table : maTable
        connection.disable_table(table_name)
        # Puis delete de la Table : maTable
        connection.delete_table(table_name)
        print('La table ', table_name, ' a été supprimée.')
except Exception as e:
    print("Delete erreur ", e)

# Créatin de la table
if table_name not in connection.tables():
    connection.create_table(table_name, families)
    print('La table ', table_name, ' a été créée.')

# Etablir la connexion à la table
table = connection.table(table_name)
counter = 0

# insertion des données
with open(csv_file_path, newline='', encoding="utf8") as file:
    reader = csv.reader(file)
    # Charger la première ligne du fichier CSV, qui contiendra les Row_Qualifier
    headers = next(reader)
    # Parcourir les lignes afin de transformer la ligne csv en un donnée prête à être intégré en Hbase
    for row in reader:
        row_key = str(row_key_counter).encode('utf-8')
        row_key_counter += 1
        data = extract_data(headers, row)
        if data:
            # Insertion en base si la ligne en cours n'est pas ignorée
            table.put(row_key, data)

# Fermeture de la
connection.close()
