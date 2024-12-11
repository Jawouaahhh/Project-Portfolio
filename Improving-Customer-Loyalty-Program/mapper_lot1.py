import sys
import csv
from datetime import datetime

"""
Ce programme est un mapper qui va seulement : 
    -  Filter la donnée sur les années de commande  
    -  Filter la donnée sur les département des clients 
    -  Envoyer sous forme de clé valeur, les données nécessaires pour la suite du traitement 
"""

YEARS_FILTER = [2006, 2010]
DEPARTMENT_FILTER = [53, 61, 28]


# Filtrage sur l'année
def is_valid_date(date_string):
    try:
        # Caster en date la donnée présente sur le fichier Csv
        date_commande = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
        # Extraire l'année seulement de la date
        year_commande = date_commande.year
        if YEARS_FILTER[0] <= year_commande <= YEARS_FILTER[1]:
            return True
    except ValueError as e:
        pass
    return False


# Filtrage sur le code postal
def is_valid_code_postal(code_postal):
    try:
        # Extraire le département du code postal
        departement = int(code_postal[0:2])
        if departement in DEPARTMENT_FILTER:
            return True
    except ValueError as e:
        pass
    return False


# Charger le fichier CSV
csv_reader = csv.reader(sys.stdin)
# Ignoer la première ligne
next(csv_reader, None)

# Lire l'entrée stdin et la traiter comme un fichier CSV
for row in csv_reader:
    # Extraire les champs du CSV
    code_postal_client, ville_client, code_commande, date_commande, timbre_commande, quantite, points = (
        row[4],
        row[5],
        row[6],
        row[7],
        row[9],
        row[15],
        row[20],
    )
    # Filter les années
    if not is_valid_date(date_commande):
        continue

    # filtrer le code postal
    if not is_valid_code_postal(code_postal_client):
        continue

    # Considérer le manque d'info du timbre_code comme égale à 0
    if timbre_commande == "NULL":
        timbre_commande = 0

    # Retirer les lignes avec des points négatifs (ça représente des cadeaux et non des commandes)
    if points != "NULL" and int(points) < 0:
        continue

    mapper_key = code_commande
    mapper_value = ville_client + "," + timbre_commande + "," + quantite

    print("%s,%s" % (mapper_key, mapper_value))
