# AI-Log-Analyzer

AI-Log-Analyzer est un projet qui a pour but la detection automatique et en temps réél d'anomalies systemes, à l'aide de l'analyse de logs. AI-Log-Analyzer peut gérer plusieurs systemes au sein d'une infrastructure. Les logs sont récupéré et centraliser dans une base de donnés grace au protocole SNMP. Ils sont ensuite analysés grace à plusieurs couche de machine learning. Enfin, une interface web permet l'affichage des données.

## Installation
Les programmes suivant sont nécessaire pour faire tourner le projet:
- Python3
- Mongodb
- pip3

Installez ensuite les dépendances nécessaire avec la commande:
```
pip3 install requirement.txt
```

## Fonctionnement interne

### Reception des logs
Les logs de chaque systemes sont envoyés via le protocole SNMP. 
