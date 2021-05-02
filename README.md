# AI-Log-Analyzer

AI-Log-Analyzer est un projet qui a pour but la detection automatique et en temps réél d'anomalies systemes, à l'aide de l'analyse de logs. AI-Log-Analyzer peut gérer plusieurs systemes au sein d'une infrastructure. Les logs sont récupéré et centralisé dans une base de donnés grace au protocole SNMP. Ils sont ensuite analysés grace à plusieurs couche de machine learning. Enfin, une interface web permet l'affichage des données.

## Installation
Les programmes suivant sont nécessaire pour faire tourner le projet:
- Python3
- Mongodb
- pip3
- cuda (si une carte graphique est utilisé)

Installez ensuite les dépendances nécessaire avec la commande:
```
./install.sh
```

## Utilisation

Lancement de l'analyseur :
python3 analyzer.py

Lancement de l'application web (dans un autre terminal) :
cd api python3 server.py
L'application web est accessible sur le port 4000 : http://localhost:4000

## Fonctionnement interne

- Time-Serie : Un algorithme de machine learning cherche des anomalies dans le nombre de log généré, par exemple une chute du nombre de log (perte d'un systeme), un pic de logs (comportement anormal).
- Analyse Semantique: Un second algorithme lit les logs et tente d'en extraire de l'information, pour comprendre le fonctionnement normal d'un systeme. C'est une implémentation partielle de [Robust-Log](https://netman.aiops.org/~peidan/ANM2019/6.LogAnomalyDetection/LectureCoverage/2019FSE_LogRobust.pdf). Dans un premier temps, les templates de logs sont extraits grace à l'outils [drain3](https://github.com/IBM/Drain3). Les templates sont ensuite transformé en vecteurs sémantiques. Cette transformation permet de gérer les nouveaux types de log, légerement different des logs habituels mais sémantiquement similaire. Pour la classification des logs nous utilisons un LSTM (une catégorie de réseau de neurone récurrent)
