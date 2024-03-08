import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler

# Pfad zur CSV-Datei
path = 'adult.csv'

# Lese die Daten ein und zeige die ersten 10 Zeilen zur Überprüfung
data = pd.read_csv(path, delimiter=',')
print(data.head(10))  # Ausgabe der ersten 10 Datensätze

# Transformiere Spalten mit Zeichenketten in numerische Werte
# Hier werden kategorische Spalten in numerische Codes umgewandelt
cols_to_transform = ['workclass','education','marital-status','occupation','relationship','race','gender', 'native-country', 'income']
data[cols_to_transform] = data[cols_to_transform].astype('category')
data[cols_to_transform] = data[cols_to_transform].apply(lambda x: x.cat.codes)
print(data.head(10))  # Zeige die ersten 10 Datensätze nach der Transformation

# Ermittle die optimale Gruppenzahl mithilfe des KElbowVisualizers
# Optional: Kommentiere den StandardScaler-Teil aus, wenn die Daten standardisiert werden sollen

#s_scaler = StandardScaler()
#data = pd.DataFrame(s_scaler.fit_transform(data), columns=data.columns)

# Initialisiere das KMeans-Modell
model = KMeans()

# Erstelle den KElbowVisualizer mit dem KMeans-Modell, Bereich für k und deaktiviere die Timing-Ausgabe
visualizer = KElbowVisualizer(model, k=(2,9), timings=False)

# Führe den Elbow-Visualizer mit den Daten aus, um die optimale Anzahl von Clustern zu finden
visualizer.fit(data)

# Zeige das Elbow-Diagramm
visualizer.show()

# Basierend auf dem Elbow-Diagramm, wähle die Anzahl der Cluster (hier: 4)
kmeans = KMeans(n_clusters=4)

# Wende das KMeans-Modell an, um Cluster-Zuordnungen für die Daten zu erhalten
pred = kmeans.fit_predict(data)

# Füge die Cluster-Zuordnungen als neue Spalte den Daten hinzu
data_new = pd.concat([data, pd.DataFrame(pred, columns=['Label'])], axis=1)

# Zeige die ersten paar Zeilen des neuen Datensatzes mit den Clustern
print(data_new.head())

# Speichere den neuen Datensatz mit Cluster-Zuordnungen in einer CSV-Datei
data_new.to_csv("Label.csv")
