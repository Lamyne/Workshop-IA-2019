### Importer les modules

import pandas as pd
import pydotplus


### 01) Lecture de la bse de donnees
data = pd.read_csv("pima-indians-diabetes.csv", sep=";", header = 0)

df = pd.DataFrame(data)
X = df.iloc[:, 0:8]
y = df.iloc[:, 8]

### 02) Recuperation des noms des variables
features = list(X)
print("La liste des variable est    =", features )
print(df.info())



### 03)  Construction du modèle

from sklearn import tree
### Mise en place des configurations
clf = tree.DecisionTreeClassifier(criterion = "gini", max_depth=6, min_samples_leaf=2)

### Affichage des configurations du modele DT
print("model ====", clf)

### Entrainement du modèle
clf = clf.fit(X, y)



### 04) Affichage du modèle

#### Sauvgarder le modele sous forme de regles
with open("DiabeteClass.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f, feature_names=features)

### Sauvgarder le modele sous forme de PDF-image
my_data = tree.export_graphviz(clf, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(my_data)
graph.write_pdf("DiabeteClass.pdf")


### 05) Prédire une nouvelle valeur :
#['grossesses', 'glucose', 'press_art', 'peau', 'insuline', 'IMC', 'fctDiabete']
NewSample = pd.DataFrame({'grossesses': [4], 'glucose': [148],'press_art': [70],'peau': [35],'insuline': [0],'IMC': [33.6],'fctDiabete': [0.625],'age': [40]})
y_pred = clf.predict(NewSample)
print("Le resultat est   === ", y_pred)
