import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
import graphviz
np.random.seed(7)

df = pd.read_csv('tmp/Tabla1_dummie.csv')

x_train, x_test, y_train, y_test = train_test_split(df[[col for col in df.columns if col != 'responsabilidad']], df['responsabilidad'], test_size=0.2, random_state=7)

clf = tree.DecisionTreeClassifier()
clf.fit(x_train, y_train)
tree.plot_tree(clf)
print(clf.score(x_test, y_test))
dot_graph = tree.export_graphviz(clf, out_file=None, feature_names=[i for i in df.columns if i != 'responsabilidad'], class_names=['comprometida', 'no comprometida'])

graph = graphviz.Source(dot_graph)
graph.render('responsabilidad')
