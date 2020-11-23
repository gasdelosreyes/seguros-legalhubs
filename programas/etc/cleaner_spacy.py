# %%[markdown]
# Tokenizamos y luego tag
# %%
from scipy.sparse import data
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing, feature_selection, metrics
import matplotlib.pyplot as plt
import seaborn as sns
from os import sep
from re import split
import spacy
import es_core_news_sm
import pandas as pd
import numpy as np
path = '../dataset/'
# %%
nlp = es_core_news_sm.load()
# %%
bici = pd.read_csv('../dataset/casos/bici_clean.csv', sep=';')
moto = pd.read_csv('../dataset/casos/moto_clean.csv', sep=';')
auto = pd.read_csv('../dataset/casos/auto_clean.csv', sep=';')
peaton = pd.read_csv('../dataset/casos/peaton_clean.csv', sep=';')
# %%
bici['descripcion_del_hecho - Final'] = bici['descripcion_del_hecho - Final'].astype(
    str).apply(str.split)
peaton['descripcion_del_hecho - Final'] = peaton['descripcion_del_hecho - Final'].astype(
    str).apply(str.split)
moto['descripcion_del_hecho - Final'] = moto['descripcion_del_hecho - Final'].astype(
    str).apply(str.split)
auto['descripcion_del_hecho - Final'] = auto['descripcion_del_hecho - Final'].astype(
    str).apply(str.split)
# %% [markdown]
bici.to_csv(path + 'token/bici_token.csv')
moto.to_csv(path + 'token/moto_token.csv')
auto.to_csv(path + 'token/auto_token.csv')
peaton.to_csv(path + 'token/peaton_token.csv')

# %%
#Tokenization - Etiquetado - Lemmatization
# La función nlp recibe un string
# El objeto Doc es una secuencia de objetos Token, osea cada fila de la columnda es un Doc.
bici = pd.read_csv('../dataset/casos/bici_clean.csv', sep=';')
moto = pd.read_csv('../dataset/casos/moto_clean.csv', sep=';')
auto = pd.read_csv('../dataset/casos/auto_clean.csv', sep=';')
peaton = pd.read_csv('../dataset/casos/peaton_clean.csv', sep=';')

bici['descripcion_del_hecho - Final'] = bici['descripcion_del_hecho - Final'].astype(
    str).apply(nlp)
peaton['descripcion_del_hecho - Final'] = peaton['descripcion_del_hecho - Final'].astype(
    str).apply(nlp)
moto['descripcion_del_hecho - Final'] = moto['descripcion_del_hecho - Final'].astype(
    str).apply(nlp)
auto['descripcion_del_hecho - Final'] = auto['descripcion_del_hecho - Final'].astype(
    str).apply(nlp)

# %%
# Funcion que recibe la fila con cada palabra tipo Doc y devuelve cada palabra en par ordenado (Lemma,Pos)


def Doc2Par(line):
    aux = []
    for i in line:
        aux.append((i.lemma_, i.pos_))
    return aux

# %%


bici['Spacy Lemma-Pos'] = bici['descripcion_del_hecho - Final'].apply(Doc2Par)
moto['Spacy Lemma-Pos'] = moto['descripcion_del_hecho - Final'].apply(Doc2Par)
auto['Spacy Lemma-Pos'] = auto['descripcion_del_hecho - Final'].apply(Doc2Par)
peaton['Spacy Lemma-Pos'] = peaton['descripcion_del_hecho - Final'].apply(
    Doc2Par)

# %% [markdown]

bici.to_csv(path + 'lemma-pos/bici_lemma-pos.csv', sep=';')
moto.to_csv(path + 'lemma-pos/moto_lemma-pos.csv', sep=';')
auto.to_csv(path + 'lemma-pos/auto_lemma-pos.csv', sep=';')
peaton.to_csv(path + 'lemma-pos/peaton_lemma-pos.csv', sep=';')

# %% [markdown]
# Aca comienza el analisis del texto y la obtención de los features
# Debemos codificar las categorias de los tipos de accidente pero como aca tenemos todo de la misma categoría tal vez no haga falta, en realidad pienso que deberíamos trabajar con los datos en bruto para que vaya decidiendo.
# %%
dataset = pd.DataFrame()
dataset = pd.concat([bici, moto, auto, peaton])
dataset.pop('Unnamed: 0')
dataset = dataset.rename(columns={'Idenx original': 'Index original'})
dataset = pd.concat([dataset, pd.get_dummies(
    dataset['tipo_de_accidente'])], axis=1)
dataset.sort_index()
# %%

dtf_train, dtf_test = model_selection.train_test_split(dataset, test_size=0.1)
# %%
y_train = dtf_train["auto - auto"].values
y_test = dtf_test["auto - auto"].values
# %%
# for bag-of-words
# Count (classic BoW)
# puede que nosotros no necesitemos trabajar con tatos features
# el ngram_range hace referencia a cuantas palabras en conjuto toma

vectorizer = feature_extraction.text.CountVectorizer(
    max_features=10000)
corpus = dtf_train['descripcion_del_hecho - Final'].astype(str)
vectorizer.fit(corpus)
x_train = vectorizer.transform(corpus)
dic_vocab = vectorizer.vocabulary_

# %%
# Si la palabra 'izquierda existe en el vocabulario, este comando imprime un número N, lo que significa que la N-ésima característica de la matriz es esa palabra.
print(
    dic_vocab['izquierda'],
    dic_vocab['izq']
    # 1358 1348

)
# %% [markdown]
# Feature selection

# %%
# y es a lo que apuntamos, debería ser de entrenamiento.
y = dtf_train['auto - auto']
X_names = vectorizer.get_feature_names()
p_value_limit = 0.95
# %%
dtf_features = pd.DataFrame()
for cat in np.unique(y):

    chi2, p = feature_selection.chi2(x_train, y == cat)

    dtf_features = dtf_features.append(pd.DataFrame(
        {"feature": X_names, "score": 1-p, "y": cat}))

    dtf_features = dtf_features.sort_values(
        ["y", "score"], ascending=[True, False])

    dtf_features = dtf_features[dtf_features["score"] > p_value_limit]
# %%
X_names = dtf_features["feature"].unique().tolist()

# %%
# aca refiteamos
vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=X_names)
vectorizer.fit(corpus)
X_train = vectorizer.transform(corpus)
dic_vocabulary = vectorizer.vocabulary_
# %%
classifier = naive_bayes.MultinomialNB()

# %%
# pipeline
model = pipeline.Pipeline(
    [("vectorizer", vectorizer), ("classifier", classifier)])  # train classifier
model["classifier"].fit(X_train, y_train)  # test
X_test = dtf_test["descripcion_del_hecho - Final"].values
predicted = model.predict(X_test)
predicted_prob = model.predict_proba(X_test)

# %%
classes = np.unique(y_test)
y_test_array = pd.get_dummies(y_test, drop_first=False).values
# Plot confusion matrix
# 82% de precision con un 0.3 y 0.2
# 86% de precision con un 0.1
cm = metrics.confusion_matrix(y_test, predicted)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
            cbar=False)
ax.set(xlabel="Pred", ylabel="True", xticklabels=classes,
       yticklabels=classes, title="Confusion matrix")
plt.yticks(rotation=0)
fig, ax = plt.subplots(nrows=1, ncols=2)
# %%
