import spacy
import es_core_news_sm
import pandas as pd

nlp = es_core_news_sm.load()
dataset = pd.read_excel('../dataset/casos_universidad.xlsx')

#doc = nlp(dataset["descripcion_del_hecho - Final"][7])

#print("Text\tLemma\tPOS\tTag\tDep\tShape\talpha\tstop")

#for token in doc:
#    print(token.text,'\t', token.lemma_,'\t', token.pos_,'\t', token.tag_,'\t', token.dep_,'\t', token.shape_,'\t', token.is_alpha,'\t', token.is_stop)

print(dataset['tipo_de_accidente'].head())
