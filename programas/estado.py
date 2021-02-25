import pandas as pd
import operator
from objetos import *

df = pd.read_csv('../dataset/casos/auto-clean.csv')


tabla = TablaCasos('Tabla1')

for i in range(len(df)):
	caso = Caso()
	caso.set_descripcion(df.loc[i, 'descripcion'])
	caso.set_idxDesc(i)  # no estoy seguro si es en i del csv o debería ser el i del csv original
	caso.set_responsabilidad(df.loc[i, 'responsabilidad'])
	tabla.set_caso(caso)

tabla.set_caso(caso, update=True)
tabla.update()
tabla.status()
import re
def get_stats(desc):
    # esto lo tengo que hacer con regex
    importat_words = 'izquierda derecha delante adelante trasera atras tercero asegurado frente delantera impacto impactando impactado colision colisionar colisionando'
    stat = {}
    for word in desc.split():
         stat.update({word : 0})
    for word in desc.split():
        if 3<len(word):
            if re.search(word,importat_words):
                stat[word] += 2.5
            else:
                stat[word]+=1
    stat = sorted(stat.items(), key=operator.itemgetter(1))
    return stat
 
for caso in tabla.get_casos():
    if not len(caso.get_impac_position()):
        print(get_stats(caso.get_descripcion()))
        break










# tabla.plot_movimiento()
# tabla.plot_casos_completos()
# tabla.plot_ubicacion_vial()
# tabla.plot_posicion_impact()
# tabla.plot_quien()
# tabla.cross_plot()
# tabla.to_csv()
