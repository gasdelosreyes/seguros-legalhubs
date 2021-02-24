import pandas as pd
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
# tabla.plot_movimiento()
tabla.plot_casos_completos()
# tabla.plot_ubicacion_vial()
# tabla.plot_posicion_impact()
# tabla.plot_quien()
# tabla.cross_plot()
tabla.to_csv()
