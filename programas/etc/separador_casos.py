#%%
import pandas as pd
#%%
datos = pd.read_excel('../dataset/casos_universidad.xlsx')

#%%

datos.drop(['nro_caso','codigo_postal','nro_siniestro','domicilio_sini','fec_siniestro','consideraciones_novedadades','consideraciones_novedadades - Final','cod_accidente'],1,inplace=True)

# %%
datos.head()
# %%
moto = datos[datos['tipo_de_accidente']=='AUTO - MOTO' ]
#744 casos
# %%
bici = datos[datos['tipo_de_accidente']=='AUTO - CICLISTA' ]
#97
#%%
peaton = datos[datos['tipo_de_accidente']=='PEATON' ]
#95
#%%
auto = datos[datos['tipo_de_accidente']=='AUTO - AUTO' ]
#193
#en total son 1129
# %%
#PASAMOS TODOS A UN .CSV SEPARADOS
path = '../dataset/'
moto.to_csv(path + 'moto.csv')
bici.to_csv(path + 'bici.csv')
peaton.to_csv(path + 'peaton.csv')
auto.to_csv(path + 'auto.csv')

# %%
