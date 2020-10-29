#%%
import pandas as pd
dataset = pd.read_excel('../dataset/casos_universidad.xlsx')

# %%
#dataset = ds
ds_df = pd.DataFrame(dataset)
# %%
ds_df.dtypes
# %%
# Con esto obtenemos la columna de "descripcion_del_hecho - Final"
ds_df.iloc[:,[11]]
# %%
