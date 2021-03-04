import pandas as pd 
import markovify as mk 

df = pd.read_csv('../dataset/casos/auto-clean.csv')
df1 = pd.read_csv('../dataset/casos/auto.csv')

col = pd.concat([df['descripcion'][df['responsabilidad']!='COMPROMETIDA'][:400],df1['descripcion'][df1['responsabilidad']!='COMPROMETIDA']][:-1],ignore_index=True)
print(col)

combined_model = None
for row in df1['descripcion'][df1['responsabilidad']!='COMPROMETIDA'].dropna():
    model = mk.Text(row,retain_original=False)
    if combined_model:
        combined_model = mk.combine([combined_model,model])
    else:
        combined_model = model
model_json = combined_model.to_json()
import json
with open("model.json", "w") as write_file:
    json.dump(model_json, write_file)

for i in range(10):
    print(combined_model.make_short_sentence(200))
    print()
    
    
