from sys import argv
import pandas as pd

from utils import dfmodule, clean, dictionary
from modules import classifier
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    dataframe = pd.DataFrame()
    dataframe = dfmodule.appendDataFrames(dataframe, dfmodule.read_file_csv('../dataset/casos/auto-clean.csv'), [0,1,2])
    # solamente toma en test los casos no comprometidos.
    if(argv[1]):
        test = str(argv[1])
    else:
        test = dataframe[dataframe['responsabilidad'] != 'COMPROMETIDA'].sample()['descripcion'].values[0]
    model = classifier.ManualClassifier('D:\proyectos\LegalHub\seguros-interno\code\models\model_kneighbors.pkl')
    print('DESCRIPCION A CLASIFICAR: ')
    print(test)
    print('EXTRACCION DE FEATURES: ')
    model.get_case_features(test)
    features = model.profile_transform_kneighbors()
    model.infer_responsability_kneighbors(features)