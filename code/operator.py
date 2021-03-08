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
        test_clean = clean.changeWords([test],dictionary.carDic())
        test_clean = clean.changeRatios(test_clean,dictionary.verbsDic())
        test_clean = clean.changeWords(test_clean,dictionary.convergeVerbsDic())
        test_clean = clean.changeWords(test_clean,dictionary.convergeVehiclesDic())
        test_clean = clean.changeWords(test_clean,dictionary.crashDic())
        test_clean = clean.changeWords(test_clean,dictionary.partsDic())
        test_clean = clean.changeWords(test_clean,dictionary.orderParts())
        test_clean = clean.changeWords(test_clean,dictionary.changebadParts())
        test_clean = clean.changeRepeated(test_clean)
    else:
        test = dataframe[dataframe['responsabilidad'] != 'COMPROMETIDA'].sample()['descripcion'].values[0]
    try:
        model = classifier.ManualClassifier('models/model_kneighbors.pkl')
    except :
        model = classifier.ManualClassifier('D:\proyectos\LegalHub\seguros-interno\code\models\model_kneighbors.pkl')

    print('DESCRIPCION A CLASIFICAR: ')
    print(test)
    # print(test_clean)
    
    print('EXTRACCION DE FEATURES: ')
    model.get_case_features(str(test_clean))
    features = model.profile_transform_kneighbors()
    model.infer_responsability_kneighbors(features)
