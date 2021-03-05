import pandas as pd
import time
from utils import dfmodule, clean, dictionary
from modules import classifier

if __name__ == "__main__":
    start = time.time()
    dataframe = pd.DataFrame()
    dataframe = dfmodule.appendDataFrames(dataframe, dfmodule.read_file_csv('../dataset/casos/auto-clean.csv'), [0,1,2])
    test = dataframe.sample()['descripcion'].values[0]
    model = classifier.ManualClassifier()
    print('DESCRIPCION A CLASIFICAR: ')
    print(test)
    print('CLASIFICACION REALIZADA: ')
    model.infer_case(case_str=test)