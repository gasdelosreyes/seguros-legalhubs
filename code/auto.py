import pandas as pd
from utils import dfmodule, clean, dictionary
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import time

if __name__ == "__main__":
    start = time.time()
    dataframe = pd.DataFrame()
    dataframe = dfmodule.appendDataFrames(dataframe, dfmodule.read_file_csv('../dataset/casos/auto.csv'), [0,1,2])
    dataframe['descripcion'] = clean.changeWords(dataframe['descripcion'], dictionary.carDic())
    dataframe['descripcion'] = clean.changeRatios(dataframe['descripcion'], dictionary.verbsDic())
    dataframe['descripcion'] = clean.changeWords(dataframe['descripcion'], dictionary.convergeVerbsDic())
    dataframe['descripcion'] = clean.changeWords(dataframe['descripcion'], dictionary.crashDic())
    dataframe['descripcion'] = clean.changeWords(dataframe['descripcion'], dictionary.partsDic())
    dataframe['descripcion'] = clean.changeWords(dataframe['descripcion'], dictionary.orderParts())
    dataframe['descripcion'] = clean.changeWords(dataframe['descripcion'], dictionary.changebadParts())
    dataframe['descripcion'] = clean.changeRepeated(dataframe['descripcion'])
    dataframe.to_csv('../dataset/casos/auto-clean.csv', index=False, header=True)
    print('Tiempo de ejecución: ',round(time.time()-start,2)//60,'min',round(time.time()-start,2)%60,'s')