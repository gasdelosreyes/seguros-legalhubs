import pandas as pd
from utils import dfmodule, clean, dictionary
from nltk.tokenize import word_tokenize
import time

if __name__ == "__main__":
    start = time.time()
    dataframe = pd.DataFrame()
    '''
        Cargando archivos
    '''
    dataframe = dfmodule.appendDataFrames(dataframe, dfmodule.read_file_xls('../dataset/casos_universidad.xlsx'), [5, 11, 7])
    dataframe = dfmodule.appendDataFrames(dataframe, dfmodule.read_file_xls('../dataset/casos_zurich_20201228.xlsx', 'Dataset'), [5, 11, 7])
    dataframe = dfmodule.appendDataFrames(dataframe, dfmodule.read_file_xls('../dataset/casos_sin-responsabilidad.xlsx'), [5, 11, 7])
    dataframe = dataframe.rename(columns={"descripcion_del_hecho - Final": "descripcion"})
    dataframe = dataframe.drop(dfmodule.wrong(dataframe))
    dataframe['descripcion'] = dataframe['descripcion'].apply(dictionary.changeDic,vector=dictionary.codexDic())
    dataframe['descripcion'] = dataframe['descripcion'].apply(dictionary.changeDic,vector=dictionary.unicodexDic())
    dataframe['descripcion'] = dataframe['descripcion'].apply(dictionary.changeDic,vector=dictionary.formatDic())
    dataframe['descripcion'] = dataframe['descripcion'].apply(clean.general)
    dataframe['descripcion'] = dataframe['descripcion'].apply(dictionary.changeDic,vector=dictionary.postformatDic())
    auto, moto, bici, peaton = dfmodule.separador(dataframe)
    auto.to_csv('../dataset/casos/auto.csv', index=False, header=True)
    moto.to_csv('../dataset/casos/moto.csv', index=False, header=True)
    bici.to_csv('../dataset/casos/bici.csv', index=False, header=True)
    peaton.to_csv('../dataset/casos/peaton.csv', index=False, header=True)
    print('Tiempo de ejecución: ',round(time.time()-start,2)//60,'min',round(time.time()-start,2)%60,'s')