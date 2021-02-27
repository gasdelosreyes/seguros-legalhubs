import pandas as pd
import time
from utils import dfmodule, clean, dictionary

if __name__ == "__main__":
    start = time.time()
    dataframe = pd.DataFrame()
    dataframe = dfmodule.appendDataFrames(dataframe, dfmodule.read_file_csv('../dataset/casos/auto.csv'), [0,1,2])
    test = dataframe.sample()
    print(test)