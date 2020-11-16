import re

import pandas as pd
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from nltk.corpus import swadesh
from nltk.probability import FreqDist
from nltk.text import Text
from nltk.tokenize import word_tokenize

dataset = pd.read_csv('../dataset/casos_filtrados/casos_filtrados_total.csv')
print(dataset['fuzzy'])
