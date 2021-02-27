from fuzzywuzzy import fuzz

def ratios(w, dic):
    try:
        if(w in dic):
            return w
        aux = 0
        word = ''
        for i in dic:
            if (aux <= fuzz.ratio(w, i) and 85 <= fuzz.ratio(w, i)):
                aux = fuzz.ratio(w, i)
                word = i
        if word != '':
            return word
        return w
    except TypeError:
        return w