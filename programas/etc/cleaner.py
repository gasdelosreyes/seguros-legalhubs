'''
    Vector with the convertion between hex and utf-8
'''
vector = [  [r'\'e1',r'á'],
            [r'\'e9',r'é'],
            [r'\'ed',r'í'],
            [r'\'fa',r'ú'],
            [r'\'f3', r'ó'],
            [r'\'e0', r'à'],
            [r'\'e8', r'è'],
            [r'\'ec', r'ì'],
            [r'\'f2', r'ò'],
            [r'\'f9', r'ù'],
            [r'\'c1',r'Á'],
            [r'\'d3',r'Ó'],
            [r'\tab',''],
            [r'\n',''],
            [r'\par_x000D_',''],
            [r'_x000D_',''],
            ]

'''
    Search for "Descripción" in the dataset
'''
description = []
with open('../dataset/casos_universidad.txt', 'r', encoding='UTF-8') as f:
    for line in f:
        if(line.startswith('Descr')):
            description.append(line)


# First try
# splitted = desc.split()
# for value in splitted:
#     for row in vector:
#         if(row[0] in value):
#             value = value.replace(row[0],row[1])
#             print(value)

# Second try
# for i in range(0, len(description)):
#     for desc in description:
#         for row in vector:
#             if(row[0] in desc):
#                 desc = desc.replace(row[0],row[1])
#         description[i] = desc


'''
    Create new file with only the description converted
'''
with open('../dataset/descripciones.txt','w') as f:
    for desc in description:
        for row in vector:
            if(row[0] in  desc):
                desc = desc.replace(row[0],row[1])
        f.write(desc)
f.close()