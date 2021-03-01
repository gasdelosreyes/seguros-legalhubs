import re

def changeDic(text, **kwargs):
    """
        Cambia todos los valores en el diccionario especificado
    """
    for value in kwargs.get('vector'):
        text = re.sub(value[0],value[1],text)
    return text

'''
    UNICODE DICTIONARYS
'''

def codexDic():
    return [
            (r'\\\'b1', r'ni'),
            (r'\\\'d0', r'ni'),
            (r'\\\'e1', r'á'),
            (r'\\\'df', r'á'),
            (r'\\\'e9', r'é'),
            (r'\\\'ed', r'í'),
            (r'\\\'cd', r'í'),
            (r'\\\'y' , r'í'),
            (r'\\\'f3', r'ó'),
            (r'\\\'cb', r'ó'),
            (r'\\\'be', r'ó'),
            (r'\\\'fa', r'ú'),
            (r'\\\'dc', r'ú'),
            (r'\\\'fd', r'ý'),
            (r'\\\'e0', r'à'),
            (r'\\\'e8', r'è'),
            (r'\\\'ec', r'ì'),
            (r'\\\'f2', r'ò'),
            (r'\\\'f9', r'ù'),
            (r'\\\'b7', r'ù'),
            (r'\\\'c0', r'À'),
            (r'\\\'c8', r'È'),
            (r'\\\'cc', r'Ì'),
            (r'\\\'d2', r'Ò'),
            (r'\\\'d9', r'Ù'),
            (r'\\\'c1', r'Á'),
            (r'\\\'dd', r'Ý'),
            (r'\\\'c9', r'É'),
            (r'\\\'cd', r'Í'),
            (r'\\\'d3', r'Ó'),
            (r'\\\'da', r'Ú'),
    ]

def unicodexDic():
    return [
            (r'\\u9524\?',r'Á'),
            (r'\\u9492\?',r'Á'),
            (r'\\u9552\?',r'Í'),
            (r'\\u9568\?',r'Í'),
            (r'\\u8215\?',r'Ó'),
            (r'\\u9556\?',r'É'),
            (r'\\u9484\?',r'Ú'),
            (r'\\nu9553\?',r'numero'),
            (r'\\u9617\?',r' grados'),
            (r'\\u9508\?',r' '),
    ]

def formatDic():
    return [
            (r'\\uc1', r''),
            (r'\\red\d+\\green\d+lue\d+\;', r''),
            (r'\\green\d+\;', r''),
            (r'ff(?:\:|\,|\.|\-)', r''),
            (r'cc(?:\:|\,|\.|\-)', r''),
            (r'denuncia asegurado:', r''),
            (r'\\fcharset\d+\ \w+\;', r''),
    ]

def postformatDic():
    return [
            (r'ff\ ', r''),
            (r'fs\ ', r''),
            (r'cc\ ', r''),
            (r'tx\ ', r''),
    ]

def carDic():
    return [
            ((  r'vt',r'vtv',r'vhs',r'vehi'r'vhlo',r'vh',r'vhc',r'vehculo',r'vhls',r'vehl',
                r'vhl',r'camioneta',r'camion',r'camien',r'moto',r'motos', r'vahiculo',
                r'auto',r'automovil',r'rodado',r'autos',r'veh',r'coche',r'vehyculo'
                ),  r'vehiculo'),
            ((  r'espejos',
                ),  r'espejo'),
            ((  r'aut',r'autop',
                ),  r'autopista'),
            ((  r'circ',r'transitaba'
                ),  r'circulaba'),
            ((  r'conductora',),r'conductor'),
            ((  r'acompanante',),r'acompaniante'),
            ((  r'no puedo evitar',r'no pudo evitar',r'no logra evitar',r'no pude evitar',r'no logre evitar',
                r'no puede evitar',r'no logran evitar',r'no pueden evitar',r'no pudiendo evitar',
                r'no logrando evitar',
                ),  r'no logro evitar'),
            ((  r'interseccien',r'interseccin',
                ),  r'interseccion'),
            ((r'eptica',),r'optica'),
            ((r'guardabarros',),r'guardabarro'),
            ((r'paragolpes',),r'paragolpe'),
            ((r'a tras',), r'atras'),
            ((r'a delante',), r'adelante'),
            ((r'garage', r'gge', r'cochera'), r'garaje'),
            ((r'av',r'avda'), r'avenida'),
            ((r'lateral', r'costado', r'sector',r'lat'), r'parte'),
            ((r'ero', r'taxi', r'taxista',r'terc',r'chofer',r'vecino',), r'tercero'),
            ((r'asegurada', r'aseg'), r'asegurado'),
            ((r'izquierdo', r'izq'),r'izquierda'),
            ((r'derecho', r'der', r'dere',r'dcha',r'dcho',r'erecho',r'erecha',r'derech'),r'derecha'),
            ((r'trasero',),r'trasera'),
            ((r'delantero', r'trompa',r'posterior',r'delantea'),r'delantera'),
    ]

def verbsDic():
    return [
        r'colisionandome',r'colisionandole',r'colisionandolo',r'colisionando',r'colisionado',r'colisionarle',r'colisionarlo',r'colisionar',r'colisiono',r'colisione',r'colisiona',r'colision',
        r'impactandome',r'impactandole',r'impactandolo',r'impactando',r'impactado',r'impactarme',r'impactarlo',r'impactarle',r'impactar',r'impactado',r'impacta',r'impacto',r'impacte',
        r'embestido',r'embestirlo',r'embestirme',r'embestirle',r'embestir',r'embesti',
        r'embistiendome',r'embistiendole',r'embistiendolo',r'embistiendo',r'embistio',r'embisto',r'embiste',
        r'chocandome',r'chocandole',r'chocandolo',r'chocarlo',r'chocarle',r'chocarme',r'chocando'r'chocado',r'chocar',r'choque',r'choco',r'choca',
        r'tocandome',r'tocandole',r'tocandolo',r'tocando',r'tocarle',r'tocarme',r'tocado',r'tocarlo',r'tocar',r'toco',r'toca',r'toque',
        r'raspandome',r'raspandole',r'raspandolo',r'rasparme',r'rasparle',r'rasparlo',r'raspar',r'raspe',r'raspo',r'raspa',
        r'daniandome',r'daniandole',r'daniandolo',r'daniando',r'daniarme',r'daniarlo',r'daniarle',r'daniar',r'dania',r'danio',r'danie',
        r'rozandome',r'rozandole',r'rozandolo',r'rozando',r'rozarme',r'rozarlo',r'rozarle',r'rozar',r'rozo',r'roza',r'roce',
        r'rayandome',r'rayandole',r'rayandolo',r'rayando',r'rayarme',r'rayarlo',r'rayarle',r'rayar',r'rayo',r'raya',r'raye',
        r'aboyandome',r'aboyandole',r'aboyandolo',r'aboyando',r'aboyarme',r'aboyarlo',r'aboyarle',r'aboyado',r'aboyar',r'aboyo',r'aboya',r'aboye',
        r'golpeandome',r'golpeandole',r'golpeandolo',r'golpeando',r'golpearme',r'golpearle',r'golpearlo',r'golpeado',r'golpear',r'golpeo',r'golpee',r'golpea',r'golpe'
    ]

def convergeVerbsDic():
    # IMPACTO, CHOQUE, TOQUE, ROCE NO SE TIENEN EN CUENTA POR AMBIGUEDAD
    return [
        ((  
            r'impactandome', r'embistiendome', r'chocandome', r'tocandome', r'raspandome',
            r'daniandome', r'rozandome', r'rayandome', r'aboyandome', r'golpeandome',
        ),r'colisionandome'),
        ((
            r'impactandole', r'embistiendole', r'chocandole', r'tocandole', r'raspandole',
            r'daniandole', r'rozandole', r'rayandole',r'aboyandole',r'golpeandole'
        ),r'colisionandole'),
        ((
            r'impactandolo', r'embistiendolo', r'chocandolo',r'tocandolo',r'raspandolo',
            r'daniandolo',r'rozandolo',r'rayandolo',r'aboyandolo',r'golpeandolo'
        ),r'colisionandolo'),
        ((
            r'impactando', r'embistiendo', r'chocando',r'tocando',r'raspando',
            r'daniando',r'rozando',r'rayando',r'aboyando',r'golpeando',
        ),r'colisionando'),
        ((
            r'impactado', r'embestido', r'chocado',r'tocado',
            r'daniado',r'rozado',r'golpeado',r'aboyado',r'rayado',
        ),r'colisionado'),
        ((
            r'impactarme', r'embestirme', r'chocarme',r'tocarme',r'rasparme',
            r'daniarme',r'rozarme',r'rayarme',r'aboyarme',r'golpearme'
        ),r'colisionarme'),
        ((
            r'impactarlo', r'embestirlo', r'chocarlo',r'tocarlo',r'rasparlo',
            r'daniarlo',r'rozarlo',r'rayarlo',r'golpearlo',r'aboyarlo'
        ),r'colisionarlo'),
        ((
            r'impactarle', r'embestirle', r'chocarle',r'tocarle',r'rozarle',
            r'rayarle',r'daniarle',r'golpearle',r'aboyarle',r'rasparle'
        ),r'colisionarle'),
        ((
            r'impactar', r'embestir', r'chocar',r'tocar',r'raspar',
            r'daniar',r'rozar',r'rayar',r'aboyar',r'golpear'
        ),r'colisionar'),
        ((
            r'embistio', r'embisto', r'choco',r'toco',
            r'raspo',r'danio',r'rozo',r'rayo',r'aboyo',r'golpeo'
        ),r'colisiono'),
        ((
            r'impacte', r'embesti',
            r'raspe',r'danie',r'raye',r'aboye',r'golpee'
        ),r'colisione'),
        ((
            r'impacta', r'choca',r'toca',r'raspa',r'dania',
            r'roza',r'raya',r'aboya',r'golpea', r'embiste'
        ),r'colisiona'),
    ]

def convergeVehiclesDic():
    return [
        ((r'mi vehiculo',r'vehiculo asegurado',r'vehiculo de asegurado',r'unidad asegurado',r'del asegurado'),r'vehiculo del asegurado'),
        ((r'vehiculo tercero',r'otro vehiculo',r'del tercero',r'un vehiculo',r'un tercero'),r'vehiculo del tercero'),
        ((r'al asegurado',r'a asegurado'),r'al vehiculo del asegurado'),
        ((r'al tercero',r'a tercero'),r'al vehiculo del tercero'),
    ]

def crashDic():
    #verbo + dome, verbo + rme, me + verbo,
    # COMPLEJOS: LO VERBO + A, LE VERBO + A,LO EMBISTE, FUE VERBO+IDO/ADO
    # COMPLEJOS: Y VERBO + A/E/O, E IMPACTO/IMPACTA/IMPACTE
    return [
            ((
                r'colisiona al asegurado',r'colisiono al asegurado',r'colisionar al asegurado',r'colisionando al asegurado'
                r'impacto al asegurado',r'impacta al asegurado',r'impactar al asegurado',r'impactando al asegurado',
                r'embiste al asegurado',r'embistio al asegurado',r'embestir al asegurado',r'embistiendo al asegurado',
                r'choca al asegurado',r'choco al asegurado',r'chocar al asegurado',r'chocando al asegurado',
                r'toca al asegurado',r'toco al asegurado',r'tocar al asegurado',r'tocando al asegurado',
                r'raspa al asegurado',r'raspo al asegurado',r'raspar al asegurado',r'raspando al asegurado',
                r'raya al asegurado',r'rayo al asegurado',r'rayar al asegurado',r'rayando al asegurado',
                r'aboya al asegurado',r'aboyo al asegurado',r'aboyar al asegurado',r'aboyando al asegurado',
                r'golpea al asegurado',r'golpeo al asegurado',r'golpear al asegurado',r'golpeando al asegurado',
                r'colisiona al vehiculo asegurado',r'colisiono al vehiculo asegurado',r'colisionar al vehiculo asegurado',r'colisionando al vehiculo asegurado'
                r'impacto al vehiculo asegurado',r'impacta al vehiculo asegurado',r'impactar al vehiculo asegurado',r'impactando al vehiculo asegurado',
                r'embiste al vehiculo asegurado',r'embistio al vehiculo asegurado',r'embestir al vehiculo asegurado',r'embistiendo al vehiculo asegurado',
                r'choca al vehiculo asegurado',r'choco al vehiculo asegurado',r'chocar al vehiculo asegurado',r'chocando al vehiculo asegurado',
                r'toca al vehiculo asegurado',r'toco al vehiculo asegurado',r'tocar al vehiculo asegurado',r'tocando al vehiculo asegurado',
                r'raspa al vehiculo asegurado',r'raspo al vehiculo asegurado',r'raspar al vehiculo asegurado',r'raspando al vehiculo asegurado',
                r'raya al vehiculo asegurado',r'rayo al vehiculo asegurado',r'rayar al vehiculo asegurado',r'rayando al vehiculo asegurado',
                r'aboya al vehiculo asegurado',r'aboyo al vehiculo asegurado',r'aboyar al vehiculo asegurado',r'aboyando al vehiculo asegurado',
                r'golpea al vehiculo asegurado',r'golpeo al vehiculo asegurado',r'golpear al vehiculo asegurado',r'golpeando al vehiculo asegurado',
                r'colisiona al vehiculo del asegurado',r'colisiono al vehiculo del asegurado',r'colisionar al vehiculo del asegurado',r'colisionando al vehiculo del asegurado'
                r'impacto al vehiculo del asegurado',r'impacta al vehiculo del asegurado',r'impactar al vehiculo del asegurado',r'impactando al vehiculo del asegurado',
                r'embiste al vehiculo del asegurado',r'embistio al vehiculo del asegurado',r'embestir al vehiculo del asegurado',r'embistiendo al vehiculo del asegurado',
                r'choca al vehiculo del asegurado',r'choco al vehiculo del asegurado',r'chocar al vehiculo del asegurado',r'chocando al vehiculo del asegurado',
                r'toca al vehiculo del asegurado',r'toco al vehiculo del asegurado',r'tocar al vehiculo del asegurado',r'tocando al vehiculo del asegurado',
                r'raspa al vehiculo del asegurado',r'raspo al vehiculo del asegurado',r'raspar al vehiculo del asegurado',r'raspando al vehiculo del asegurado',
                r'raya al vehiculo del asegurado',r'rayo al vehiculo del asegurado',r'rayar al vehiculo del asegurado',r'rayando al vehiculo del asegurado',
                r'aboya al vehiculo del asegurado',r'aboyo al vehiculo del asegurado',r'aboyar al vehiculo del asegurado',r'aboyando al vehiculo del asegurado',
                r'golpea al vehiculo del asegurado',r'golpeo al vehiculo del asegurado',r'golpear al vehiculo del asegurado',r'golpeando al vehiculo del asegurado',
                ),r'tercero colisiona'),
            ((
                r'colisiona al tercero',r'colisiono al tercero',r'colisionar al tercero',r'colisionando al tercero',
                r'impacto al tercero',r'impacta al tercero',r'impactar al tercero',r'impactando al tercero',
                r'embiste al tercero',r'embistio al tercero',r'embestir al tercero',r'embistiendo al tercero',
                r'choca al tercero',r'choco al tercero',r'chocar al tercero',r'chocando al tercero',
                r'toca al tercero',r'toco al tercero',r'tocar al tercero',r'tocando al tercero',
                r'raspa al tercero',r'raspo al tercero',r'raspar al tercero',r'raspando al tercero',
                r'raya al tercero',r'rayo al tercero',r'rayar al tercero',r'rayando al tercero',
                r'aboya al tercero',r'aboyo al tercero',r'aboyar al tercero',r'aboyando al tercero',
                r'golpea al tercero',r'golpeo al tercero',r'golpear al tercero',r'golpeando al tercero',
                r'colisiona a un tercero',r'colisiono a un tercero',r'colisionar a un tercero',r'colisionando a un tercero',
                r'impacto a un tercero',r'impacta a un tercero',r'impactar a un tercero',r'impactando a un tercero',
                r'embiste a un tercero',r'embistio a un tercero',r'embestir a un tercero',r'embistiendo a un tercero',
                r'choca a un tercero',r'choco a un tercero',r'chocar a un tercero',r'chocando a un tercero',
                r'toca a un tercero',r'toco a un tercero',r'tocar a un tercero',r'tocando a un tercero',
                r'raspa a un tercero',r'raspo a un tercero',r'raspar a un tercero',r'raspando a un tercero',
                r'raya a un tercero',r'rayo a un tercero',r'rayar a un tercero',r'rayando a un tercero',
                r'aboya a un tercero',r'aboyo a un tercero',r'aboyar a un tercero',r'aboyando a un tercero',
                r'golpea a un tercero',r'golpeo a un tercero',r'golpear a un tercero',r'golpeando a un tercero',
                r'colisiona a un vehiculo tercero',r'colisiono a un vehiculo tercero',r'colisionar a un vehiculo tercero',r'colisionando a un vehiculo tercero',
                r'impacto a un vehiculo tercero',r'impacta a un vehiculo tercero',r'impactar a un vehiculo tercero',r'impactando a un vehiculo tercero',
                r'embiste a un vehiculo tercero',r'embistio a un vehiculo tercero',r'embestir a un vehiculo tercero',r'embistiendo a un vehiculo tercero',
                r'choca a un vehiculo tercero',r'choco a un vehiculo tercero',r'chocar a un vehiculo tercero',r'chocando a un vehiculo tercero',
                r'toca a un vehiculo tercero',r'toco a un vehiculo tercero',r'tocar a un vehiculo tercero',r'tocando a un vehiculo tercero',
                r'raspa a un vehiculo tercero',r'raspo a un vehiculo tercero',r'raspar a un vehiculo tercero',r'raspando a un vehiculo tercero',
                r'raya a un vehiculo tercero',r'rayo a un vehiculo tercero',r'rayar a un vehiculo tercero',r'rayando a un vehiculo tercero',
                r'aboya a un vehiculo tercero',r'aboyo a un vehiculo tercero',r'aboyar a un vehiculo tercero',r'aboyando a un vehiculo tercero',
                r'golpea a un vehiculo tercero',r'golpeo a un vehiculo tercero',r'golpear a un vehiculo tercero',r'golpeando a un vehiculo tercero',
                r'colisiona a otro vehiculo',r'colisiono a otro vehiculo',r'colisionar a otro vehiculo',r'colisionando a otro vehiculo',
                r'impacto a otro vehiculo',r'impacta a otro vehiculo',r'impactar a otro vehiculo',r'impactando a otro vehiculo',
                r'embiste a otro vehiculo',r'embistio a otro vehiculo',r'embestir a otro vehiculo',r'embistiendo a otro vehiculo',
                r'choca a otro vehiculo',r'choco a otro vehiculo',r'chocar a otro vehiculo',r'chocando a otro vehiculo',
                r'toca a otro vehiculo',r'toco a otro vehiculo',r'tocar a otro vehiculo',r'tocando a otro vehiculo',
                r'raspa a otro vehiculo',r'raspo a otro vehiculo',r'raspar a otro vehiculo',r'raspando a otro vehiculo',
                r'raya a otro vehiculo',r'rayo a otro vehiculo',r'rayar a otro vehiculo',r'rayando a otro vehiculo',
                r'aboya a otro vehiculo',r'aboyo a otro vehiculo',r'aboyar a otro vehiculo',r'aboyando a otro vehiculo',
                r'golpea a otro vehiculo',r'golpeo a otro vehiculo',r'golpear a otro vehiculo',r'golpeando a otro vehiculo',
                r'impacto al otro vehiculo',r'impacta al otro vehiculo',r'impactar al otro vehiculo',r'impactando al otro vehiculo',
                r'embiste al otro vehiculo',r'embistio al otro vehiculo',r'embestir al otro vehiculo',r'embistiendo al otro vehiculo',
                r'choca al otro vehiculo',r'choco al otro vehiculo',r'chocar al otro vehiculo',r'chocando al otro vehiculo',
                r'toca al otro vehiculo',r'toco al otro vehiculo',r'tocar al otro vehiculo',r'tocando al otro vehiculo',
                r'raspa al otro vehiculo',r'raspo al otro vehiculo',r'raspar al otro vehiculo',r'raspando al otro vehiculo',
                r'raya al otro vehiculo',r'rayo al otro vehiculo',r'rayar al otro vehiculo',r'rayando al otro vehiculo',
                r'aboya al otro vehiculo',r'aboyo al otro vehiculo',r'aboyar al otro vehiculo',r'aboyando al otro vehiculo',
                r'golpea al otro vehiculo',r'golpeo al otro vehiculo',r'golpear al otro vehiculo',r'golpeando al otro vehiculo',
                r'colisiona al vehiculo tercero',r'colisiono al vehiculo tercero',r'colisionar al vehiculo tercero',r'colisionando al vehiculo tercero',
                r'impacto al vehiculo tercero',r'impacta al vehiculo tercero',r'impactar al vehiculo tercero',r'impactando al vehiculo tercero',
                r'embiste al vehiculo tercero',r'embistio al vehiculo tercero',r'embestir al vehiculo tercero',r'embistiendo al vehiculo tercero',
                r'choca al vehiculo tercero',r'choco al vehiculo tercero',r'chocar al vehiculo tercero',r'chocando al vehiculo tercero',
                r'toca al vehiculo tercero',r'toco al vehiculo tercero',r'tocar al vehiculo tercero',r'tocando al vehiculo tercero',
                r'raspa al vehiculo tercero',r'raspo al vehiculo tercero',r'raspar al vehiculo tercero',r'raspando al vehiculo tercero',
                r'raya al vehiculo tercero',r'rayo al vehiculo tercero',r'rayar al vehiculo tercero',r'rayando al vehiculo tercero',
                r'aboya al vehiculo tercero',r'aboyo al vehiculo tercero',r'aboyar al vehiculo tercero',r'aboyando al vehiculo tercero',
                r'golpea al vehiculo tercero',r'golpeo al vehiculo tercero',r'golpear al vehiculo tercero',r'golpeando al vehiculo tercero',
                r'colisiona al vehiculo del asegurado',r'colisiono al vehiculo del asegurado',r'colisionar al vehiculo del asegurado',r'colisionando al vehiculo del asegurado',
                r'impacto al vehiculo del tercero',r'impacta al vehiculo del tercero',r'impactar al vehiculo del tercero',r'impactando al vehiculo del tercero',
                r'embiste al vehiculo del tercero',r'embistio al vehiculo del tercero',r'embestir al vehiculo del tercero',r'embistiendo al vehiculo del tercero',
                r'choca al vehiculo del tercero',r'choco al vehiculo del tercero',r'chocar al vehiculo del tercero',r'chocando al vehiculo del tercero',
                r'toca al vehiculo del tercero',r'toco al vehiculo del tercero',r'tocar al vehiculo del tercero',r'tocando al vehiculo del tercero',
                r'raspa al vehiculo del tercero',r'raspo al vehiculo del tercero',r'raspar al vehiculo del tercero',r'raspando al vehiculo del tercero',
                r'raya al vehiculo del tercero',r'rayo al vehiculo del tercero',r'rayar al vehiculo del tercero',r'rayando al vehiculo del tercero',
                r'aboya al vehiculo del tercero',r'aboyo al vehiculo del tercero',r'aboyar al vehiculo del tercero',r'aboyando al vehiculo del tercero',
                r'golpea al vehiculo del tercero',r'golpeo al vehiculo del tercero',r'golpear al vehiculo del tercero',r'golpeando al vehiculo del tercero',
                r'cuando colisiona con un tercero',r'cuando colisiona en un tercero',r'cuando colisiona en tercero',r'cuando colisiona con tercero'
                ),r'asegurado colisiona'),
            ((  
                r'colisionandome',r'impactandome', r'embistiendome',r'chocandome',r'daniandome',
                r'tocandome',r'golpeandome',r'rozandome',r'raspandome',r'aboyandome',r'rayandome',
                r'tocarme',r'embestirme',r'chocarme',r'impactarme',r'daniarme',r'rayarme',
                r'golpearme',r'rozarme',r'rasparme',r'colisionarme',r'aboyarme',
                r'me colisiona',r'me colisiono',r'me embiste',r'me embistio',r'me impacta',r'me impacto',
                r'me toca',r'me toco',r'me choca',r'me choco',r'me raspa',r'me raspo',r'me dania',r'me danio',
                r'me aboya',r'me aboyo',r'me golpea',r'me golpeo',r'me raya',r'me rayo',r'me roza',r'me rozo',
                r'fui colisionado',r'fui impactado',r'fui embestido',r'fui chocado',r'fui tocado',
                r'fui raspado',r'fui daniado',r'fui rozado',r'fui rayado',r'fui aboyado',r'fui golpeado',
                r'soy colisionado',r'soy impactado',r'soy embestido',r'soy chocado',r'soy tocado',
                r'soy raspado',r'soy daniado',r'soy rozado',r'soy rayado',r'soy aboyado',r'soy golpeado',
                r'es colisionado',r'es impactado',r'es embestido',r'es chocado',r'es tocado',
                r'es raspado',r'es daniado',r'es rozado',r'es rayado',r'es aboyado',r'es golpeado',
                r'tercero toca',r'tercero toco',r'tercero golpea',r'tercero golpeo',
                r'tercero impacta',r'tercero golpeo',r'tercero embiste',r'tercero embistio',
                r'tercero raspa',r'tercero raspo',r'tercero choca',r'tercero choco',r'tercero danio',
                r'tercero dania',r'tercero colisiono',r'tercero impacto',r'tercero toca',r'tercero toco',
                r'tercero lo toca',r'tercero lo toco',r'tercero lo golpea',r'tercero lo golpeo',
                r'tercero lo impacta',r'tercero lo golpeo',r'tercero lo embiste',r'tercero lo embistio',
                r'tercero lo raspa',r'tercero lo raspo',r'tercero lo choca',r'tercero lo choco',r'tercero lo danio',
                r'tercero lo dania',r'tercero lo colisiono',r'tercero lo impacto',r'tercero lo toca',r'tercero lo toco',
                r'nos colisiona',r'nos colisiono'
            ), r'tercero colisiona'),
            ((  
                r'colisionandolo',r'impactandolo', r'embistiendolo',r'chocandolo',r'daniandolo',
                r'tocandolo',r'golpeandolo',r'rozandolo',r'raspandolo',r'aboyandolo',r'rayandolo',
                r'tocarlo',r'embestirlo',r'chocarlo',r'impactarlo',r'daniarlo',r'rayarlo',
                r'golpearlo',r'rozarlo',r'rasparlo',r'colisionarlo',r'aboyarlo',
                r'colisionandole',r'impactandole', r'embistiendole',r'chocandole',r'daniandole',
                r'tocandole',r'golpeandole',r'rozandole',r'raspandole',r'aboyandole',r'rayandole',
                r'tocarle',r'embestirle',r'chocarle',r'impactarle',r'daniarle',r'rayarle',
                r'golpearle',r'rozarle',r'rasparle',r'colisionarle',r'aboyarle',
                r'lo colisiono',r'lo colisione',r'lo impacto',r'lo impacte',r'lo embisto',r'lo embesti',
                r'lo choco',r'lo choque',r'lo danio',r'lo danie',r'lo rayo',r'lo raye',r'lo aboyo',r'lo aboye',
                r'lo rozo',r'lo roce',r'lo golpeo',r'lo golpee',r'lo raspo',r'lo raspe',
                r'le colisiono',r'le colisione',r'le impacto',r'le impacte',r'le embisto',r'le embesti',
                r'le choco',r'le choque',r'le danio',r'le danie',r'le rayo',r'le raye',r'le aboyo',r'le aboye',
                r'le rozo',r'le roce',r'le golpeo',r'le golpee',r'le raspo',r'le raspe',
                r'asegurado toca',r'asegurado toco',r'asegurado golpea',r'asegurado golpeo',
                r'asegurado impacta',r'asegurado golpeo',r'asegurado embiste',r'asegurado embistio',
                r'asegurado raspa',r'asegurado raspo',r'asegurado choca',r'asegurado choco',r'asegurado danio',
                r'asegurado dania',r'asegurado colisiono',r'asegurado impacto',r'asegurado toca',r'asegurado toco',
                r'asegurado lo toca',r'asegurado lo toco',r'asegurado lo golpea',r'asegurado lo golpeo',
                r'asegurado lo impacta',r'asegurado lo golpeo',r'asegurado lo embiste',r'asegurado lo embistio',
                r'asegurado lo raspa',r'asegurado lo raspo',r'asegurado lo choca',r'asegurado lo choco',r'asegurado lo danio',
                r'asegurado lo dania',r'asegurado lo colisiono',r'asegurado lo impacto',r'asegurado lo toca',r'asegurado lo toco',
                r'no logro evitar colisionar',r'no logro evitar impactar',r'no logro evitar chocar',
                r'no logro evitar tocar',r'no logro evitar embestir',r'no logro evitar daniar',
                r'no logro evitar aboyar',r'no logro evitar golpear',r'no logro evitar rayar',
                r'no logro evitar rozar',r'no logro evitar raspar',
                r'cuando colisiono',r'cuando colisione',r'asegurado colisione',r'asegurado lo colisiona',
                r'e impacte',r'e impacto',r'y colisione',r'y colisiono',r'e colisiono',r'e colisione',r'e colisiona'
                r'y embisto',r'y embesti',r'y aboye',r'y roce',
                r'y golpee',r'y raye',r'y raspe', r'y toque',r'y choque',
                r'y le colisione',r'y le embisto',r'y le embesti',r'y le aboye',r'y le roce',
                r'y le golpee',r'y le raye',r'y le raspe', r'y le toque',r'y le choque',
                r'y lo colisione',r'y lo embisto',r'y lo embesti',r'y lo aboye',r'y lo roce',
                r'y lo golpee',r'y lo raye',r'y lo raspe', r'y lo toque',r'y lo choque',
                r'colisione'
            ), r'asegurado colisiona')
    ]

def partsDic():
    return [
            ((  r'paragolpe delantera',r'guardabarro delantera',r'parte frontal',r'espejo delantera',
                r'parte de adelante', r'parte delante', r'rueda delantera', r'rueda de adelante',r'puerta delantera',r'optica del lado delantera',
                r'optica delantera',r'frente delantera',r'su frente'
                ),  r'parte delantera'),
            ((  r'paragolpe trasera',r'guardabarro trasera',r'parte de atras',r'parte atras',
                r'rueda trasera',r'rueda de atras',r'puerta trasera',r'optica del lado trasera',
                r'optica trasera',r'a trasera'
                ),  r'parte trasera'),
            ((  r'rueda derecha',r'rueda del lado derecha',r'rueda lado derecha', 
                r'puerta del lado derecha', r'puerta derecha', r'puerta lado derecha',
                r'paragolpe derecha',r'paragolpe lado derecha',r'guardabarro derecha',r'guardabarro del lado derecha',
                r'guardabarro lado derecha',r'espejo derecha',r'espejo lado derecha',r'espejos del lado derecha',
                r'espejo retrovisor derecha',r'espejo retrovisor del lado derecha',r'retrovisor derecha',r'retrovisor del lado derecha',
                r'optica del lado derecha',r'optica derecha',r'parte del lado derecha',r'parte lado derecha'
            ),r'parte derecha'),
            ((  r'rueda izquierda', r'puerta del lado izquierda', r'puerta izquierda', r'paragolpe izquierda',
                r'paragolpes izquierda',r'espejo izquierda',r'espejo del lado izquierda',r'guardabarro izquierda',
                r'espejo retrovisor izquierda',r'espejo retrovisor del lado izquierda',r'retrovisor izquierda',r'retrovisor del lado izquierda',
                r'guardabarros izquierda',r'guardabarros del lado izquierda',r'parte del lado izquierda',r'guardabarro del lado izquierda',
                r'paragolpe del lado izquierda',r'paragolpes del lado izquierda',r'paragolpe lado izquierda',r'paragolpes lado izquierda',
                r'optica del lado izquierda',r'optica izquierda',r'parte lado izquierda',r'parte del lado izquierda',
            ),r'parte izquierda'),
            ((  r'derecha delantera',r'delantera del lado derecha',r'delantera lado derecha',
                r'derecha del lado delantera'),r'delantera derecha'),
            ((  r'izquierda delantera',r'delantera del lado izquierda',r'delantera lado izquierda',
                r'izquierda del lado delantera'),r'delantera izquierda'),
            ((  r'derecha trasera',r'trasera del lado derecha',r'trasera lado derecha',
                r'derecha del lado trasera'),r'trasera derecha'),
            ((  r'izquierda trasera',r'trasera lado izquierda', r'trasera del lado izquierda',
                r'izquierda del lado trasera'),r'trasera izquierda'),
            ((r'mi parte',), r'asegurado parte'),
            ((r'mi delantera',r'mi parte delantera', r'mi frente'), r'asegurado parte delantera'),
            ((r'mi trasera',), r'asegurado parte trasera'),
    ]

def orderParts():
    return [
        ((  r'parte delantera del vehiculo tercero',r'parte delantera de vehiculo tercero',
            r'parte delantera del tercero',r'parte delantera de tercero',
            r'parte delantera del vehiculo de tercero',r'parte delantera de vehiculo de tercero',
            r'parte delantera del vehiculo del tercero',r'parte delantera de vehiculo del tercero',
            r'parte delantera del otro vehiculo', r'parte delantera de otro vehiculo',
            r'parte delantera de un tercero', r'parte delantera de un vehiculo', r'parte delantera en tercero',
            ),r'tercero parte delantera'),
        ((  r'parte delantera derecha del vehiculo tercero',r'parte delantera derecha de vehiculo tercero',
            r'parte delantera derecha del tercero',r'parte delantera derecha de tercero',
            r'parte delantera derecha del vehiculo de tercero',r'parte delantera derecha de vehiculo de tercero',
            r'parte delantera derecha del vehiculo del tercero',r'parte delantera derecha de vehiculo del tercero',
            r'parte delantera derecha del otro vehiculo', r'parte delantera derecha de otro vehiculo',
            r'parte delantera derecha de un tercero', r'parte delantera derecha de un vehiculo', r'parte delantera derecha en tercero',
        ),r'tercero parte delantera derecha'),
        ((  r'parte delantera izquierda del vehiculo tercero',r'parte delantera izquierda de vehiculo tercero',
            r'parte delantera izquierda del tercero',r'parte delantera izquierda de tercero',
            r'parte delantera izquierda del vehiculo de tercero',r'parte delantera izquierda de vehiculo de tercero',
            r'parte delantera izquierda del vehiculo del tercero',r'parte delantera izquierda de vehiculo del tercero',
            r'parte delantera izquierda del otro vehiculo', r'parte delantera izquierda de otro vehiculo',
            r'parte delantera izquierda de un tercero', r'parte delantera izquierda de un vehiculo', r'parte delantera izquierda en tercero',
        ),r'tercero parte delantera izquierda'),
        ((  r'parte delantera trasera del vehiculo tercero',r'parte trasera de vehiculo tercero',
            r'parte trasera del tercero',r'parte trasera de tercero',
            r'parte trasera del vehiculo de tercero',r'parte trasera de vehiculo de tercero',
            r'parte trasera del vehiculo del tercero',r'parte trasera de vehiculo del tercero',
            r'parte trasera del otro vehiculo', r'parte trasera de otro vehiculo',
            r'parte trasera de un tercero', r'parte trasera de un vehiculo', r'parte trasera en tercero',
        ),r'tercero parte trasera'),
        ((  r'parte trasera derecha del vehiculo tercero',r'parte trasera derecha de vehiculo tercero',
            r'parte trasera derecha del tercero',r'parte trasera derecha de tercero',
            r'parte trasera derecha del vehiculo de tercero',r'parte trasera derecha de vehiculo de tercero',
            r'parte trasera derecha del vehiculo del tercero',r'parte trasera derecha de vehiculo del tercero',
            r'parte trasera derecha del otro vehiculo', r'parte trasera derecha de otro vehiculo',
            r'parte trasera derecha de un tercero', r'parte trasera derecha de un vehiculo', r'parte trasera derecha en tercero',
        ),r'tercero parte trasera derecha'),
        ((  r'parte trasera izquierda del vehiculo tercero',r'parte trasera izquierda de vehiculo tercero',
            r'parte trasera izquierda del tercero',r'parte trasera izquierda de tercero',
            r'parte trasera izquierda del vehiculo de tercero',r'parte trasera izquierda de vehiculo de tercero',
            r'parte trasera izquierda del vehiculo del tercero',r'parte trasera izquierda de vehiculo del tercero',
            r'parte trasera izquierda del otro vehiculo', r'parte trasera izquierda de otro vehiculo',
            r'parte trasera izquierda de un tercero', r'parte trasera izquierda de un vehiculo', r'parte trasera izquierda en tercero',
        ),r'tercero parte trasera izquierda'),
        ((  r'parte derecha del vehiculo tercero',r'parte derecha de vehiculo tercero',
            r'parte derecha del tercero',r'parte derecha de tercero',
            r'parte derecha del vehiculo de tercero',r'parte derecha de vehiculo de tercero',
            r'parte derecha del vehiculo del tercero',r'parte derecha de vehiculo del tercero',
            r'parte derecha del otro vehiculo', r'parte derecha de otro vehiculo',
            r'parte derecha de un tercero', r'parte derecha de un vehiculo', r'parte derecha en tercero',
        ),r'tercero parte derecha'),
        ((  r'parte izquierda del vehiculo tercero',r'parte izquierda de vehiculo tercero',
            r'parte izquierda del tercero',r'parte izquierda de tercero',
            r'parte izquierda del vehiculo de tercero',r'parte izquierda de vehiculo de tercero',
            r'parte izquierda del vehiculo del tercero',r'parte izquierda de vehiculo del tercero',
            r'parte izquierda del otro vehiculo', r'parte izquierda de otro vehiculo',
            r'parte izquierda de un tercero', r'parte izquierda de un vehiculo', r'parte izquierda en tercero',
        ),r'tercero parte izquierda'),
        ((  r'parte delantera del vehiculo asegurado',r'parte delantera de vehiculo asegurado',
            r'parte delantera del asegurado',r'parte delantera de asegurado',
            r'parte delantera del vehiculo de asegurado',r'parte delantera de vehiculo de asegurado',
            r'parte delantera del vehiculo del asegurado',r'parte delantera de vehiculo del asegurado',
            r'parte delantera de mi vehiculo',r'parte delantera de unidad asegurado'
            ),r'asegurado parte delantera'),
        ((  r'parte delantera derecha del vehiculo asegurado',r'parte delantera derecha de vehiculo asegurado',
            r'parte delantera derecha del asegurado',r'parte delantera derecha de asegurado',
            r'parte delantera derecha del vehiculo de asegurado',r'parte delantera derecha de vehiculo de asegurado',
            r'parte delantera derecha del vehiculo del asegurado',r'parte delantera derecha de vehiculo del asegurado',
            r'parte delantera derecha de mi vehiculo',r'parte delantera derecha de unidad asegurado'
        ),r'asegurado parte delantera derecha'),
        ((  r'parte delantera izquierda del vehiculo asegurado',r'parte delantera izquierda de vehiculo asegurado',
            r'parte delantera izquierda del asegurado',r'parte delantera izquierda de asegurado',
            r'parte delantera izquierda del vehiculo de asegurado',r'parte delantera izquierda de vehiculo de asegurado',
            r'parte delantera izquierda del vehiculo del asegurado',r'parte delantera izquierda de vehiculo del asegurado',
            r'parte delantera izquierda de mi vehiculo',r'parte delantera izquierda de unidad asegurado'
        ),r'asegurado parte delantera izquierda'),
        ((  r'parte trasera del vehiculo asegurado',r'parte trasera de vehiculo asegurado',
            r'parte trasera del asegurado',r'parte trasera de asegurado',
            r'parte trasera del vehiculo de asegurado',r'parte trasera de vehiculo de asegurado',
            r'parte trasera del vehiculo del asegurado',r'parte trasera de vehiculo del asegurado',
            r'parte trasera de mi vehiculo',r'parte trasera de unidad asegurado'
        ),r'asegurado parte trasera'),
        ((  r'parte trasera derecha del vehiculo asegurado',r'parte trasera derecha de vehiculo asegurado',
            r'parte trasera derecha del asegurado',r'parte trasera derecha de asegurado',
            r'parte trasera derecha del vehiculo de asegurado',r'parte trasera derecha de vehiculo de asegurado',
            r'parte trasera derecha del vehiculo del asegurado',r'parte trasera derecha de vehiculo del asegurado',
            r'parte trasera derecha de mi vehiculo',r'parte trasera derecha de unidad asegurado'
        ),r'asegurado parte trasera derecha'),
        ((  r'parte trasera izquierda del vehiculo asegurado',r'parte trasera izquierda de vehiculo asegurado',
            r'parte trasera izquierda del asegurado',r'parte trasera izquierda de asegurado',
            r'parte trasera izquierda del vehiculo de asegurado',r'parte trasera izquierda de vehiculo de asegurado',
            r'parte trasera izquierda del vehiculo del asegurado',r'parte trasera izquierda de vehiculo del asegurado',
            r'parte trasera izquierda de mi vehiculo',r'parte trasera izquierda de unidad asegurado'
        ),r'asegurado parte trasera izquierda'),
        ((  r'parte derecha del vehiculo asegurado',r'parte derecha de vehiculo asegurado',
            r'parte derecha del asegurado',r'parte derecha de asegurado',
            r'parte derecha del vehiculo de asegurado',r'parte derecha de vehiculo de asegurado',
            r'parte derecha del vehiculo del asegurado',r'parte derecha de vehiculo del asegurado',
            r'parte derecha de mi vehiculo',r'parte derecha de unidad asegurado'
        ),r'asegurado parte derecha'),
        ((  r'parte izquierda del vehiculo asegurado',r'parte izquierda de vehiculo asegurado',
            r'parte izquierda del asegurado',r'parte izquierda de asegurado',
            r'parte izquierda del vehiculo de asegurado',r'parte izquierda de vehiculo de asegurado',
            r'parte izquierda del vehiculo del asegurado',r'parte izquierda de vehiculo del asegurado',
            r'parte izquierda de mi vehiculo',r'parte izquierda de unidad asegurado'
        ),r'asegurado parte izquierda'),
    ]

def changebadParts():
    return [
        ((
            r'asegurado colisiona con su parte',r'asegurado colisiona con parte',r'asegurado colisiona su parte'
        ), r'asegurado colisiona con asegurado parte'),
        ((
            r'asegurado colisiona en su parte',r'asegurado colisiona en parte'
        ), r'asegurado colisiona en tercero parte'),
        ((
            r'tercero colisiona en su parte',r'tercero colisiona en parte'
        ), r'tercero colisiona en asegurado parte'),
        ((
            r'tercero colisiona con su parte',r'tercero colisiona con parte'
        ), r'tercero colisiona con tercero parte'),
    ]