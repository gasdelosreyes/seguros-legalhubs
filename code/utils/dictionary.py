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

def verbsDic():
    return [
        r'colisionandome',r'colisionandole',r'colisionandolo',r'colisionando',r'colisionado',r'colisionar',r'colisiono',r'colisione',r'colisiona',r'colision',
        r'impactandome',r'impactandole',r'impactandolo',r'impactando',r'impactado',r'impactarme',r'impactarlo',r'impactarle',r'impactar',r'impactado',r'impacta',r'impacto',r'impacte',
        r'embestido',r'embestirlo',r'embestirme',r'embestirle',r'embestir',r'embesti',
        r'embistiendome',r'embistiendole',r'embistiendolo',r'embistiendo',r'embistio',r'embisto',r'embiste',
        r'chocandome',r'chocandole',r'chocandolo',r'chocarlo',r'chocarle',r'chocarme',r'chocando'r'chocado',r'chocar',r'choque',r'choco',r'choca',
        r'tocandome',r'tocandole',r'tocandolo',r'tocando',r'tocarle',r'tocarme',r'tocado',r'tocarlo',r'tocar',r'toco',r'toca',r'toque',
        r'raspandome',r'raspandole',r'raspandolo',r'rasparme',r'rasparle',r'raspar',r'raspe',r'raspo',r'raspa',
        r'daniandome',r'daniandole',r'daniandolo',r'daniando',r'daniarme',r'daniarlo',r'daniarle',r'daniar',r'dania',r'danio',r'danie',
        r'rozandome',r'rozandole',r'rozandolo',r'rozando',r'rozarme',r'rozarlo',r'rozarle',r'rozar',r'rozo',r'roza',r'roce',
        r'rayandome',r'rayandole',r'rayandolo',r'rayando',r'rayarme',r'rayarlo',r'rayarle',r'rayar',r'rayo',r'raya',r'raye',
        r'aboyarme',r'aboyandole',r'aboyandolo',r'aboyando',r'aboyarme',r'aboyarlo',r'aboyarle',r'aboyar',r'aboyo',r'aboya',r'aboye'
    ]

def carDic():
    return [
            ((  r'vt',r'vtv',r'vhs',r'vehi'r'vhlo',r'vh',r'vhc',r'vehculo',r'vhls',
                r'vhl',r'camioneta',r'camion',r'camien',r'moto'
                r'auto',r'automovil',r'rodado',r'autos',r'veh',r'coche',r'vehyculo'
            ), r'vehiculo'),
            ((  r'aut',r'autop',
            ),r'autopista'),
            ((  r'mi vehiculo', r'vehiculo asegurado',
            ), r'vehiculo del asegurado'),
            ((  r'vehiculo de un tercero',r'vehiculo e un tercero',r'vehiculo de tercero',r'vehiculo tercero',
                r'otro vehiculo', r'un vehiculo',r'un tercero'
            ),r'vehiculo del tercero'),
            ((  r'circ',r'transitaba'
            ),r'circulaba'),
            ((  r'del acompanante',r'del acompaniante'
            ),r'derecha'),
            ((  r'no puedo evitar',r'no pudo evitar',r'no logra evitar',r'no pude evitar',r'no logre evitar',
                r'no puede evitar',r'no logran evitar',r'no pueden evitar',r'no pudiendo evitar',
                r'no logrando evitar',
            ),r'no logro evitar'),
            ((  r'interseccien',r'interseccin',
            ),  r'interseccion'),
            ((r'eptica',),r'optica'),
            ((r'guardabarros',),r'guardabarro'),
            ((r'paragolpes',),r'paragolpe'),
            ((r'del conductor', ),r'izquierda'),
            ((r'a tras',), r'atras'),
            ((r'a delante',), r'adelante'),
            ((r'garage', r'gge', r'cochera'), r'garaje'),
            ((r'av',r'avda'), r'avenida'),
            ((r'lateral', r'costado', r'sector',r'lat'), r'parte'),
            ((r'ero', r'taxi', r'taxista',r'terc',r'chofer',r'er',r'vecino'), r'tercero'),
            ((r'asegurada', r'aseg'), r'asegurado'),
            ((r'izquierdo', r'izq'),r'izquierda'),
            ((r'derecho', r'der', r'dere',r'dcha'),r'derecha'),
            ((r'trasero',),r'trasera'),
            ((r'delantero', r'trompa',r'posterior'),r'delantera'),
    ]

def crashDic():
    return [
            ((  r'colisionandome',r'impactandome', r'embistiendome',r'chocandome',
                r'tocandome',r'golpeandome',r'embestirme',r'chocarme',r'tocarme',
                r'me impacto',r'me impacta',r'me colisiona',r'me embistio',
                r'me embiste',r'me choco',r'me choca',r'me raspa',r'me raspo', r'me toca',
                r'me toco', r'es colisionado',r'fui colisionado',r'fue colisionado',r'es embestido',
                r'fui embestido',r'fue embestido',r'es chocado',r'fue chocado',r'fui chocado',r'es impactado',
                r'soy colisionado',r'soy impactado',r'soy embestido', r'soy golpeado',
                r'soy daniado',r'daniandome',r'me dania',r'me roza',r'me rozo',r'me danio',
                r'soy rozado',r'rozandome',r'es daniado',r'fue daniado',r'fui daniado',
                r'es rozado',r'fui rozado',r'fue rozado',r'choca al asegurado',r'colisiona al asegurado',
            ), r'tercero colisiona'),
            ((
                r'lo choque', r'lo toque',r'lo toco',r'lo golpee',r'lo colisione',r'lo impacto',
                r'lo golpeo', r'lo colisiono', r'lo embesti', r'lo embisto',r'lo raspe',r'lo raspo',
                r'lo impacte',r'le choque',r'le choco',
                r'y embisto',r'y embiste', r'y embesti', r'y colisione',r'y colisiono',
                r'y toco', r'y toque',r'y choco', r'y choque',r'y raspo', r'y raspe',
                r'e impacto',r'e impacte',r'le raspo',r'le raspe',r'le embisto',r'le embesti',
                r'le colisiono',r'le colisione',r'le impacto',r'daniandole',r'daniandolo',
                r'lo rozo',r'le rozo',r'lo danio',r'lo dania',r'lo danie',r'le danie',
                r'lo roce',r'le roce',r'rozandole',r'y roce',r'y rozo',r'y danio',r'y danie',
                r'asegurado toca',r'asegurado toco',r'asegurado golpea',r'asegurado golpeo',
                r'asegurado impacta',r'asegurado golpeo',r'asegurado embiste',r'asegurado embistio',
                r'asegurado raspa',r'asegurado raspo',r'asegurado choca',r'asegurado choco',r'asegurado danio',
                r'asegurado dania',r'asegurado colisiono',r'asegurado impacto',r'asegurado toca',r'asegurado toco',
                r'colisiono con',r'toco con',r'impacto con',r'embistiendolo con',r'impactando con',r'colisione con',
                r'no logro evitar colisionar',
            ), r'asegurado colisiona')
    ]

def partsDic():
    return [
            ((  r'paragolpe delantera',r'paragolpes delantera',r'guardabarro delantera',
                r'guardabarros delantera', r'parte frontal',r'espejo delantera', r'parte de adelante',
                r'parte delante', r'rueda delantera', r'rueda de adelante',r'puerta delantera',r'optica del lado delantera',
                r'optica delantera',r'frente delantera'
            ),r'parte delantera'),
            ((  r'paragolpe trasera',r'paragolpes trasera',r'guardabarro trasera',r'guardabarros trasera',
                r'parte de atras', r'parte atras',r'rueda trasera',r'rueda de atras',r'puerta trasera',r'optica del lado trasera',
                r'optica trasera',
            ),r'parte trasera'),
            ((  r'rueda derecha',r'rueda del lado derecha',r'rueda lado derecha', 
                r'puerta del lado derecha', r'puerta derecha', r'puerta lado derecha',
                r'paragolpes derecha',r'paragolpe derecha',r'paragolpes lado derecha',r'paragolpe lado derecha',
                r'guardabarro derecha',r'guardabarros derecha',r'guardabarros del lado derecha',r'guardabarro del lado derecha',
                r'guardabarro lado derecha',r'guardabarros lado derecha',r'espejo derecha',r'espejo lado derecha',r'espejos del lado derecha',
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
            ((r'mi delantera', r'mi frente'), r'asegurado parte delantera'),
            ((r'mi trasera',), r'asegurado parte trasera')
    ]