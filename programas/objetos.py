import pandas as pd
import re

# from stats import *


def printDic(dic):
    for i in range(len(dic)):
        key = list(dic.keys())
        value = list(dic.values())
        print(key[i], value[i])


def is_complete(caso):
    if len(caso.get_impac_position()) and caso.get_ubicacion_vial() != 'desconocido' and caso.get_quien():
        return True
    return False


class TablaCasos:
    """Tabla con todos los casos"""

    def __init__(self, TableName):
        self.TableName = TableName
        self.impac_position = ''
        self.casos = []

    def set_caso(self, caso, update=False):  # revisar el update porque no funciona bien
        exist = False
        for i in self.casos:
            if caso.get_idxDesc() == i.get_idxDesc():
                exist = True
        if not exist:
            self.casos.append(caso)

    def get_casos(self):
        return self.casos

    def __str__(self):
        print(self.TableName)
        for caso in self.casos:
            print(caso.get_idxDesc(), caso.get_descripcion(), caso.get_responsabilidad())

    def to_csv(self):
        df = pd.DataFrame()
        df['idxDesc'] = pd.Series([caso.get_idxDesc() for caso in self.casos])
        # df['descripcion_original'] = pd.Series([caso.get_descripcion_original() for caso in self.casos])
        df['descripcion'] = pd.Series([caso.get_descripcion() for caso in self.casos])
        df['ubicacion_vial'] = pd.Series([caso.get_ubicacion_vial() for caso in self.casos])
        df['movimiento'] = pd.Series([caso.get_movement() for caso in self.casos])
        df['responsabilidad'] = pd.Series([caso.get_responsabilidad() for caso in self.casos])
        df['impac_position'] = pd.Series([caso.get_impac_position() for caso in self.casos])
        df['quien'] = pd.Series([caso.get_quien() for caso in self.casos])
        df['responsabilidad_predic'] = pd.Series([caso.get_responsabilidad_predic() for caso in self.casos])

        df.to_csv('tmp/' + self.TableName + '.csv', index=False)

    def update(self):
        """
        Actualiza las estadísticas que se deseen conocer
        """
        self.casos_totales = len(self.casos)
        self.responsabilidad_comprometido = len([i.responsabilidad for i in self.casos if i.get_responsabilidad() == 'COMPROMETIDA'])
        self.responsabilidad_comprometido += len([i for i in self.casos if i.get_responsabilidad() == 'DISCUTIDA'])
        self.responsabilidad_no_comprometido = self.casos_totales - self.responsabilidad_comprometido
        self.movimiento = len([i for i in self.casos if i.get_movement() == 'si'])
        self.no_movimiento = self.casos_totales - self.movimiento
        self.delantera = len([i for i in self.casos if i.get_impac_position() == 'delantera'])
        self.delantera += len([i for i in self.casos if i.get_impac_position() == 'delantera izquierda'])
        self.delantera += len([i for i in self.casos if i.get_impac_position() == 'delantera derecha'])
        self.trasera = len([i for i in self.casos if i.get_impac_position() == 'trasera'])
        self.trasera += len([i for i in self.casos if i.get_impac_position() == 'trasera izquierda'])
        self.trasera += len([i for i in self.casos if i.get_impac_position() == 'trasera derecha'])
        self.ubicaciones_viales = set([i.get_ubicacion_vial() for i in self.casos])
        self.asegurado = len([i for i in self.casos if i.get_quien() == 'asegurado'])
        self.tercero = len([i for i in self.casos if i.get_quien() == 'tercero'])

    def status(self):
        stat = {'casos totales': self.casos_totales, 'comprometidos': self.responsabilidad_comprometido, 'no_comprometidos': self.responsabilidad_no_comprometido, 'en_movimiento': self.movimiento, 'no_movimiento': self.no_movimiento, 'delantera': self.delantera, 'trasera': self.trasera, 'asegurado_colisiona': self.asegurado, 'tercero_colisiona': self.tercero}  # , 'lugares posibles': self.ubicaciones_viales}
        printDic(stat)

    def plot_movimiento(self):
        plt.clf()
        """ 
        :returns: devuelve un grafico de torta de 
        la proporción de asegurado en movimiento 
        """
        label, sizes, colors = set_pie([caso.get_movement() for caso in self.casos])
        mov = plt.subplot(111)
        mov.axis('off')
        mov.pie(sizes, labels=label, colors=colors, autopct='%1.1f%%', shadow=False, startangle=140)
        plt.savefig(self.TableName + '_movimiento.png')

    def plot_casos_completos(self):
        plt.clf()
        """
        :function: un caso completo es aquel totalmente etiquetado
        :returns: devuelve un grafico de torta de 
        la cantidad de casos totalmente etiquetados
        """
        aux = []
        for caso in self.casos:
            if len(caso.get_impac_position()) and caso.get_ubicacion_vial() != 'desconocido' and caso.get_quien():
                aux.append('completo')
            elif not len(caso.get_impac_position()) and caso.get_ubicacion_vial() == 'desconocido' and not caso.get_quien():
                aux.append('vacio')
            else:
                aux.append('parcial')
        label, sizes, colors = set_pie(aux)
        pie = plt.subplot(111)
        pie.axis('off')
        pie.pie(sizes, labels=label, colors=colors, autopct='%1.1f%%', shadow=False, startangle=140)
        plt.savefig(self.TableName + '_completitud.png')

    def plot_ubicacion_vial(self):
        plt.clf()
        """
        :function: las ubicaiones viales serán garaje, estacionamiento, avenida,calle, interseccion,
        autopista, esquina, rotonda, carril, cruce, peaje, tunel, semaforo
        :returns: retorna la proporción de cada ubicación vial

        """
        location = ['calle', r'garaje', r'roton\w*', 'autopista', 'avenida', 'cruce', r'esquina\w*', r'estacionami\w*', 'carril', 'ruta', r'semaforo\w*', r'intersec.?', 'desconocido']
        explode = [0.1 for i in location]
        aux = []
        for loc in location:
            for caso in self.casos:
                if re.search(loc, caso.get_ubicacion_vial()):
                    aux.append(loc)
                    labels, sizes, colors = set_pie(aux)
                    pie = plt.subplot(111)
                    pie.axis('off')
                    pie.pie(sizes, labels=labels, explode=explode, colors=colors, autopct='%1.1f%%', shadow=False, startangle=140)
                    plt.savefig(self.TableName + '_ubicacion_vial.png')

    def plot_posicion_impact(self):
        plt.clf()
        """
        :function: las ubicaiones viales serán garaje, estacionamiento, avenida,calle, interseccion,
        autopista, esquina, rotonda, carril, cruce, peaje, tunel, semaforo
        :returns: retorna la proporción de cada ubicación vial

        """
        aux = []
        for caso in self.casos:
            if caso.get_impac_position() == 'delantera':
                aux.append('delantera')
            elif caso.get_impac_position() == 'trasera':
                aux.append('trasera')
            else:
                aux.append('desconocido')

        labels, sizes, colors = set_pie(aux)
        pie = plt.subplot(111)
        pie.axis('off')
        pie.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=False, startangle=140)
        plt.savefig(self.TableName + '_impact_posicion.png')

    def plot_quien(self):
        plt.clf()
        """
        :function: las ubicaiones viales serán garaje, estacionamiento, avenida,calle, interseccion,
        autopista, esquina, rotonda, carril, cruce, peaje, tunel, semaforo
        :returns: retorna la proporción de cada ubicación vial

        """
        aux = []
        for caso in self.casos:
            if caso.get_quien() == 'asegurado':
                aux.append('asegurado')
            elif caso.get_quien() == 'tercero':
                aux.append('tercero')
            else:
                aux.append('desconocido')

        labels, sizes, colors = set_pie(aux)
        pie = plt.subplot(111)
        pie.axis('off')
        pie.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=False, startangle=140)
        plt.savefig(self.TableName + '_quien.png')

    def cross_plot(self):
        plt.clf()
        plt.title('Clasifiación completa')
        aux, aux1, aux2 = [], [], []
        for caso in self.casos:
            if is_complete(caso):
                aux.append(caso.get_responsabilidad())
            elif not len(caso.get_impac_position()) and caso.get_ubicacion_vial() == 'desconocido' and not caso.get_quien():
                aux1.append(caso.get_responsabilidad())
            else:
                aux2.append(caso.get_responsabilidad())

        label, sizes, colors = set_pie(aux)
        pie = plt.subplot(111)
        pie.axis('off')
        pie.pie(sizes, labels=label, colors=colors, autopct='%1.1f%%', shadow=False, startangle=140)
        plt.savefig(self.TableName + '_responsabilidad_completo.png')

        plt.clf()
        plt.title('Clasifiación vacía')

        label, sizes, colors = set_pie(aux1)
        pie = plt.subplot(111)
        pie.axis('off')
        pie.pie(sizes, labels=label, colors=colors, autopct='%1.1f%%', shadow=False, startangle=140)
        plt.savefig(self.TableName + '_responsabilidad_vacio.png')

        plt.clf()
        plt.title('Clasifiación parcial')

        label, sizes, colors = set_pie(aux2)
        pie = plt.subplot(111)
        pie.axis('off')
        pie.pie(sizes, labels=label, colors=colors, autopct='%1.1f%%', shadow=False, startangle=140)
        plt.savefig(self.TableName + '_responsabilidad_parcial.png')


class Caso:
    """
    Atributos del caso en vistas de una BD
    """

    def __init__(self):
        self.impac_position = ''
        self.responsabilidad_predic = 0

    def set_descripcion(self, descr):
        self.descripcion = descr

    def get_descripcion(self):
        return self.descripcion

    def set_idxDesc(self, idx):
        self.idxDesc = idx

    def get_idxDesc(self):
        return self.idxDesc

    def set_responsabilidad(self, resp):
        self.responsabilidad = resp

    def get_responsabilidad(self):
        return self.responsabilidad

    def set_responsabilidad_predic(self, pred):
        """ 
        asigna la responsabilidad predicha
        """
        self.responsabilidad_predic += pred

    def get_responsabilidad_predic(self):
        return self.responsabilidad_predic
    # def set_descripcion_original(self, original):
    #     self.descripcion_original = original

    # def get_descripcion_original(self):
    #     return self.descripcion_original

    def get_ubicacion_vial(self):
        location = ['calle', r'garaje', r'roton\w*', 'autopista', 'avenida', 'cruce', 'cruze', r'esquina\w*', r'estacionami\w*', 'carril', 'ruta', r'semaforo\w*', r'intersec.?', 'tunel', 'peaje']
        aux = []
        for loc in location:
            st = re.search(loc, self.descripcion)
            if st:
                aux.append(st.group())
        if aux:
            self.ubicacion_vial = ' '.join(set(aux))
            return self.ubicacion_vial
        else:
            self.ubicacion_vial = 'desconocido'
            return self.ubicacion_vial

    def get_impac_position(self):
        delantera = ['delante mio', 'trate de frenar', r'de(?:\s|)frente',
                     r'no.?(?:pude evitar|lleg.*?|logro evitar)', r'en asegurado parte (?:delantera|frontal)', r'con asegurado parte delantera',
                     'me llevo puesto', r'me \w* en parte delantera', 'con asegurado frente', r'asegurado (parte frontal|frente)',
                     'tercero retroce', 'lo embisto', r'impact*\w', 'colis.*? .* asegurado parte delantera', 'asegurado .* colis.*? con su parte delantera',
                     'parte trasera .* tercero', 'tercero frena']
        trasera = ['detras mio', 'marcha atras', r'retrocedo', 'retroceso', 'retrocedi', 'hacia atras', 'soy embestido', r'siento.*?impact.*?', 'reversa',
                   r'asegurado estaba (?:detenido|parado|estacionado)', 'de atras',
                   'desde atras', r'marcha (atra|a atra)', 'parte trasera vehiculo asegurado']
        for sentence in delantera:
            if re.search(sentence, self.descripcion) and not re.search('marcha atras', self.descripcion):
                if (re.search('con la parte delantera', self.descripcion) or re.search('con parte delantera', self.descripcion)) and self.get_quien() == 'asegurado':
                    self.impac_position = 'delantera'
                    return self.impac_position
                words = self.descripcion.split()
                for i in range(len(words) - 1):
                    if words[i] == 'delantera':
                        if words[i + 1] == 'izquierda':
                            self.impac_position = 'delantera izquierda'
                            return self.impac_position
                        elif words[i + 1] == 'derecha':
                            self.impac_position = 'delantera derecha'
                            return self.impac_position
                self.impac_position = 'delantera'
                return self.impac_position
        for sentence in trasera:
            if re.search(sentence, self.descripcion):
                words = self.descripcion.split()
                for i in range(len(words) - 1):
                    if words[i] == 'trasera':
                        if words[i + 1] == 'izquierda':
                            self.impac_position = 'trasera izquierda'
                            return self.impac_position
                        elif words[i + 1] == 'trasera derecha':
                            self.impac_position = 'trasera derecha'
                            return self.impac_position
                self.impac_position = 'trasera'
                return self.impac_position
        return self.impac_position

    def get_movement(self):
        if re.search('circul', self.descripcion) or re.search('venia', self.descripcion):
            self.movimiento = 'si'
            return self.movimiento  # 119
        movement_words = [r'\w*ando', r'\w*endo', 'circulaba', 'cirulaba', 'cirulanso', 'circulanso', 'avanzaba', 'frenar', 'doblaba', 'adelantar', 'adelantaba', 'yendo', 'transitando', 'frena de golpe', 'dirigia']
        no_momvement_words = [r'estacionado', r'dete.?', r'arranc.?', 'saliendo', 'entrando', r'avanz.?', r'manio.?', r'parado.?', 'salia', 'entraba', 'estacionaba', 'estacionando', 'retiraba', 'parad']

        st_move, st_no_move = False, False

        for mw in movement_words:
            if st_move:
                break
            st_move = re.search(mw, self.descripcion)

        for nmw in no_momvement_words:
            if st_no_move:
                break
            st_no_move = re.search(nmw, self.descripcion)

        if st_no_move:
            self.movimiento = 'no'
            return self.movimiento
        else:
            self.movimiento = 'si'
            return 'si'

    def get_quien(self):
        asegurado = ['asegurado colisiona', 'embesti', 'no logro evitar', 'embisto', 'trate de frenar', r'no.?(?:pude evitar|lleg.*?|logro evitar)', 'me llevo puesto', 'delante mio', 'de frente', 'delante de mi', r'(?:colisiono|colisiono con|toco con|impacto con|embistiendolo con|impactando con|colisione con) asegurado parte delantera', 'impacto a vehiculo', 'asegurado .* colis.*? con su parte delantera', 'parte trasera .* tercero']
        tercero = ['soy \w{0,2} embes.*?']
        for st in asegurado:
            if re.search(st, self.descripcion):
                self.quien = 'asegurado'
                return self.quien

        if re.search('tercero colisiona', self.descripcion):
            self.quien = 'tercero'
            return self.quien
