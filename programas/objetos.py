import re
import pandas as pd


def printDic(dic):
	for i in range(len(dic)):
		key = list(dic.keys())
		value = list(dic.values())
		print(key[i], value[i])


class TablaCasos:
	"""Tabla con todos los casos"""

	def __init__(self, TableName):
		self.TableName = TableName
		self.impac_position = ''
		self.casos = []

	def set_caso(self, caso, update=False):
		exist = False
		for i in self.casos:
			if caso.get_idxDesc() == i.get_idxDesc():
				exist = True

		if not exist:
			self.casos.append(caso)
		if update:
			self.update()

	def get_casos(self):
		return self.casos

	def __str__(self):
		print(self.TableName)
		for caso in self.casos:
			print(caso.get_idxDesc(), caso.get_descripcion(), caso.get_responsabilidad())

	def to_csv(self):
		df = pd.DataFrame()
		df['idxDesc'] = pd.Series([caso.get_idxDesc() for caso in self.casos])
		df['descripcion'] = pd.Series([caso.get_descripcion() for caso in self.casos])
		df['ubicacion_vial'] = pd.Series([caso.get_ubicacion_vial() for caso in self.casos])
		df['movimiento'] = pd.Series([caso.get_movement() for caso in self.casos])
		df['responsabilidad'] = pd.Series([caso.get_responsabilidad() for caso in self.casos])
		df['impac_position'] = pd.Series([caso.get_impac_position() for caso in self.casos])

		df.to_csv(self.TableName + '.csv', index=False)

	def update(self):
		"""
		Actualiza las estadísticas que se deseen conocer
		"""
		self.casos_totales = len(self.casos)
		self.responsabilidad_comprometido = len([i.responsabilidad for i in self.casos if i.get_responsabilidad() == 'COMPROMETIDA'])
		self.responsabilidad_no_comprometido = len(self.casos) - len([i.responsabilidad for i in self.casos if i.get_responsabilidad() == 'COMPROMETIDA'])
		self.movimiento = len([i for i in self.casos if i.get_movement() == 'si'])
		self.no_movimiento = self.casos_totales - self.movimiento
		self.delantera = len([i for i in self.casos if i.get_impac_position() == 'delantera'])
		self.trasera = len([i for i in self.casos if i.get_impac_position() == 'trasera'])
		self.ubicaciones_viales = set([i.get_ubicacion_vial() for i in self.casos])

	def status(self):
		stat = {'casos totales': self.casos_totales, 'comprometidos': self.responsabilidad_comprometido, 'no_comprometidos': self.responsabilidad_no_comprometido, 'en_movimiento': self.movimiento, 'no_movimiento': self.no_movimiento, 'delantera': self.delantera, 'trasera': self.trasera}  # , 'lugares posibles': self.ubicaciones_viales}
		printDic(stat)


class Caso:
	"""Atributos del caso en vistas de una BD"""

	def __init__(self):
		self.impac_position = ''

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

	def get_ubicacion_vial(self):
		location = ['calle', r'garaje', r'roton\w*', 'autopista', 'avenida', 'cruce', 'cruze', r'esquina\w*', r'estacionami\w*', 'carril', 'ruta', r'semaforo\w*', r'intersec.?', 'tunel', 'aut','peaje']
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
                    r'no.?(?:pude evitar|lleg.*?)', r'en mi parte (?:delantera|frontal)', r'con mi parte delantera',
                    'me llevo puesto', r'me \w* en parte delantera', 'con mi frente', r'mi (parte frontal|frente)',
                    'tercero retroce','lo embisto']  # r'no lleg.*? frenar', r'no lleg.*? a'
		trasera = ['detras mio', 'marcha atras', r'retrocedo', 'retroceso', 'retrocedi', 'hacia atras', 'soy embestido', r'siento.*?impact.*?', 'reversa', r'asegurado estaba (?:detenido|parado|estacionado)', 'de atras',
                    'desde atras', r'marcha (atra|a atra)']
		# derecha = [r'para.*?doblar.*?derecha', 'dobla a la derecha', 'doblar para la derecha']
		# izquierda = [r'para.*?doblar.*?izquierda', 'dobla a la izquierda', 'doblar para la izquierda']
		for sentence in delantera:
			if re.search(sentence, self.descripcion):
				self.impac_position = 'delantera'
				return self.impac_position
		for sentence in trasera:
			if re.search(sentence, self.descripcion):
				self.impac_position = 'trasera'
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
