import re

class ManualExtraction:
    @staticmethod
    def get_quien(descripcion):
        asegurado = ['asegurado colisiona', 'embesti', 'no logro evitar', 'embisto', 'trate de frenar', r'no.?(?:pude evitar|lleg.*?|logro evitar)', 'me llevo puesto', 'delante mio', 'de frente', 'delante de mi', r'(?:colisiono|colisiono con|toco con|impacto con|embistiendolo con|impactando con|colisione con) asegurado parte delantera', 'impacto a vehiculo', 'asegurado .* colis.*? con su parte delantera', 'parte trasera .* tercero']
        tercero = ['soy \w{0,2} embes.*?']
        for st in asegurado:
            if re.search(st, descripcion):
                return 'asegurado'
        if re.search('tercero colisiona', descripcion):
            return 'tercero'
        return 'desconocido'
    
    @staticmethod
    def get_movement(descripcion):
        if re.search('circul', descripcion) or re.search('venia', descripcion):
            return 'si'
        movement_words = [r'\w*ando', r'\w*endo', 'circulaba', 'cirulaba', 'cirulanso', 'circulanso', 'avanzaba', 'frenar', 'doblaba', 'adelantar', 'adelantaba', 'yendo', 'transitando', 'frena de golpe', 'dirigia']
        no_momvement_words = [r'estacionado', r'dete.?', r'arranc.?', 'saliendo', 'entrando', r'avanz.?', r'manio.?', r'parado.?', 'salia', 'entraba', 'estacionaba', 'estacionando', 'retiraba', 'parad']
        st_move, st_no_move = False, False
        for mw in movement_words:
            if st_move:
                break
            st_move = re.search(mw, descripcion)
        for nmw in no_momvement_words:
            if st_no_move:
                break
            st_no_move = re.search(nmw, descripcion)
        if st_no_move:
            return 'no'
        else:
            return 'si'

    @staticmethod
    def get_ubicacion_vial(descripcion):
        location = ['calle', r'garaje', r'roton\w*', 'autopista', 'avenida', 'cruce', 'cruze', r'esquina\w*', r'estacionami\w*', 'carril', 'ruta', r'semaforo\w*', r'intersec.?', 'tunel', 'peaje']
        aux = []
        for loc in location:
            st = re.search(loc, descripcion)
            if st:
                aux.append(st.group())
        if aux:
            return ' '.join(set(aux))
        else:
            return 'desconocido'

    @staticmethod
    def get_impac_position(descripcion):
        delantera = ['delante mio', 'trate de frenar', r'de(?:\s|)frente',
                    r'no.?(?:pude evitar|lleg.*?|logro evitar)', r'en asegurado parte (?:delantera|frontal)', r'con asegurado parte delantera',
                    'me llevo puesto', r'me \w* en parte delantera', 'con asegurado frente', r'asegurado (parte frontal|frente)',
                    'tercero retroce', 'lo embisto', r'impact*\w', 'colis.*? .* asegurado parte delantera', 'asegurado .* colis.*? con su parte delantera',
                    'parte trasera .* tercero', 'tercero frena']
        trasera = ['detras mio', 'marcha atras', r'retrocedo', 'retroceso', 'retrocedi', 'hacia atras', 'soy embestido', r'siento.*?impact.*?', 'reversa',
                r'asegurado estaba (?:detenido|parado|estacionado)', 'de atras',
                'desde atras', r'marcha (atra|a atra)', 'parte trasera vehiculo asegurado']
        for sentence in delantera:
            if re.search(sentence, descripcion) and not re.search('marcha atras', descripcion):
                if (re.search('con la parte delantera', descripcion) or re.search('con parte delantera', descripcion)) :
                    return 'delantera'
                words = descripcion.split()
                for i in range(len(words) - 1):
                    if words[i] == 'delantera':
                        if words[i + 1] == 'izquierda':
                            return 'delantera izquierda'
                        elif words[i + 1] == 'derecha':
                            return 'delantera derecha'
                return 'delantera'
        for sentence in trasera:
            if re.search(sentence, descripcion):
                words = descripcion.split()
                for i in range(len(words) - 1):
                    if words[i] == 'trasera':
                        if words[i + 1] == 'izquierda':
                            return 'trasera izquierda'
                        elif words[i + 1] == 'trasera derecha':
                            return 'trasera derecha'
                return 'trasera'
        return 'desconocido'
