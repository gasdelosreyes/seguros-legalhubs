import pandas as pd
from objetos import *


def ResponsabilityPredictor(tabla):
    """
    se le suministra una tabla con todos los casos
    luego selecciona los perfiles completos y predice su responsabilidad
    agrega una nueva propiedad que es la responsabilidad_predic
    calcula el porcentaje de acierto
    """
    for caso in tabla.get_casos():
        # caso.set_responsabilidad_predic(0)
        if caso.get_movement() == 'si':
            if caso.get_quien() == 'asegurado':
                if 'delantera' in caso.get_impac_position():
                    if re.search(r'estacionami*\w', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                        # porque puede pasar que el tercero es el que salía del estacionamiento
                    elif re.search('semaforo', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(1)
                        # porque no alcanzo a frenar
                    elif re.search('garaje', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                        # porque si el asegurado sale y choca o le chocan es su culpa
                    elif re.search('esquina', caso.get_ubicacion_vial()):
                        if re.search('izquierda', caso.get_ubicacion_vial()):
                            caso.set_responsabilidad_predic(-1)
                            # el asegurado tenia el paso
                        else:
                            caso.set_responsabilidad_predic(1)
                    elif re.search('intersec.?', caso.get_ubicacion_vial()):
                        if re.search('izquierda', caso.get_ubicacion_vial()):
                            caso.set_responsabilidad_predic(-1)
                            # el asegurado tenia el paso
                        else:
                            caso.set_responsabilidad_predic(1)
                    elif re.search('calle', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(1)
                    elif re.search('avenida', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(1)
                    elif re.search('ruta', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(1)
                    elif re.search('autopista', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(1)
                    elif re.search('roton.?', caso.get_ubicacion_vial()):
                        if re.search('derecha', caso.get_impac_position()):
                            caso.set_responsabilidad_predic(-1)
                            # porque el asegurado estaba dentro de la rotonda
                        else:
                            caso.set_responsabilidad_predic(1)
                    elif re.search('cruce', caso.get_ubicacion_vial()):
                        if re.search('izquierda', caso.get_ubicacion_vial()):
                            caso.set_responsabilidad_predic(-1)
                        else:
                            caso.set_responsabilidad_predic(1)
                    elif re.search('carril', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                        # porque alguien mas estaba cambiando de carril
                    elif re.search('ruta', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(1)
                    elif re.search('desconocido', caso.get_ubicacion_vial()):
                    	caso.set_responsabilidad_predic(1)
                if 'trasera' in caso.get_impac_position():
                    if re.search(r'estacionami*\w', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(1)
                        # porque puede pasar que el tercero es el que salía del estacionamiento
                    elif re.search('semaforo', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(1)
                        # porque hace marcha atras
                    elif re.search('garaje', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(1)
                        # porque si esta saliendo y colisiona es su culpa
                    elif re.search('esquina', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(1)
                        # porque el colisiona a alquien que estaba doblando
                    elif re.search('intersec.?', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                        # improbable de que pase pero si lo hace no es su culpa
                    elif re.search('calle', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(1)
                        # el hace marcha atras
                    elif re.search('avenida', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(1)
                    elif re.search('ruta', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('autopista', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('roton.?', caso.get_ubicacion_vial()):
                            # el sgt if/else se lo puede meter en una funcion PrioridadPaso()--> ret
                        if re.search('derecha', caso.get_impac_position()):
                            caso.set_responsabilidad_predic(-1)
                        else:
                            caso.set_responsabilidad_predic(1)
                    elif re.search('cruce', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('carril', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('ruta', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('desconocido', caso.get_ubicacion_vial()):
                    	caso.set_responsabilidad_predic(1)
            elif caso.get_quien() == 'tercero':
                if 'delantera' in caso.get_impac_position():
                    if re.search(r'estacionami*\w', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                        # porque puede pasar que el tercero es el que salía del estacionamiento
                    elif re.search('semaforo', caso.get_ubicacion_vial()):
                        if re.search('izquierda', caso.get_impac_position()):
                            caso.set_responsabilidad_predic(-1)
                        else:
                            caso.set_responsabilidad_predic(1)
                        # porque no alcanzo a frenar
                    elif re.search('garaje', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                        # porque si el asegurado sale y choca o le chocan es su culpa
                    elif re.search('esquina', caso.get_ubicacion_vial()):
                        if re.search('izquierda', caso.get_ubicacion_vial()):
                            caso.set_responsabilidad_predic(-1)
                            # el asegurado tenia el paso
                        else:
                            caso.set_responsabilidad_predic(1)
                    elif re.search('intersec.?', caso.get_ubicacion_vial()):
                        if re.search('izquierda', caso.get_ubicacion_vial()):
                            caso.set_responsabilidad_predic(-1)
                            # el asegurado tenia el paso
                        else:
                            caso.set_responsabilidad_predic(1)
                    elif re.search('calle', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('avenida', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('ruta', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('autopista', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('roton.?', caso.get_ubicacion_vial()):
                        if re.search('derecha', caso.get_impac_position()):
                            caso.set_responsabilidad_predic(-1)
                            # porque el asegurado estaba dentro de la rotonda
                        else:
                            caso.set_responsabilidad_predic(1)
                    elif re.search('cruce', caso.get_ubicacion_vial()):
                        if re.search('izquierda', caso.get_ubicacion_vial()):
                            caso.set_responsabilidad_predic(-1)
                        else:
                            caso.set_responsabilidad_predic(1)
                    elif re.search('carril', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                        # porque alguien mas estaba cambiando de carril
                    elif re.search('ruta', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('desconocido', caso.get_ubicacion_vial()):
                    	caso.set_responsabilidad_predic(-1)
                if 'trasera' in caso.get_impac_position():
                    if re.search(r'estacionami*\w', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                        # porque puede pasar que el tercero es el que salía del estacionamiento
                    elif re.search('semaforo', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                        # porque hace marcha atras
                    elif re.search('garaje', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                        # porque si esta saliendo y colisiona es su culpa
                    elif re.search('esquina', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                        # porque el colisiona a alquien que estaba doblando
                    elif re.search('intersec.?', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                        # improbable de que pase pero si lo hace no es su culpa
                    elif re.search('calle', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                        # el hace marcha atras
                    elif re.search('avenida', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('ruta', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('autopista', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('roton.?', caso.get_ubicacion_vial()):
                            # el sgt if/else se lo puede meter en una funcion PrioridadPaso()--> ret
                        if re.search('derecha', caso.get_impac_position()):
                            caso.set_responsabilidad_predic(-1)
                        else:
                            caso.set_responsabilidad_predic(1)
                    elif re.search('cruce', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('carril', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('ruta', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('desconocido', caso.get_ubicacion_vial()):
                    	caso.set_responsabilidad_predic(-1)

        elif caso.get_movement() == 'no':
            if caso.get_quien() == 'asegurado':
                if 'delantera' in caso.get_impac_position():
                    if re.search(r'estacionami*\w', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(1)
                        # porque puede pasar que el tercero es el que salía del estacionamiento
                    elif re.search('semaforo', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(1)
                        # porque no alcanzo a frenar
                    elif re.search('garaje', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(1)
                        # porque si el asegurado sale y choca o le chocan es su culpa
                    elif re.search('esquina', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(1)
                    elif re.search('intersec.?', caso.get_ubicacion_vial()):
                           caso.set_responsabilidad_predic(1)
                    elif re.search('calle', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(1)
                    elif re.search('avenida', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(1)
                    elif re.search('ruta', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(1)
                    elif re.search('autopista', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(1)
                    elif re.search('roton.?', caso.get_ubicacion_vial()):
                        if re.search('derecha', caso.get_impac_position()):
                            caso.set_responsabilidad_predic(-1)
                            # porque el asegurado estaba dentro de la rotonda
                        else:
                            caso.set_responsabilidad_predic(1)
                    elif re.search('cruce', caso.get_ubicacion_vial()):
                        if re.search('izquierda', caso.get_ubicacion_vial()):
                            caso.set_responsabilidad_predic(-1)
                        else:
                            caso.set_responsabilidad_predic(1)
                    elif re.search('carril', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(1)
                        # porque alguien mas estaba cambiando de carril
                    elif re.search('ruta', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(1)
                    elif re.search('desconocido', caso.get_ubicacion_vial()):
                    	caso.set_responsabilidad_predic(1)
                if 'trasera' in caso.get_impac_position():
                    if re.search(r'estacionami*\w', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(1)
                        # porque puede pasar que el tercero es el que salía del estacionamiento
                    elif re.search('semaforo', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                        # porque hace marcha atras
                    elif re.search('garaje', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(1)
                        # porque si esta saliendo y colisiona es su culpa
                    elif re.search('esquina', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                        # porque el colisiona a alquien que estaba doblando
                    elif re.search('intersec.?', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(1)
                        # improbable de que pase pero si lo hace no es su culpa
                    elif re.search('calle', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                        # el hace marcha atras
                    elif re.search('avenida', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('ruta', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('autopista', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('roton.?', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('cruce', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('carril', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('ruta', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('desconocido', caso.get_ubicacion_vial()):
                    	caso.set_responsabilidad_predic(1)
            elif caso.get_quien() == 'tercero':
                if 'delantera' in caso.get_impac_position():
                    if re.search(r'estacionami*\w', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                        # porque puede pasar que el tercero es el que salía del estacionamiento
                    elif re.search('semaforo', caso.get_ubicacion_vial()):
                        if re.search('izquierda', caso.get_impac_position()):
                            caso.set_responsabilidad_predic(-1)
                        else:
                            caso.set_responsabilidad_predic(1)
                        # porque no alcanzo a frenar
                    elif re.search('garaje', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                        # porque si el asegurado sale y choca o le chocan es su culpa
                    elif re.search('esquina', caso.get_ubicacion_vial()):
                        if re.search('izquierda', caso.get_ubicacion_vial()):
                            caso.set_responsabilidad_predic(-1)
                            # el asegurado tenia el paso
                        else:
                            caso.set_responsabilidad_predic(1)
                    elif re.search('intersec.?', caso.get_ubicacion_vial()):
                        if re.search('izquierda', caso.get_ubicacion_vial()):
                            caso.set_responsabilidad_predic(-1)
                            # el asegurado tenia el paso
                        else:
                            caso.set_responsabilidad_predic(1)
                    elif re.search('calle', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('avenida', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('ruta', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('autopista', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('roton.?', caso.get_ubicacion_vial()):
                        if re.search('derecha', caso.get_impac_position()):
                            caso.set_responsabilidad_predic(-1)
                            # porque el asegurado estaba dentro de la rotonda
                        else:
                            caso.set_responsabilidad_predic(1)
                    elif re.search('cruce', caso.get_ubicacion_vial()):
                        if re.search('izquierda', caso.get_ubicacion_vial()):
                            caso.set_responsabilidad_predic(-1)
                        else:
                            caso.set_responsabilidad_predic(1)
                    elif re.search('carril', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                        # porque alguien mas estaba cambiando de carril
                    elif re.search('ruta', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('desconocido', caso.get_ubicacion_vial()):
                    	caso.set_responsabilidad_predic(-1)
                if 'trasera' in caso.get_impac_position():
                    if re.search(r'estacionami*\w', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                        # porque puede pasar que el tercero es el que salía del estacionamiento
                    elif re.search('semaforo', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                        # porque hace marcha atras
                    elif re.search('garaje', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                        # porque si esta saliendo y colisiona es su culpa
                    elif re.search('esquina', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                        # porque el colisiona a alquien que estaba doblando
                    elif re.search('intersec.?', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                        # improbable de que pase pero si lo hace no es su culpa
                    elif re.search('calle', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                        # el hace marcha atras
                    elif re.search('avenida', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('ruta', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('autopista', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('roton.?', caso.get_ubicacion_vial()):
                            # el sgt if/else se lo puede meter en una funcion PrioridadPaso()--> ret
                        if re.search('derecha', caso.get_impac_position()):
                            caso.set_responsabilidad_predic(-1)
                        else:
                            caso.set_responsabilidad_predic(1)
                    elif re.search('cruce', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('carril', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('ruta', caso.get_ubicacion_vial()):
                        caso.set_responsabilidad_predic(-1)
                    elif re.search('desconocido', caso.get_ubicacion_vial()):
                    	caso.set_responsabilidad_predic(-1)
    kpi(tabla)


def kpi(tabla):
    """
    retorna el porcentaje de aciertos en las responsabilidades
    """
    good, totales, bad = 0, 0, 0
    for caso in tabla.casos:
        if is_complete(caso):
            totales += 1
            if caso.get_responsabilidad() == 'COMPROMETIDA' or caso.get_responsabilidad() == 'DISCUTIDA':
                if 0 < caso.get_responsabilidad_predic():
                    good += 1
                else:
                    bad += 1
            else:
                if caso.get_responsabilidad_predic() < 0:
                    good += 1
                else:
                    bad = + 1
    print('bien: ', good / totales, '\nmal: ', bad / totales, '\nno clasificados: ', (totales - good - bad) / totales)


df = pd.read_csv('../dataset/casos/auto-clean.csv')


tabla = TablaCasos('Tabla1')

for i in range(len(df)):
	caso = Caso()
	caso.set_descripcion(df.loc[i, 'descripcion'])
	caso.set_idxDesc(i)  # no estoy seguro si es en i del csv o debería ser el i del csv original
	caso.set_responsabilidad(df.loc[i, 'responsabilidad'])
	tabla.set_caso(caso)

tabla.set_caso(caso)
tabla.update()
tabla.status()
# ResponsabilityPredictor(tabla)
# kpi(tabla)


tabla.to_csv()
