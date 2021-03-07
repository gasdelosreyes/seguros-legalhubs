from models.features_extraction import ManualExtraction
import joblib

class ManualClassifier:
    def getResponsable(self):
        return self.responsable

    def getAsegImpact(self):
        return self.asegImpact
    
    def getTercImpact(self):
        return self.tercImpact
    
    def getVialLocation(self):
        return self.vialLocation
    
    def getMovement(self):
        return self.movement

    def getResponsability(self):
        return self.responsability

    def getResponsabilityKNeighbors(self):
        return self.responsability_kneighbors

    def setResponsable(self,value):
        self.responsable = value

    def setAsegImpact(self, value):
        self.asegImpact = value
    
    def setTercImpact(self,value):
        self.tercImpact = value
    
    def setVialLocation(self,value):
        self.vialLocation = value
    
    def setMovement(self,value):
        self.movement = value

    def setResponsability(self,value):
        self.responsability = value
    
    def setResponsabilityKNeighbors(self,value):
        self.responsability_kneighbors = value

    def chargeModel(self,path):
        try:
            return joblib.load(path)
        except:
            print('NO SE PUDO CARGAR EL MODELO')
            return None
    
    def __init__(self, kneighbor_model=str):
        self.setResponsable(None)
        self.setAsegImpact(None)
        self.setTercImpact(None)
        self.setVialLocation(None)
        self.setMovement(None)
        # self.setResponsabilityManual(None)
        # self.setResponsabilityANN(None)
        self.setResponsabilityKNeighbors(None)
        # self.setResponsabilityDecisionTree(None)
        # self.modelANN = None
        self.modelKNeighbors = self.chargeModel(kneighbor_model)
        # self.modelDecisionTree = None

    def get_case_features(self, case_str = str):
        """
        case_str = descripcion del caso
        returns: perfil del accidente sin la predicción de la responsabilidad.
        """
        self.setResponsable(ManualExtraction.get_quien(case_str))
        self.setAsegImpact(ManualExtraction.get_impac_position(case_str))
        self.setTercImpact('desconocido')
        self.setVialLocation(ManualExtraction.get_ubicacion_vial(case_str))
        self.setMovement(ManualExtraction.get_movement(case_str))
        print('RESPONSABLE DEL ACCIDENTE: ' + str(self.getResponsable()))
        print('IMPACTO ASEGURADO: ' + str(self.getAsegImpact()))
        print('IMPACTO TERCERO: ' + str(self.getTercImpact()))
        print('UBICACION VIAL: ' + str(self.getVialLocation()))
        print('ESTADO DE MOVIMIENTO: ' + str(self.getMovement()))

    def profile_transform_kneighbors(self):
        """
        returns: el perfil codificado en una vector de 18x1:
        movimiento	impac_position[1-6]	quien[0-1]	
        calle	garaje	roton\w*	autopista	avenida	    cruce
        cruze	esquina\w*	estacionami\w*	carril	ruta	semaforo\w*
        intersec.?	tunel	peaje
        """
        import re 
        import numpy as np 

        vector = []
        vector.append(1 if self.getMovement() == 'si' else 0)
        positions = ['delantera','delantera izquierda','delantera derecha','trasera','trasera izquierda','trasera derecha']
        
        for i,row in enumerate(positions):
            if(self.getAsegImpact() != 'desconocido'):
                if row==self.getAsegImpact():
                    vector.append(i+1)
                    break
            else:
                vector.append(0)
                break
        
        vector.append(1 if self.getResponsable() == 'asegurado' else 0)
        location = [ 'calle', r'garaje', r'roton\w*', 'autopista', 'avenida', 'cruce', 'cruze', r'esquina\w*', r'estacionami\w*', 'carril', 'ruta', r'semaforo\w*', r'intersec.?', 'tunel', 'peaje']
        for i in location:
            if re.search(i,str(self.getVialLocation())):
                vector.append(1)
            else:
                vector.append(0)
        # vector += [1 for i in location if re.search(i,self.getVialLocation()) else 0]
        return vector
        
    def infer_responsability_kneighbors(self, vector_features):
        # print(self.modelKNeighbors.predict([vector_features])[0])
        predict = self.modelKNeighbors.predict([vector_features])[0]
        self.setResponsabilityKNeighbors(predict)
        if(str(self.responsability_kneighbors) == '0'):
            print('RESPONSABILIDAD KNEIGHBORS: NO COMPROMETIDA')
        elif(str(self.responsability_kneighbors) == '1'):
            print('RESPONSABILIDAD KNEIGHBORS: COMPROMETIDA')