import cv2
import mediapipe
import numpy as np
import os 
import time

dataPath = 'C:/Users/gabri/OneDrive/Escritorio/YAIR/facemask_Detection/Mascarillas dataset/Dataset_faces'
dirList = os.listdir(dataPath)
print ('Lista de archivos: ', dirList)

Etiquetas = []
facesData = []
Etiqueta = 0

for nameDir in dirList:
    personPath = dataPath + '/' + nameDir
    print ('leyendo las imagenes...')

    for fileName in os.listdir(personPath):
        print('Rostros: ',nameDir + '/' + fileName)
        Etiquetas.append(Etiqueta)
        facesData.append(cv2.imread(personPath + '/'+ fileName,0))
        #imagen = cv2.imread(personPath+ '/'+ fileName,0)
        #cv2.imshow('imagen',imagen)
        #cv2.waitKey(0)
    Etiqueta = Etiqueta +1

print ('etiquetas= ', Etiquetas)
print ('Numero de etiquetas 0 :', np.count_nonzero(np.array(Etiquetas)==0))
print ('Numero de etiquetas 1 :', np.count_nonzero(np.array(Etiquetas)==1))

#elegimos el metodo de entrenamiento para elreconocimiento de rostros
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

print ('Entrenando Metodo...')
print ('....................')
inicio = time.time()
face_recognizer.train(facesData, np.array(Etiquetas))
timeTrainig = time.time()-inicio
print ('Tiempo de entrenamiento: ', timeTrainig)

#almacenamos el modelo del entrenamiento creado
#face_recognizer.write ('modeloEigenFace.xml')
#face_recognizer.write ('modeloFisherFace.xml')
face_recognizer.write ('face_mask_model.xml')
print ('Modelo Almacenado con exito!!!')