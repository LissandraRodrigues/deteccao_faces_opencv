# 16/05/2020
# Luiza Lissandra R. Rosa
# Contato: luizalissandrarosa@poli.ufrj.br
# Descrição: Programa que detecta o número de faces existentes em um frame.

# Importação.
import cv2

# Programa principal.
def main():

    # Arquivo XML que contêm as características de faces. 
    classifierPerson = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml') 

    colorText = (0, 0, 0)

    fontText = cv2.FONT_HERSHEY_COMPLEX_SMALL

    video = cv2.VideoCapture(0)

    while (True):

        connected, frame = video.read()

        # Converte o frame para cinza, pois otimiza a detecção.
        # Em alguns casos, COLOR_BGR2HSV se mostrou melhor que o COLOR_BGR2GRAY.
        convertedImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # A partir do classificador, detecta se há faces no frame.
        facesDetector = classifierPerson.detectMultiScale(convertedImage)

        facesCounter = 0

        # Insere bounding box para cada face detectada.
        for (x, y, l, a) in facesDetector:

            cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 255, 0), 2)  # BGR (Retângulo verde).

            facesCounter += 1

        cv2.putText(frame, 'Faces: ' + str(facesCounter), (10,30), fontText, 1, colorText, 2)

        cv2.imshow('Detector de Faces', frame)  

        # O programa é fechado ao apertar a tecla 'x'.
        if(cv2.waitKey(1) == ord('x')):

            break

    video.release()
    cv2.destroyAllWindows()

main()
