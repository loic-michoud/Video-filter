

import cv2
from PIL import Image
import dlib
from tkinter import *
from interface import Interface
import numpy as np
import sys
import cmath
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from fonctions import cartoon
from fonctions import insert_filtre
from fonctions import draw_landmarks
from fonctions import filtre_vertical
from fonctions import inserer_lunettes
from fonctions import filtre_negatif
import numpy
from fonctions import inserer_cornes
import scipy.ndimage
from fonctions import inserer_coeur
from fonctions import inserer_gps
from music import Music
from fonctions import reconstruction
from fonctions import recup_carré
from fonctions import recup_rectangle_bouche
from fonctions import reconstruction_bouche
from fonctions import bouche_filter
from faceSwap import faceSwap
from faceSwap import resize
from faceSwap import inserer_macron
from faceSwap import masque

fenetre = Tk()
interface = Interface(fenetre)
interface.mainloop()


macron=faceSwap()


music = Music(0.5, "data/starwars.mp3")
cap = cv2.VideoCapture(0)
width = 320
height = 240
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

ret, frame = cap.read()
sys.setrecursionlimit(frame.size)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

lunettes=Image.open('data/glasses.jpg')
cornes=Image.open('data/cornes.jpg')
coeur=Image.open('data/coeur.jpg')
gps=Image.open('data/gps.jpg')
bouche=Image.open('data/bouche.jpg')



choix=0
mettre_lunettes=False
mettre_cornes=False
filtre_cartoon=False
tomber_coeur=False
flou_gaussien=False
filtre_gps=False
space=False
filtre_bouche=False
play_music=False
macron_face=False

cpt=0
cpt2=0
cpt3=0
screen=0
j=0
while (True):

    c = cv2.waitKey(1) % 256
    ret, frame = cap.read()


    img_tampon=np.copy(frame)


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    if (filtre_cartoon):
        frame = cartoon(frame)  # fond de l'image
    if(flou_gaussien):
        frame=scipy.ndimage.gaussian_filter(frame, sigma=3.0)

    if (macron_face):
        frame[min_y:max_y, min_x:max_x] = inserer_macron(frame[min_y:max_y, min_x:max_x], macron, min_y, max_y,
                                                         min_x, max_x)


    image_nature = np.copy(frame)
    rects = detector(gray, 1)

    pointx = []
    pointy = []


    for (i, rect) in enumerate(rects):
        landmarks = predictor(gray, rect)
        draw_landmarks(landmarks, frame, pointx, pointy)

        max_y = np.amax(pointy)
        min_y = np.amin(pointy)
        max_x = np.amax(pointx)
        min_x = np.amin(pointx)

        pos = filtre_vertical(frame, min_y, max_y, min_x - 1, max_x + 1, image_nature)
        visage=frame



        if (space):

            j = j + 1
            if (j == 1):
                cap2 = cv2.VideoCapture('data/starwars.gif')
            if(play_music==False):
                music.start()
                play_music=True
            ret2, frame_space = cap2.read()
            frame_space = PIL.Image.fromarray(frame_space)
            frame_space = frame_space.resize((320, 240), Image.ANTIALIAS)
            frame_space = np.array(frame_space)
            frame = insert_filtre(frame_space, visage, pos)
            if (j == 84):

                j = 0

        if(choix==1):
            visage =cv2.Canny(img_tampon,100,150)
            frame = insert_filtre(frame, visage, pos)
        if(choix==2):
            visage = filtre_negatif(img_tampon)
            frame = insert_filtre(frame, visage, pos)

        if (filtre_bouche):
            en_attendant = np.copy(frame)
            [carré, pos_x] = recup_rectangle_bouche(en_attendant, pointx[48] - 10, pointy[52] - 10, pointx[54] + 10,
                                                    pointy[52] - 10, pointx[48] - 10, pointy[57] + 10, pointx[54] + 10,
                                                    pointy[57] + 10)
            visage_bouche = bouche_filter(carré, pointx[48], pointy[48], pointx[54], pointy[54], bouche)
            reconstruction_bouche(frame, visage_bouche, pos_x, pointx[48] - 10, pointy[52] - 10, pointx[54] + 10,
                                  pointy[52] - 10, pointx[48] - 10, pointy[57] + 10, pointx[54] + 10, pointy[57] + 10)

        if(mettre_lunettes):
            # visage = inserer_lunettes(frame[min_y:max_y - 30, min_x + 5:max_x], lunettes)
            # frame[min_y:max_y - 30, min_x + 5:max_x] = visage
            en_attendant = np.copy(frame)
            [carré, pos_x] = recup_carré(en_attendant, min(pointx[0], pointx[19]), min(pointy[0], pointy[19]),
                                         pointx[3], pointy[3], max(pointx[16], pointx[24]), min(pointy[16], pointy[24]),
                                         pointx[13], pointy[13], min_y, max_y, min_x, max_x)
            visage = inserer_lunettes(carré, lunettes, width, height, pointy[17], pointy[26], pointx[27], pointx[26],
                                      pos_x)
            reconstruction(frame, visage, pos_x, min_y, min_x)

        if(mettre_cornes):
            hypo = np.sqrt(abs(pointx[0] - pointx[16]) * abs(pointx[0] - pointx[16]) + (pointy[0] - pointy[16]) * (
                        pointy[0] - pointy[16]))
            angle = (np.arcsin(abs(pointx[0] - pointx[16]) / hypo) * 180) / 3.14
            if (pointy[0] - pointy[16]) < 0:
                angle = angle - 90
            else:
                angle = 90 - angle
            visage = inserer_cornes(
                frame[pointy[27] - int(0.6 * abs(max_y - min_y)): pointy[27] - int(0.2 * abs(max_y - min_y)),
                pointx[27] - int(0.5 * abs(max_x - min_x)) - int(angle * 0.005 * abs(max_x - min_x)):pointx[27] + int(
                    0.5 * abs(max_x - min_x)) - int(angle * 0.005 * abs(max_x - min_x))], cornes, pointx[36],
                pointy[36], pointx[45], pointy[45])

    if (tomber_coeur):
        frame, cpt= inserer_coeur(frame,coeur,0,20,50,70,cpt,(0,0,255))
        frame,cpt2=inserer_coeur(frame,coeur,30,50,130,150,cpt2,(161,25,239))
        frame,cpt3=inserer_coeur(frame,coeur,80,100,200,220,cpt3,(25,119,239))
        frame,cpt=inserer_coeur(frame,coeur,0,20,280,300,cpt3,(239,171,25))


    if (filtre_gps):
        frame[0:gps.size[1], 0:gps.size[0]] = inserer_gps(frame[0:gps.size[1], 0:gps.size[0]], gps)



    frame = PIL.Image.fromarray(frame)
    frame=np.array(frame.resize((800, 600), Image.BICUBIC))

    if c != 0xFF:

        if c == ord('q'):
            print("Fermeture de l'application :'(")
            break

        if c==ord(' '):
            cv2.imwrite('screenshots/screenshot'+'%d'%screen+'.jpg',frame)
            screen=screen+1

        if c ==ord('z'):
            if(choix!=1):
               choix = 1
               macron_face=False
            else:

                choix=0

        if c == ord('e'):
            if(choix!=2):
                choix = 2
                macron_face = False
            else:
                choix=0
        if c == ord('r'):
            if(mettre_lunettes==False):
                mettre_lunettes=True
            else:
                mettre_lunettes=False

        if c== ord('t'):
            if(mettre_cornes==False):
                mettre_cornes=True
            else:
                mettre_cornes=False

        if c== ord('s'):
            if(filtre_cartoon==False):
                filtre_cartoon=True
                flou_gaussien=False
            else:
                filtre_cartoon=False

        if c == ord('m'):
            if (macron_face == False):
                macron_face = True
                choix = 0
            else:
                macron_face = False

        if c == ord('d'):
            if(tomber_coeur==False):
                tomber_coeur=True
            else:
                tomber_coeur=False
        if c == ord('f'):
            if (flou_gaussien ==False):
                flou_gaussien = True
                filtre_cartoon=False
            else:
                flou_gaussien = False
        if c==ord('a'):
            if(filtre_gps==False):
                filtre_gps=True
            else:
                filtre_gps=False
        if c == ord('g'):
            if (space == False):
                if(play_music):
                    music.play_music()
                space = True
            else:
                space = False
                music.stop()
                j=0


        if c == ord('b'):
            if (filtre_bouche == False):
                filtre_bouche = True
            else:
                filtre_bouche = False


    cv2.imshow("Projet Majeure",frame)
    cpt=cpt+10
    cpt2=cpt2+10
    cpt3=cpt3+10

cap.release()
cv2.destroyAllWindows()






