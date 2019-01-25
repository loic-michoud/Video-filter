from fonctions import filtre_vertical
import PIL
from PIL import Image
import cv2
import dlib
import numpy as np
from fonctions import draw_landmarks
from fonctions import rotation
import matplotlib.pyplot as plt


def masque(img, pos, min_y, max_y, min_x, max_x ):

    gros_carré = np.zeros((int(max(img.shape[1], img.shape[0])), int(max(img.shape[1], img.shape[0])), 3),
                          dtype=np.uint8)
    color = tuple((0, 255, 0))
    # Fill image with color
    gros_carré[:] = color

    for (i,p) in enumerate(pos):
                gros_carré[p[0],p[1]]=img[p[0],p[1]]
    return gros_carré[min_y:max_y,min_x:max_x]


def resize(img, min_y, max_y, min_x, max_x):

    img=Image.fromarray(img)
    w=max_y-min_y
    h=max_x-min_x
    img=img.resize((w,h), Image.ANTIALIAS)
    img=np.array(img)
    return img


def faceSwap():
    img=cv2.imread('data/macron.jpg')
    img=Image.fromarray(img)
    img=img.resize((320,240), Image.ANTIALIAS)
    img=np.array(img)
    image_nature = np.copy(img)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    pointx = []
    pointy = []

    for (i, rect) in enumerate(rects):

        landmarks = predictor(gray, rect)
        draw_landmarks(landmarks, img, pointx, pointy)

        max_y = np.amax(pointy)
        min_y = np.amin(pointy)
        max_x = np.amax(pointx)
        min_x = np.amin(pointx)
        pos = filtre_vertical(img, min_y, max_y, min_x - 1, max_x + 1, image_nature)
        img=masque(img,pos,min_y, max_y, min_x , max_x)

    return  img

def inserer_macron(img,macron,min_y, max_y, min_x , max_x):

    macron=resize(macron, min_x , max_x,min_y, max_y)
    for (i) in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(i<img.shape[0] and i>0 and j<img.shape[1] and j>0):
                if (macron[i,j][0]>40 and macron[i,j][1]<200 and macron[i,j][2]>30 ):
                    img[i,j]=macron[i,j]
    return img
