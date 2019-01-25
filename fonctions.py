import cv2
import numpy as np
import PIL
from PIL import Image
import numpy
import pygame as pg

import matplotlib.pyplot as plt
def inser(img, lunettes,pointy_17, pointy_26,pointx_27, pointx_26, pos_x):
    lunettes = lunettes.resize((img.shape[1], img.shape[0]), PIL.Image.ANTIALIAS)
    lunettes = numpy.array(lunettes.getdata()).reshape(lunettes.size[1], lunettes.size[0], 3)
    lunettes = cv2.convertScaleAbs(lunettes)
    lunettes = rotation(lunettes, img.shape[1], img.shape[0], pointy_17 - pointy_26, pointx_27 - pointx_26,False)
    gros_carré = np.zeros((int(max(img.shape[1],img.shape[0])), int(max(img.shape[1],img.shape[0])), 3), dtype=np.uint8)
    color = tuple((0,255,0))
    # Fill image with color
    gros_carré[:] = color
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
                if(lunettes[i,j][1]<70): # and lunettes[i,j][2] !=0)  or (lunettes[i,j][0] !=0 and lunettes[i,j][1] !=255 and lunettes[i,j][2] !=0): #or (lunettes[i,j][0]!=(0) and lunettes[i,j][1]!=(0) and lunettes[i,j][2]!=(0)) ):
                    if img.shape[0] > img.shape[1]:
                        gros_carré[i, j + img.shape[0]-img.shape[1]] = lunettes[i,j]
                    else:
                        gros_carré[i + int((img.shape[1] - img.shape[0])/2), j] = lunettes[i, j]
                else:
                    if img.shape[0] > img.shape[1]:
                        gros_carré[i, j + img.shape[0]-img.shape[1]] = img[i,j]
                    else:
                        gros_carré[i + int((img.shape[1] - img.shape[0])/2), j] = img[i, j]
                if(gros_carré[i +int((img.shape[1] - img.shape[0])/2), j][0]==0 and gros_carré[i +int((img.shape[1] - img.shape[0])/2), j][1]==255 and gros_carré[i +int((img.shape[1] - img.shape[0])/2), j][2]==0):
                    gros_carré[i + int((img.shape[1] - img.shape[0]) / 2), j] = img[i, j]
    # img = rotation(img, img.shape[1], img.shape[0], pointy_17 - pointy_26, pointx_27 - pointx_26)
    return gros_carré



def reconstruction(frame, visage, pos_x, min_y, min_x):
    for i in range(visage.shape[0]):
        if len(pos_x[i]) == 2:
            for j in range(pos_x[i][0], pos_x[i][1]):
                if visage[i,j-min_x+1][0] != 0 and visage[i,j-min_x+1][1] != 255 and visage[i,j-min_x+1][2] != 0:
                    frame[i+min_y-1,j] = visage[i,j-min_x+1]
def inserer_lunettes(img, lunettes,width,height, pointy_17, pointy_26,pointx_27, pointx_26, pos_x):
    lunettes = lunettes.resize((img.shape[1]-10, img.shape[0]-35), PIL.Image.ANTIALIAS)
    lunettes=numpy.array(lunettes.getdata()).reshape(lunettes.size[1], lunettes.size[0], 3)
    lunettes = cv2.convertScaleAbs(lunettes)
    lunettes = rotation(lunettes, img.shape[1]-10,img.shape[0]-35, pointy_17 -pointy_26, pointx_27 - pointx_26, False)
    # img = rotation(img, img.shape[1], img.shape[0], pointy_17 - pointy_26, pointx_27 - pointx_26)
    for i in range(img.shape[0]-35):
        for j in range(img.shape[1]-10):
            if( i<img.shape[0] and i>0 and j<img.shape[1] and j>0):
                if(lunettes[i,j][0]<100): #or (lunettes[i,j][0]!=(0) and lunettes[i,j][1]!=(0) and lunettes[i,j][2]!=(0)) ):
                    img[i, j+7] = lunettes[i,j]
    # img = rotation(img, img.shape[1], img.shape[0], pointy_17 - pointy_26, pointx_27 - pointx_26)
    return img

def reconstruction_bouche(frame, visage, pos_x, pointx_0, pointy_0, pointx_1 , pointy_1, pointx_2, pointy_2, pointx_3, pointy_3):
    for i in range(visage.shape[0]):
        if len(pos_x[i]) == 2:
            for j in range(pos_x[i][0], pos_x[i][1]):
                if ( visage[i,j - pos_x[i][0]][2] >= visage[i,j - pos_x[i][0]][1]) and (visage[i,j - pos_x[i][0]][0] !=0 and visage[i,j - pos_x[i][0]][1] < 220 and visage[i,j - pos_x[i][0]][2] != 0):
                    frame[i+pointy_0,j+ pointx_0 - pos_x[i][0]] = visage[i,j - pos_x[i][0]]
def recup_carré(frame, pointx_0,pointy_0, pointx_1,pointy_1, pointx_2,pointy_2, pointx_3,pointy_3, min_y, max_y, min_x, max_x):
    cv2.line(frame, (pointx_0,pointy_0), (pointx_2,pointy_2), (0, 255, 0), 2)
    cv2.line(frame, (pointx_0,pointy_0), (pointx_1, pointy_1), (0, 255, 0), 2)
    cv2.line(frame, (pointx_1, pointy_1), (pointx_3, pointy_3), (0, 255, 0), 2)
    cv2.line(frame, (pointx_3, pointy_3), (pointx_2,pointy_2), (0, 255, 0), 2)
    carré = np.zeros((int(abs(max_y+1 - min_y +1)), int(abs(max_x +1 - min_x +1)), 3), dtype=np.uint8)
    pos_x = []
    for i in range(min_y-1, max_y+1):
        cpt = 0
        cpt_tempo = 0
        pos_x.append([])
        tempo_x = []
        for j in range(min_x-1, max_x+1):
            if frame[i, j][1] == 255 and frame[i, j][0] == 0 and frame[i, j][2] == 0:
                if cpt ==0:
                    pos_x[i - min_y].append(j)
                    tempo_x.append(j)
                    cpt = cpt + 1
                    cpt_tempo = cpt_tempo + 1
                else:
                    if abs(j-tempo_x[cpt_tempo-1])>1:
                        pos_x[i - min_y].append(j)
                        tempo_x.append(j)
                        cpt = cpt + 1
                        cpt_tempo = cpt_tempo + 1
                    else:
                        tempo_x.append(j)
                        cpt_tempo = cpt_tempo + 1
        if cpt == 2:
            for j in range(pos_x[i - min_y][0]-1, pos_x[i - min_y][1]+1):
                if (i < 240 and i >= 0 and j >= 0 and j < 320):
                    carré[i-min_y,j- min_x] = frame[i,j]
    return [carré, pos_x]


def recup_rectangle_bouche(frame, pointx_0,pointy_0, pointx_1,pointy_1, pointx_2,pointy_2, pointx_3,pointy_3):
    cv2.line(frame, (pointx_0, pointy_0), (pointx_2, pointy_2), (0, 255, 0), 2)
    cv2.line(frame, (pointx_0, pointy_0), (pointx_1, pointy_1), (0, 255, 0), 2)
    cv2.line(frame, (pointx_1, pointy_1), (pointx_3, pointy_3), (0, 255, 0), 2)
    cv2.line(frame, (pointx_3, pointy_3), (pointx_2, pointy_2), (0, 255, 0), 2)
    carré = np.zeros((abs(pointy_2 - pointy_0), abs(pointx_1 - pointx_0), 3), dtype=np.uint8)
    pos_x = []
    for i in range(pointy_0, pointy_2):
        cpt = 0
        cpt_tempo = 0
        pos_x.append([])
        tempo_x = []
        for j in range(pointx_0, pointx_1):
            if frame[i, j][1] == 255 and frame[i, j][0] == 0 and frame[i, j][2] == 0:
                if cpt == 0:
                    pos_x[i-pointy_0].append(j)
                    tempo_x.append(j)
                    cpt = cpt + 1
                    cpt_tempo = cpt_tempo + 1
                else:
                    if abs(j - tempo_x[cpt_tempo - 1]) > 1:
                        pos_x[i-pointy_0].append(j)
                        tempo_x.append(j)
                        cpt = cpt + 1
                        cpt_tempo = cpt_tempo + 1
                    else:
                        tempo_x.append(j)
                        cpt_tempo = cpt_tempo + 1
        if cpt == 2:
            for j in range(pos_x[i-pointy_0][0] - 1, pos_x[i-pointy_0][1] + 1):
                if (i < 240 and i >= 0 and j >= 0 and j < 320):
                    carré[i - pointy_0, j - pointx_0] = frame[i, j]

    return [carré, pos_x]

def bouche_filter(image, pointx_0, pointy_0, pointx_1 , pointy_1, bouche):
    bouche = bouche.resize((image.shape[1], image.shape[0]), PIL.Image.ANTIALIAS)
    bouche = numpy.array(bouche.getdata()).reshape(bouche.size[1], bouche.size[0], 3)
    bouche = cv2.convertScaleAbs(bouche)
    bouche = rotation(bouche, image.shape[1],image.shape[0], pointy_0 -pointy_1, abs(pointx_1 - pointx_0), True)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (bouche[i, j][1] < 200) :#and bouche[i, j][0] > 2 and bouche[i, j][2] > 2:
                image[i , j][0] = bouche[i, j][2]
                image[i, j][1] = bouche[i, j][1]
                image[i, j][2] = bouche[i, j][0]
    return image


def cartoon(img):
    num_down = 2  # number of downsampling steps
    num_bilateral = 7  # number of bilateral filtering steps

    img_rgb =img

    # downsample image using Gaussian pyramid
    img_color = img_rgb
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)

    # repeatedly apply small bilateral filter instead of
    # applying one large filter
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, d=9,
                                        sigmaColor=9,
                                        sigmaSpace=7)

    # upsample image to original size
    for _ in range(num_down):
        img_color = cv2.pyrUp(img_color)
        # convert to grayscale and apply median blur
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    # detect and enhance edges
    img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY,
                                     blockSize=9,
                                     C=2)
    # convert back to color, bit-AND with color image
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    img_cartoon = cv2.bitwise_and(img_color, img_edge)
    return img_cartoon

def rotation(img,w,h, oppose, adjacent, angle_normal):
    rows, cols = [h,w]
    hypo = np.sqrt(adjacent*adjacent + oppose*oppose)
    if angle_normal:
        angle = (np.arcsin(oppose / hypo) * 180) / (3.14)
    else:
        angle = 0.45*(np.arcsin(oppose/hypo)*180)/(3.14)
    # angle = np.arcsin(oppose/hypo)
    R = cv2.getRotationMatrix2D((cols /2, rows/2 ), angle, 1)
    T = np.float32([[1, 0, 0], [0, 1, 0]])
    dst = cv2.warpAffine(img, R, (cols, rows))
    dst=cv2.warpAffine(dst,T,(cols, rows))
    return dst



def inserer_coeur(frame,coeur,x,y,x2,y2,cpt,c):

    coeur=coeur.convert('LA')
    coeur=coeur.convert('RGB')

    coeur = np.array(coeur)



    if (y + cpt < frame.shape[0]):

        for i in range(20):
            for j in range(20):
                if(coeur[i,j][0]>150 ):

                    coeur[i,j]=frame[x+i+cpt, x2 + j]



                else:
                    coeur[i, j]=c


        frame[x + cpt:y + cpt, x2:y2] = coeur


    else:
        cpt = 0

    return frame,cpt



# def inserer_lunettes(img, lunettes):
#
#     lunettes = lunettes.resize((img.shape[1],img.shape[0]), PIL.Image.ANTIALIAS)
#
#     lunettes=numpy.array(lunettes.getdata()).reshape(lunettes.size[1], lunettes.size[0], 3)
#
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             if( i<img.shape[0] and i>0 and j<img.shape[1] and j>0):
#                 if(lunettes[i,j][0]<80):
#                     img[i, j] = lunettes[i,j]
#
#
#
#     return img

def reconstruct_cornes(frame, visage, min_y,max_y, min_x,  max_x):
    for i in range(visage.shape[0]):
        for j in range(visage.shape[1]):
            if visage[i,j][0] != 0 and visage[i,j][1] != 0 and visage[i,j][2] != 0:
                frame[i+min_y, j+min_x] = visage[i,j]




def inserer_cornes(img,cornes, pointx_0, pointy_0, pointx_1, pointy_1):
    if(img.shape[1]>0 and img.shape[0]>0):
        cornes = cornes.resize((img.shape[1], img.shape[0]), PIL.Image.ANTIALIAS)
        cornes = numpy.array(cornes.getdata()).reshape(cornes.size[1], cornes.size[0], 3)
        cornes = cv2.convertScaleAbs(cornes)
        cornes = rotation(cornes, img.shape[1], img.shape[0], pointy_0 - pointy_1, abs(pointx_1 - pointx_0), True)
        for i in range(img.shape[0]+1):
            for j in range(img.shape[1]+1):
                if (i < img.shape[0] and i > 0 and j < img.shape[1] and j > 0):
                    if (cornes[i, j][0] <100) and (cornes[i, j][1] <85)and (cornes[i, j][0] !=0 and cornes[i, j][1] !=0 and cornes[i, j][2] !=0) and (cornes[i, j][0]!=0 and cornes[i, j][1] !=255 and cornes[i, j][2] !=0):
                        img[i, j] = cornes[i, j]
    return img






def filtre_negatif(img):
    img2=np.copy(img)
    img2[:,:]=(255,255,255)-img[:,:]
    return img2


def insert_filtre(image,image_filtree, pos):
    for (i,p) in enumerate(pos):
                image[p[0],p[1]]=image_filtree[p[0],p[1]]
    return image



def draw_BB(rect,image):
    x = rect.left()-10     #changer les paramètres pour bien avoir la détection du visage dans la Box
    y = rect.top()
    w = rect.right() - x+10
    h = rect.bottom() - y+15
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # cv2.circle(image, (rect.tl_corner().x, rect.tl_corner().y), 1, (0, 255, 0), -1)
    cv2.circle(image, (rect.br_corner().x, rect.br_corner().y), 1, (0, 0, 255), -1)


def draw_landmarks(landmarks, image, pointx, pointy):

    for i, part in enumerate(landmarks.parts()):
        px = int(part.x)
        py = int(part.y)
        pointx.append(px)
        pointy.append(py)

        if (i < 17):
            # cv2.circle(image, (px, py), 1, (0, 255, 0), -1)
            if (i > 0):
                cv2.line(image, (px, py), (px2, py2), (0, 255, 0),2)
            px2 = int(part.x)
            py2 = int(part.y)
        if (i < 23):
            # cv2.circle(image, (px, py), 1, (0, 255, 0), -1)
            if (i > 17):
                cv2.line(image, (px, py), (px2, py2), (0, 255, 0),2)
            px2 = int(part.x)
            py2 = int(part.y)
        if (i < 27):
            # cv2.circle(image, (px, py), 1, (0, 255, 0), -1)
            if (i > 22):
                cv2.line(image, (px, py), (px2, py2), (0, 255, 0),2)
            px2 = int(part.x)
            py2 = int(part.y)
        if (i == 0):
            px_0 = int(part.x)
            py_0 = int(part.y)
        if (i == 16):
            px_16 = int(part.x)
            py_16 = int(part.y)
        if (i == 17):
            px_17 = int(part.x)
            py_17 = int(part.y)
        if (i == 26):
            px_26 = int(part.x)
            py_26 = int(part.y)
    cv2.line(image, (px_0, py_0), (px_17, py_17), (0, 255, 0),2)
    cv2.line(image, (px_16, py_16), (px_26, py_26), (0, 255, 0),2)

def filtre_vertical(image, min_y, max_y, min_x, max_x,image_nature):
    pos_x = []
    pos = []
    for i in range(min_x - 1, max_x + 2):
        pos_x.append([])
        cpt = 0
        cpt_tempo = 0
        tempo_x = []

        for j in range(min_y - 1, max_y + 2):

            if (j < 240 and j >= 0 and i >= 0 and i < 320):
                if image[j, i][1] == 255 and image[j, i][0] == 0 and image[j, i][2] == 0:
                    image[j, i] = image_nature[j, i]
                    if cpt == 0:
                        pos_x[i - min_x].append(j)
                        tempo_x.append(j)
                        cpt = cpt + 1
                        cpt_tempo = cpt_tempo + 1
                    else:
                        if abs(j - tempo_x[cpt_tempo - 1]) > 1:
                            pos_x[i - min_x].append(j)
                            tempo_x.append(j)
                            cpt = cpt + 1
                            cpt_tempo = cpt_tempo + 1
                        else:
                            tempo_x.append(j)
                            cpt_tempo = cpt_tempo + 1

        if cpt == 2:
            for j in range(pos_x[i - min_x][0] + 1, pos_x[i - min_x][1]):
                if (j < 240 and j >= 0 and i >= 0 and i < 320):
                    pos.append([j, i])

        if cpt == 0 and i - min_x - 1 > 0 and len(pos_x[i - min_x - 1]) == 2:

            for j in range(pos_x[i - min_x - 1][0] + 1, pos_x[i - min_x - 1][1]):
                if (j < 240 and j >= 0 and i >= 0 and i < 320):
                    pos.append([j, i])


    return pos

def inserer_gps(frame,gps):

    gps=np.array(gps)
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if (gps[i,j][0]<80):
                    gps[i,j]=(255,100,51)
                    frame[i, j] = gps[i, j]


    return frame