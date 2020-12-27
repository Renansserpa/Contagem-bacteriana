### Redimensionar a imagem para tamanho menor ou maior ###
import cv2
def redimensionar(imagem,porcentagem):
    width = int(imagem.shape[1] * porcentagem / 100)
    height = int(imagem.shape[0] * porcentagem / 100)
    dim = (width, height)
    resized = cv2.resize(imagem, dim, interpolation = cv2.INTER_AREA)
    return resized
### Redimensionar a imagem para quadros unidos ###
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
### Criam janelas para análise de parâmetros de cores ###
def empty(a):
    pass
def testar_cores_hsv(img,escala):
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars",640,240)
    cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
    cv2.createTrackbar("Hue Max","TrackBars",19,179,empty)
    cv2.createTrackbar("Sat Min","TrackBars",110,255,empty)
    cv2.createTrackbar("Sat Max","TrackBars",240,255,empty)
    cv2.createTrackbar("Val Min","TrackBars",153,255,empty)
    cv2.createTrackbar("Val Max","TrackBars",255,255,empty)
    while True:
        img_hsv = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2HSV)
        h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
        h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
        s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
        s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
        v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
        v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
        lower = np.array([h_min,s_min,v_min])
        upper = np.array([h_max,s_max,v_max])
        mask = cv2.inRange(img_hsv,lower,upper)
        imgResult = cv2.bitwise_and(img,img,mask=mask)
        imgStack = stackImages(escala,([img,img_hsv],[mask,imgResult]))
        cv2.imshow("Stacked Images", imgStack)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
def testar_cores_rgb(img,escala):
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars",640,240)
    cv2.createTrackbar("R Min","TrackBars",0,255,empty)
    cv2.createTrackbar("R Max","TrackBars",19,255,empty)
    cv2.createTrackbar("G Min","TrackBars",110,255,empty)
    cv2.createTrackbar("G Max","TrackBars",240,255,empty)
    cv2.createTrackbar("B Min","TrackBars",153,255,empty)
    cv2.createTrackbar("B Max","TrackBars",255,255,empty)
    while True:
        img_rgb = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2RGB)
        r_min = cv2.getTrackbarPos("R Min","TrackBars")
        r_max = cv2.getTrackbarPos("R Max", "TrackBars")
        g_min = cv2.getTrackbarPos("G Min", "TrackBars")
        g_max = cv2.getTrackbarPos("G Max", "TrackBars")
        b_min = cv2.getTrackbarPos("B Min", "TrackBars")
        b_max = cv2.getTrackbarPos("B Max", "TrackBars")
        lower = np.array([r_min,g_min,b_min])
        upper = np.array([r_max,r_max,b_max])
        mask = cv2.inRange(img_rgb,lower,upper)
        imgResult = cv2.bitwise_and(img,img,mask=mask)
        imgStack = stackImages(escala,([img,img_rgb],[mask,imgResult]))
        cv2.imshow("Stacked Images", imgStack)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
#Exemplo de uso
#import cv2
#import numpy as np
#import os
#img=cv2.imread(os.getcwd()+ '\\imagens\\20_3.JPG')
#img= redimensionar(img,40)
#cv2.imshow('imagem',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#testar_cores_rgb(img,0.2)