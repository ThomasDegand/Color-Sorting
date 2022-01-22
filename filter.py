from numba import jit, cuda
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from copy import deepcopy
from cmath import phase
from colorsys import rgb_to_hsv
from colorsys import hsv_to_rgb

##r_theta = 0
##g_theta = np.deg2rad(120)
##b_theta = np.deg2rad(240)
##d = 5.85 + 8.65 + 1.25
##ar, ag, ab = 5.85/d, 8.65/d, 1.25/d
##sr, sg, sb = np.sin(r_theta), np.sin(g_theta), np.sin(b_theta)
##cr, cg, cb = np.cos(r_theta), np.cos(g_theta), np.cos(b_theta)
##A = np.array([[ar, ag, ab], [sr, sg, sb], [cr, cg, cb]], dtype=float)
##B = np.linalg.inv(A)
##print(A)
##print(B)


##Conversion rgb to thetals
@jit
def c2thetaBis(c):
    r, g, b = c
    msin = r*sr + g*sg + b*sb
    mcos = r*cr + g*cg + b*cb
    theta = np.rad2deg(phase(mcos + msin*1j))
    a = int(msin <= 0)
    theta = 360*a + theta
    return theta

@jit
def c2theta(c):
    r, g, b = c
    return int(rgb_to_hsv(r, g, b)[0]*360)

@jit
def c2lBis(c):
    r, g, b = c
    l = r*ar + g*ag + b*ab
    return l

@jit
def c2l(c):
    r, g, b = c
    return rgb_to_hsv(r, g, b)[2]

@jit
def c2sBis(c):
    r, g, b = c
    msin = r*sr + g*sg + b*sb
    mcos = r*cr + g*cg + b*cb
    norme = (msin**2 + mcos**2)**0.5
    return norme

@jit
def c2s(c):
    r, g, b = c
    return int(255*rgb_to_hsv(r, g, b)[1])

##Conversion thetals to rgb
@jit
def thetals2cBis(thetals):
    theta, l, s = thetals
    pmsin = np.sin(np.deg2rad(theta))
    pmcos = np.cos(np.deg2rad(theta))
    msin = s*pmsin
    mcos = s*pmcos
    r = l*B[0][0] + msin*B[0][1] + mcos*B[0][2]
    g = l*B[1][0] + msin*B[1][1] + mcos*B[1][2]
    b = l*B[2][0] + msin*B[2][1] + mcos*B[2][2]
    return (r, g, b)

@jit
def thetals2c(thetals):
    v = v/255
    s = s/255
    h, v, s = thetals
    return tuple(hsv_to_rgb(h, s, v))

##Obtention des couleurs de l'image
@jit
def image2colors(image):
    colors = []
    p, q, m = np.shape(image)
    imageN = np.zeros((p, q, 3), dtype=np.uint16)
    for x in range(p):
        for y in range(q):
            c = image[x][y]
            colors.append(tuple(c))
            imageN[x][y][0], imageN[x][y][1], imageN[x][y][2] = round(c2theta(c))%360, int(c2l(c)), int(c2s(c))
    return list(set(colors)), imageN


##Trie des couleurs
@jit
def ordercolorsTheta(colors):
    table = [[] for i in range(360)]
    for c in colors:
        table[round(c2theta(c))%360].append(c)
    return table

@jit
def savecolorsTheta(colors, folder):
    colorsCopy = deepcopy(colors)
    for k in range(360):
        N = len(colorsCopy[k])
        H = int(N**0.5) + 1
        for i in range(H*H - N):
            colorsCopy[k].append((0,0,0))
        for i in range(H*H):
            colorsCopy[k][i] = np.array(colorsCopy[k][i], dtype=np.uint8)
        color = np.array(colorsCopy[k])
        color = np.resize(color, (H, H, 3))
        color = Image.fromarray(color)
        color.save(folder+"/color"+str(k).zfill(3)+".png")


@jit
def ordercolorsL(colors):
    table = [[] for i in range(256)]
    for c in colors:
        table[int(c2l(c))].append(c)
    return table

@jit
def savecolorsL(colors, folder):
    colorsCopy = deepcopy(colors)
    for k in range(256):
        N = len(colorsCopy[k])
        H = int(N**0.5) + 1
        for i in range(H*H - N):
            colorsCopy[k].append((0,0,0))
        for i in range(H*H):
            colorsCopy[k][i] = np.array(colorsCopy[k][i], dtype=np.uint8)
        color = np.array(colorsCopy[k])
        color = np.resize(color, (H, H, 3))
        color = Image.fromarray(color)
        color.save(folder+"/lumen"+str(k).zfill(3)+".png")


@jit
def ordercolorsS(colors):
    table = [[] for i in range(360)]
    for c in colors:
        table[int(c2s(c))].append(c)
    return table

@jit
def savecolorsS(colors, folder):
    colorsCopy = deepcopy(colors)
    for k in range(256):
        N = len(colorsCopy[k])
        H = int(N**0.5) + 1
        for i in range(H*H - N):
            colorsCopy[k].append((0,0,0))
        for i in range(H*H):
            colorsCopy[k][i] = np.array(colorsCopy[k][i], dtype=np.uint8)
        color = np.array(colorsCopy[k])
        color = np.resize(color, (H, H, 3))
        color = Image.fromarray(color)
        color.save(folder+"/satur"+str(k).zfill(3)+".png")


##Amélioration du contraste
@jit
def minmax(colorsTheta):
    lumens = [[] for i in range(360)]
    maxN = [255 for i in range(360)]
    minN = [0 for i in range(360)]
    for k in range(360):
        for c in colorsTheta[k]:
            lumens[k].append(c2l(c))
        p = len(lumens[k])
        if len(lumens[k]) > 1:
            maxN[k] = int(max(lumens[k]))
            minN[k] = int(min(lumens[k]))
    return maxN, minN

@jit
def extend(imageN, maxN, minN):
    p, q = len(imageN), len(imageN[0])
    imageE = np.zeros((p, q, 3), dtype=np.uint16)
    for x in range(p):
        for y in range(q):
            theta, l, s = imageN[x][y][0], imageN[x][y][1], imageN[x][y][2]
            imageE[x][y][0] = imageN[x][y][0]
            maxNV, minNV = maxN[theta], minN[theta]
            imageE[x][y][1] = 255*((l - minNV)/(maxNV - minNV))
            imageE[x][y][2] = imageN[x][y][2]
    return imageE

@jit
def result(imageE):
    p, q = len(imageE), len(imageE[0])
    imageR = np.zeros((p, q, 3), dtype=np.uint8)
    for x in range(p):
        for y in range(q):
            r, g, b = thetals2c(imageE[x][y])
            imageR[x][y][0], imageR[x][y][1], imageR[x][y][2] = int(r), int(g), int(b)
    return imageR

##Instructions
image = np.array(Image.open("photosmall.jpg"))
colors, imageN = image2colors(image)

colorsTheta = ordercolorsTheta(colors)
savecolorsTheta(colorsTheta, "colors")
colorsL = ordercolorsL(colors)
savecolorsL(colorsL, "lumens")
colorsS = ordercolorsS(colors)
savecolorsS(colorsS, "saturates")

#maxN, minN = minmax(colorsTheta)
#imageE = extend(imageN, maxN, minN)
#imageR = result(imageE)
#imageR = Image.fromarray(imageR)
#imageR.save("result.png")
