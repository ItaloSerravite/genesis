import genesis as gs
import numpy as np
from PIL import Image
import cv2 as cv

altura = 1024
largura = 1024
angulo = 60.0

gs.init(backend=gs.cpu)

scene = gs.Scene(
    show_viewer    = True
)

textura = gs.surfaces.Rough(
    roughness = 1.0,
    diffuse_texture=gs.textures.ImageTexture(image_path="textures/col.jpg"),
    #roughness_texture=gs.textures.ImageTexture(image_path="textures/rough.jpg", encoding="linear"),
    normal_texture=gs.textures.ImageTexture(image_path="textures/norgl.exr", encoding="linear"),
)

cadeira=scene.add_entity(
    morph=gs.morphs.Mesh(file="objs/cadeira.obj",pos = (0,0,0), file_meshes_are_zup=False),
    surface=textura
)
cadeira2=scene.add_entity(
    morph=gs.morphs.Mesh(file="objs/cadeira.obj",pos = (0,-1,0), file_meshes_are_zup=False),
    surface=textura
)
mesa=scene.add_entity(
    morph=gs.morphs.Mesh(file="objs/mesa.obj",pos = (0,1,0), file_meshes_are_zup=False,),
    surface=textura
)
floor=scene.add_entity(
    morph=gs.morphs.Plane(
        pos=(0.0, 0.0, 0.0)
    )
)

cam1 = scene.add_camera(
    res=(altura, largura),
    pos=(3, 1, 2),
    lookat=(0, 0, 0.5),
    fov = angulo,
    model="pinhole",
)
cam2 = scene.add_camera(
    res=(altura, largura),
    pos=(4, 3, 2),
    lookat=(0, 0, 0),
    fov = angulo,
    model="pinhole",
)

#coloca um pondo da imagem nas coordenadas do mundo
def _2D_to_3D_to_world(u,v,intrinsic, transform,depth):
    cx = intrinsic[0][2]
    cy = intrinsic[1][2]
    fx = intrinsic[0][0]
    fy = intrinsic[1][1]
    #Z = depth[v][u]/np.sqrt((((u-cx)/fx)**2) + (((v-cy)/fy)**2)+1)
    Z = depth[v][u]
    X = ((u - cx)/fx)*Z
    Y = ((v - cy)/fy)*Z
    temp = np.array([X, Y, Z,1])
    resultado = (transform@temp)
    return resultado

#coloca os pontos da imagem nas coordenadas do mundo
def image_points_to_3d_coodinate(_altura, _largura, intrinsic, transform,depth):
    imagem_mundo = np.zeros((_altura, _largura, 4))
    for i in range(_altura):
        for j in range(_largura):
            imagem_mundo[i][j] = _2D_to_3D_to_world(i,j, intrinsic, transform, depth)
    return imagem_mundo

   
def find_correspondences(epsilon, img1, img2, matriz_posicao1, matriz_posicao2, intrinsic2, extrinsic2):
    
    imagem_final = np.hstack((img1, img2))
    _altura = img1.shape[0]
    _largura = img1.shape[1]
    ponto1 = np.array([0,0])
    np.random.seed(150)
    extrinsic2 = extrinsic2[:3, :]
    projetiva2 = intrinsic2@extrinsic2
    ponto1[0] = 0
    ponto1[1] = 0
    
    for i in range(1,_altura,4):
        for j in range(1,_largura,4):
            ponto1[0] = i
            ponto1[1] = j
            x = matriz_posicao1[i][j][0]
            y = matriz_posicao1[i][j][1]
            z = matriz_posicao1[i][j][2]

            if(z < 0.05 or z > 10 or x > 10 or x < -10 or y > 10 or y < -10):
                continue    

            ponto_imagem_2 = projetiva2@matriz_posicao1[i][j]
            ponto_imagem_2[0] /= ponto_imagem_2[2]
            ponto_imagem_2[1] /= ponto_imagem_2[2]
            ponto_imagem_2[0] = int(np.round(ponto_imagem_2[0]))
            ponto_imagem_2[1] = int(np.round(ponto_imagem_2[1]))
            if(ponto_imagem_2[0] < _altura and ponto_imagem_2[0] >=0 and ponto_imagem_2[1] < _largura and ponto_imagem_2[1] >= 0):
                distancia = np.linalg.norm(matriz_posicao1[i][j] - matriz_posicao2[int(ponto_imagem_2[0])][int(ponto_imagem_2[1])])
                if(distancia < epsilon):
                    p1 = ( int(i), int(j) )
                    p2 = ( int(ponto_imagem_2[0] + _largura), int(ponto_imagem_2[1]) )
                    cor = (np.random.randint(255),np.random.randint(255),np.random.randint(255))
                    if(np.random.randint(15) == 1):
                        cv.line(imagem_final, p1,p2,cor, 1)

    return imagem_final

#-------------------------------------
scene.build()

rgb1, depth1, seg1, normal1 = cam1.render(rgb=True, depth=True, segmentation=True, normal= True)
rgb2, depth2, seg2, normal2 = cam2.render(rgb=True, depth=True, segmentation=True, normal= True)

img1 = Image.fromarray(rgb1)
img1.save('rgb1.png')
np.save("depth_map1.npy", depth1)
img2 = Image.fromarray(rgb2)
img2.save('rgb2.png')
np.save("depth_map2.npy", depth2)

intrinsic1 = cam1.intrinsics
extrinsic1 = cam1.extrinsics
transform1 = np.linalg.inv(extrinsic1)
intrinsic2 = cam2.intrinsics
extrinsic2 = cam2.extrinsics
transform2 = np.linalg.inv(extrinsic2)


matriz_posicao1 = image_points_to_3d_coodinate(altura, largura, intrinsic1, transform1,depth1)
matriz_posicao2 = image_points_to_3d_coodinate(altura, largura, intrinsic2, transform2,depth2)

img3 = Image.fromarray(find_correspondences(0.01, rgb1, rgb2, matriz_posicao1, matriz_posicao2, intrinsic2, extrinsic2))
img3.save('correspondences.png')