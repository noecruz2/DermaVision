from PIL import Image
import numpy as np

UMBRAL_BINARIO = 145
MAX_VALOR = 255

#Funcion para convertir la imagen cargada en RGB a escala de grises
# Se define la funcion y se manda la imagen como parametro
# La imagen se convierte en un array, se recorre cada fila de la imagen
# Se crea una nueva fila para la imagen en escala de grises
# Recorre cada pixel de la fila, obtiene los valores RGB y los guarda en variables
# Se calcula el valor de gris usando la formula de luminosidad
# Agregar el valor de gris a la nueva fila, se agrega la fila a la imagen en escala de grises
#Se retorna la imagen en escala de grises como un array de numpy

def convertir_a_grises(img):
    
    imagen_gris = [] 

    for fila in img:
        nueva_fila = [] 
        for pixel in fila:
            r, g, b = pixel
            gris = int(0.21 * r + 0.72 * g + 0.07 * b)
            nueva_fila.append(gris)
        imagen_gris.append(nueva_fila) 
    return np.array(imagen_gris)

#Funcion para convertir la imagen en escala de grises a una imagen binaria
# Se define la funcion y se manda la imagen como parametro
# La imagen se convierte en un array, se recorre cada fila de la imagen
# Se crea una nueva fila para la imagen binaria
# Recorre cada pixel de la fila, obtiene los valores y si es mayor al umbral se asigna blanco
# Si es menor o igual al umbral se asigna negro
# Agregar el valor de la fila binaria a la imagen binaria
#Se retorna la imagen binaria como un array de numpy
def convertir_a_binario(img_gris):

    imagen_binaria = []
    
    for fila in img_gris:
        nueva_fila = [] 
        for valor in fila: 
            if valor > UMBRAL_BINARIO:
                nueva_fila.append(MAX_VALOR)
            else:
                nueva_fila.append(0)
        imagen_binaria.append(nueva_fila)
    
    return np.array(imagen_binaria)

#Funcion para quedarse con el componente mas grande de la imagen binaria
#Se define la función y se manda como parametro la imagen binaria
#Se obtiene el alto y ancho de la imagen binaria 
#Se crea una matriz en false, al visitar los pixeles cambia a true
#mejor_pixels almacena los pixeles del componente mas grande
#Se recorre cada fila y columna de la imagen binaria
#Si el pixel ya fue visitado o es blanco se continua al siguiente pixel
#Se inicia un flood fill en los pixeles, almacenando 
def componente_mas_grande(img_binaria):
    alto, ancho = img_binaria.shape
    visitado = np.zeros((alto, ancho), dtype=bool) #
    mejor_pixels = []

    for y in range(alto):
        for x in range(ancho):
            if visitado[y][x]:
                continue
            if img_binaria[y][x] != 0:
                continue

            # flood fill en negros
            stack = [(y, x)] #esto sirve para almacenar los pixeles a visitar
            visitado[y][x] = True
            pixeles = []

            while stack:
                cy, cx = stack.pop()
                pixeles.append((cy, cx))

                for ny, nx in ((cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)):
                    if 0 <= ny < alto and 0 <= nx < ancho and not visitado[ny][nx]:
                        if img_binaria[ny][nx] == 0:
                            visitado[ny][nx] = True
                            stack.append((ny, nx))

            if len(pixeles) > len(mejor_pixels):
                mejor_pixels = pixeles

    if not mejor_pixels:
        return img_binaria

    salida = np.full_like(img_binaria, MAX_VALOR)
    for y, x in mejor_pixels:
        salida[y][x] = 0

    return salida

#Funcion para recortar la imagen binaria alrededor del unar
#Se define la funcion y se manda la imagen binaria como parametro
#Se crean dos listas para almacenar las coordenadas de los pixeles del lunar
#Se recorre cada fila y columna de la imagen binaria

def recortar_lunar(img_binaria):
    
    ys, xs = [], [] #Listas para almacenar las coordenadas de los pixeles del lunar

    for y in range(img_binaria.shape[0]): #Recorre cada fila de la imagen binaria
        for x in range(img_binaria.shape[1]): #Recorre cada columna de la imagen binaria
            if img_binaria[y][x] == 0:  #Si el pixel es negro (lunar)
                ys.append(y) #Agrega la coordenada y a la lista
                xs.append(x) #Agrega la coordenada x a la lista 
                
    if not ys or not xs:
        return img_binaria

    min_y, max_y = min(ys), max(ys) #Obtiene las coordenadas minimas y maximas en y
    min_x, max_x = min(xs), max(xs) #Obtiene las coordenadas minimas y maximas en x

    return img_binaria[min_y:max_y+1, min_x:max_x+1] #Retorna la imagen binaria recortada alrededor del lunar
 
def rellenar_huecos(img_binaria):
    alto, ancho = img_binaria.shape

    # matriz para marcar fondo real
    visitado = np.zeros((alto, ancho), dtype=bool)

    # pila para flood fill
    stack = []

    # 1. agregar píxeles blancos del borde
    for x in range(ancho):
        if img_binaria[0][x] == 255:
            stack.append((0, x))
        if img_binaria[alto-1][x] == 255:
            stack.append((alto-1, x))

    for y in range(alto):
        if img_binaria[y][0] == 255:
            stack.append((y, 0))
        if img_binaria[y][ancho-1] == 255:
            stack.append((y, ancho-1))

    # 2. flood fill desde el borde
    while stack:
        y, x = stack.pop()

        if y < 0 or y >= alto or x < 0 or x >= ancho:
            continue
        if visitado[y][x]:
            continue
        if img_binaria[y][x] != MAX_VALOR:
            continue

        visitado[y][x] = True

        # vecinos 4
        stack.extend([
            (y-1, x),
            (y+1, x),
            (y, x-1),
            (y, x+1)
        ])

    # 3. rellenar huecos
    resultado = img_binaria.copy()

    for y in range(alto):
        for x in range(ancho):
            if img_binaria[y][x] == MAX_VALOR and not visitado[y][x]:
                resultado[y][x] = 0  # es hueco → lunar

    return resultado

def erosionar(img_binaria):
    alto, ancho = img_binaria.shape
    salida = img_binaria.copy()

    for y in range(1, alto - 1):
        for x in range(1, ancho - 1):
            vecinos = img_binaria[y-1:y+2, x-1:x+2]
            salida[y][x] = 0 if np.all(vecinos == 0) else MAX_VALOR

    return salida

def dilatar(img_binaria):
    alto, ancho = img_binaria.shape
    salida = img_binaria.copy()

    for y in range(1, alto - 1):
        for x in range(1, ancho - 1):
            vecinos = img_binaria[y-1:y+2, x-1:x+2]
            salida[y][x] = 0 if np.any(vecinos == 0) else MAX_VALOR

    return salida

def contorno_morfologico(img_binaria):
    dilatada = dilatar(img_binaria)
    erosionada = erosionar(img_binaria)
    contorno = np.where(dilatada != erosionada, MAX_VALOR, 0).astype(np.uint8)
    return contorno

def procesar_imagen(ruta):
    #abrir la imagen con pillow
    img = Image.open(ruta)
    img.show()

    #convertir la imagen a un array con numpy
    img_array = np.array(img)

    #convertir la imagen a escala de grises
    gris_manual = convertir_a_grises(img_array)
    print(gris_manual.shape)
    print(gris_manual.min(), gris_manual.max())
    img_gris = Image.fromarray(gris_manual.astype(np.uint8))
    img_gris.show()

    #convertir la imagen en escala de grises a máscara binaria
    binaria_manual = convertir_a_binario(gris_manual)
    print(binaria_manual.shape)
    print(binaria_manual.min(), binaria_manual.max())
    img_bin = Image.fromarray(binaria_manual.astype(np.uint8))
    img_bin.show()

    #quedarse con el componente negro más grande para evitar ruido
    binaria_filtrada = componente_mas_grande(binaria_manual)

    #recortar la máscara binaria
    binaria_recorte = recortar_lunar(binaria_filtrada)
    img_bin_recorte = Image.fromarray(binaria_recorte.astype(np.uint8))
    img_bin_recorte.show()

    binaria_limpia = rellenar_huecos(binaria_recorte)
    Image.fromarray(binaria_limpia.astype(np.uint8)).show()

    contorno = contorno_morfologico(binaria_limpia)
    Image.fromarray(contorno).show()

    print(img_array.shape)
    print(img_array)


if __name__ == "__main__":
    procesar_imagen("./DataSet/tatuaje.jpeg")
