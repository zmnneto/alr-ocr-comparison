import glob
import os
import re
import cv2
import numpy as np
import pytesseract


def extrair_placa_mercosul(img):
    """
    Localiza a placa pela faixa azul superior do padrão Mercosul no espaço HSV,
    corrige a inclinação e recorta diretamente a região dos 7 caracteres.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Máscara para o azul da faixa Mercosul (cobre também trechos com sombra)
    azul_min = np.array([98, 70, 40])
    azul_max = np.array([132, 255, 255])
    mascara = cv2.inRange(hsv, azul_min, azul_max)

    # Fecha pequenas falhas na faixa azul (ex.: texto "BRASIL" em branco no meio)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None

    # Seleciona o maior contorno azul com formato horizontal de cabeçalho de placa
    melhor = None
    maior_area = 0
    for c in contornos:
        rect = cv2.minAreaRect(c)
        (_, _), (rw, rh), _ = rect
        if rw == 0 or rh == 0:
            continue
        largura = max(rw, rh)
        altura = min(rw, rh)
        razao = largura / altura
        area = largura * altura

        if 4.0 <= razao <= 40.0 and area > maior_area and area > 300:
            maior_area = area
            melhor = (c, rect)

    if melhor is None:
        return None

    c, rect = melhor
    (cx, cy), (rw, rh), angulo = rect

    # Ajusta o ângulo para alinhar a faixa azul perfeitamente na horizontal
    if rw < rh:
        angulo += 90.0

    # Rotaciona a imagem inteira em torno do centro da faixa azul
    h_img, w_img = img.shape[:2]
    matriz_rot = cv2.getRotationMatrix2D((cx, cy), angulo, 1.0)
    img_alinhada = cv2.warpAffine(
        img, matriz_rot, (w_img, h_img), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    # Transforma o contorno da faixa azul para as coordenadas alinhadas
    pts = cv2.transform(c, matriz_rot)
    bx, by, bw, bh = cv2.boundingRect(pts)

    # Recorta a área dos caracteres logo abaixo da faixa azul:
    # - pula ~10% à esquerda para excluir o "BR"
    # - estende ~11% à direita para cobrir o espaço abaixo da bandeira do Brasil
    x1 = max(0, bx + int(bw * 0.10))
    x2 = min(w_img, bx + int(bw * 1.11))
    y1 = max(0, by + int(bh * 1.05))
    y2 = min(h_img, by + int(bh * 4.55))

    if x2 <= x1 or y2 <= y1:
        return None

    return img_alinhada[y1:y2, x1:x2]


def preprocessar_placa(placa):
    """
    Corrige gradientes de sombra, realça contraste e binariza para o Tesseract.
    """
    # Normaliza a altura para 120px mantendo a proporção (estabiliza o tamanho da fonte)
    h, w = placa.shape[:2]
    escala = 120.0 / max(h, 1)
    redim = cv2.resize(placa, (int(w * escala), 120), interpolation=cv2.INTER_CUBIC)

    cinza = cv2.cvtColor(redim, cv2.COLOR_BGR2GRAY)

    # Remoção de sombra: divide a