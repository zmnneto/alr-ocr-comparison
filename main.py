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
    h, w = placa.shape[:2]
    escala = 120.0 / max(h, 1)
    redim = cv2.resize(placa, (int(w * escala), 120), interpolation=cv2.INTER_CUBIC)

    cinza = cv2.cvtColor(redim, cv2.COLOR_BGR2GRAY)

    # Remoção de sombra: divide a imagem pelo fundo estimado com blur forte
    fundo = cv2.GaussianBlur(cinza, (0, 0), sigmaX=31, sigmaY=31)
    sem_sombra = cv2.divide(cinza, fundo, scale=255)

    # Suavização leve + binarização de Otsu
    blur = cv2.GaussianBlur(sem_sombra, (3, 3), 0)
    _, binaria = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Fecha riscos brancos da marca d'água dentro das letras pretas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)

    # Margem branca para o Tesseract não ignorar o primeiro/último caractere
    return cv2.copyMakeBorder(binaria, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)


def ordem_numerica(caminho):
    numeros = re.findall(r"\d+", os.path.basename(caminho))
    return int(numeros[0]) if numeros else 0


os.makedirs("debug_placas", exist_ok=True)
resultados = []

for caminho in sorted(glob.glob("dataset/*.*"), key=ordem_numerica):
    img = cv2.imread(caminho)
    nome = os.path.basename(caminho)
    placa = extrair_placa_mercosul(img)

    if placa is None:
        linha = f"{nome}: NAO_DETECTADA"
        resultados.append(linha)
        print(linha, flush=True)
        continue

    binaria = preprocessar_placa(placa)

    cv2.imwrite(f"debug_placas/crop_{nome}", placa)
    cv2.imwrite(f"debug_placas/bin_{nome}", binaria)

    texto = pytesseract.image_to_string(
        binaria,
        config="--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    ).strip()

    linha = f"{nome}: {texto}"
    resultados.append(linha)
    print(linha, flush=True)

with open("resultados_tesseract.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(resultados))
