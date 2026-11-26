import glob
import os
import re
import cv2
import numpy as np
import pytesseract

EXTENSOES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
PARA_LETRA = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"}
PARA_DIGITO = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "A": "4", "S": "5", "G": "6", "B": "8"}


def calcular_angulo_horizontal(rect):
    box = cv2.boxPoints(rect)
    d01 = np.linalg.norm(box[1] - box[0])
    d12 = np.linalg.norm(box[2] - box[1])
    p1, p2 = (box[0], box[1]) if d01 >= d12 else (box[1], box[2])
    if p2[0] < p1[0]:
        p1, p2 = p2, p1
    return float(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])))


def extrair_placa_mercosul(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_img, w_img = img.shape[:2]

    for s_min in (95, 65, 45):
        azul_min = np.array([96, s_min, 25])
        azul_max = np.array([134, 255, 255])
        mascara = cv2.inRange(hsv, azul_min, azul_max)

        kernel_w = max(15, int(w_img * 0.025))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 5))
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidatos = []

        for c in contornos:
            rect = cv2.minAreaRect(c)
            (_, _), (rw, rh), _ = rect
            if rw == 0 or rh == 0:
                continue
            largura = max(rw, rh)
            altura = min(rw, rh)
            razao = largura / altura
            area = largura * altura

            # Aceita até placas bem esticadas horizontalmente (ex: 5.JPG)
            if 4.0 <= razao <= 48.0 and area > 250:
                candidatos.append((area, c, rect))

        for _, c, rect in sorted(candidatos, key=lambda x: x[0], reverse=True):
            (cx, cy) = rect[0]
            angulo = calcular_angulo_horizontal(rect)
            if abs(angulo) > 35.0:
                continue

            matriz_rot = cv2.getRotationMatrix2D((cx, cy), angulo, 1.0)
            img_alinhada = cv2.warpAffine(
                img, matriz_rot, (w_img, h_img), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )

            pts = cv2.transform(c, matriz_rot)
            bx, by, bw, bh = cv2.boundingRect(pts)

            # Caracteres ficam estritamente sob a faixa azul (antes da bandeira à direita)
            x1 = max(0, bx + int(bw * 0.08))
            x2 = min(w_img, bx + int(bw * 1.02))
            y1 = max(0, by + int(bh * 1.02))
            y2 = min(h_img, by + int(bh * 4.10))

            if x2 - x1 < 40 or y2 - y1 < 15:
                continue

            crop = img_alinhada[y1:y2, x1:x2]
            cinza_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if np.percentile(cinza_crop, 85) < 75:
                continue

            return crop

    return None


def preprocessar_placa(placa):
    h, w = placa.shape[:2]
    escala = 120.0 / max(h, 1)
    redim = cv2.resize(placa, (int(w * escala), 120), interpolation=cv2.INTER_CUBIC)
    cinza = cv2.cvtColor(redim, cv2.COLOR_BGR2GRAY)

    # Realce local de contraste (ajuda placas lavadas/claras como 5.JPG e sombras fortes como 1.JPG e 6.jpg)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    realcada = clahe.apply(cinza)
    suave = cv2.GaussianBlur(realcada, (3, 3), 0)

    # Binarização adaptativa: lida com sombra atravessando metade da placa
    binaria = cv2.adaptiveThreshold(
        suave, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 45, 12
    )

    # Trabalha com texto em branco (inv = 255) para morfologia intuitiva
    inv = 255 - binaria
    h_b, w_b = inv.shape

    # 1. Remove barras horizontais da moldura (superior/inferior) sem tocar nas letras
    kernel_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, int(w_b * 0.28)), 3))
    linhas_horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel_horiz)
    inv = cv2.subtract(inv, linhas_horiz)

    # 2. Fecha os riscos finos da marca d'água ANTES de filtrar componentes conectados
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel_close)

    # 3. Filtra apenas ruídos pequenos e blocos de borda lateral (sem limite máximo rígido de altura)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    limpa_inv = np.zeros_like(inv)

    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]
        toca_borda_lateral = (x <= 1) or (x + cw >= w_b - 1)
        eh_bloco_lateral = toca_borda_lateral and (cw > 0.16 * w_b or ch < 0.35 * h_b)

        altura_ok = ch >= (0.32 * h_b)
        largura_ok = (0.012 * w_b) <= cw <= (0.24 * w_b)

        if altura_ok and largura_ok and not eh_bloco_lateral and area >= 90:
            limpa_inv[labels == i] = 255

    limpa = 255 - limpa_inv
    return cv2.copyMakeBorder(limpa, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=255)


def normalizar_mercosul(texto):
    limpo = re.sub(r"[^A-Z0-9]", "", texto.upper())
    if len(limpo) < 7:
        return limpo

    padrao = [True, True, True, False, True, False, False]  # LLLNLNN
    melhor = limpo[:7]
    menor_custo = 99

    for i in range(len(limpo) - 6):
        trecho = limpo[i : i + 7]
        custo = 0
        corrigido = []
        for ch, espera_letra in zip(trecho, padrao):
            if espera_letra:
                if ch.isalpha():
                    corrigido.append(ch)
                elif ch in PARA_LETRA:
                    corrigido.append(PARA_LETRA[ch])
                    custo += 1
                else:
                    corrigido.append(ch)
                    custo += 5
            else:
                if ch.isdigit():
                    corrigido.append(ch)
                elif ch in PARA_DIGITO:
                    corrigido.append(PARA_DIGITO[ch])
                    custo += 1
                else:
                    corrigido.append(ch)
                    custo += 5
        if custo < menor_custo:
            menor_custo = custo
            melhor = "".join(corrigido)

    return melhor


def ler_com_tesseract(binaria):
    melhor_bruto = ""
    for psm in (7, 8, 6):
        bruto = pytesseract.image_to_string(
            binaria,
            config=f"--oem 3 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        ).strip()
        limpo = re.sub(r"[^A-Z0-9]", "", bruto.upper())
        if len(limpo) == 7:
            return bruto
        if len(limpo) > len(re.sub(r"[^A-Z0-9]", "", melhor_bruto.upper())):
            melhor_bruto = bruto
    return melhor_bruto


def ordem_numerica(caminho):
    return [int(n) for n in re.findall(r"\d+", os.path.basename(caminho))] or [0]


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_GLOB = os.path.join(BASE_DIR, "dataset", "*.*")
DEBUG_DIR = os.path.join(BASE_DIR, "debug_placas")
SAIDA_TXT = os.path.join(BASE_DIR, "resultados_tesseract.txt")

os.makedirs(DEBUG_DIR, exist_ok=True)
resultados = []

caminhos = [
    c for c in sorted(glob.glob(DATASET_GLOB), key=ordem_numerica)
    if os.path.splitext(c)[1].lower() in EXTENSOES
]

for caminho in caminhos:
    img = cv2.imread(caminho)
    nome = os.path.basename(caminho)
    if img is None:
        continue

    placa = extrair_placa_mercosul(img)
    if placa is None:
        linha = f"{nome}: NAO_DETECTADA"
        resultados.append(linha)
        print(linha, flush=True)
        continue

    binaria = preprocessar_placa(placa)

    cv2.imwrite(os.path.join(DEBUG_DIR, f"crop_{nome}"), placa)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"bin_{nome}"), binaria)

    bruto = ler_com_tesseract(binaria)
    texto = normalizar_mercosul(bruto)

    linha = f"{nome}: {texto} (bruto: {bruto})"
    resultados.append(linha)
    print(linha, flush=True)

with open(SAIDA_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(resultados))