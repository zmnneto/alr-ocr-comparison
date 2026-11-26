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
    """Calcula o ângulo do lado maior do retângulo usando os vértices (independente da versão do OpenCV)."""
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

    # Tenta primeiro azul bem saturado (evita pegar branco na sombra); se falhar, relaxa a saturação
    for s_min in (95, 65):
        azul_min = np.array([98, s_min, 30])
        azul_max = np.array([132, 255, 255])
        mascara = cv2.inRange(hsv, azul_min, azul_max)

        kernel_w = max(15, int(w_img * 0.02))
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

            # Faixa azul Mercosul real tem proporção bem alongada (~6 a 18)
            if 4.5 <= razao <= 28.0 and area > 300:
                candidatos.append((area, c, rect))

        # Testa do maior candidato para o menor, validando se existe área clara de placa logo abaixo
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

            # Recorte mais justo para evitar a moldura preta inferior e direita
            x1 = max(0, bx + int(bw * 0.09))
            x2 = min(w_img, bx + int(bw * 1.03))
            y1 = max(0, by + int(bh * 1.05))
            y2 = min(h_img, by + int(bh * 3.85))

            if x2 - x1 < 40 or y2 - y1 < 15:
                continue

            crop = img_alinhada[y1:y2, x1:x2]
            # Valida que o recorte não é para-choque preto/escapamento (exige fundo minimamente claro)
            cinza_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if np.percentile(cinza_crop, 85) < 85:
                continue

            return crop

    return None


def preprocessar_placa(placa):
    h, w = placa.shape[:2]
    escala = 120.0 / max(h, 1)
    redim = cv2.resize(placa, (int(w * escala), 120), interpolation=cv2.INTER_CUBIC)
    cinza = cv2.cvtColor(redim, cv2.COLOR_BGR2GRAY)

    # Remoção de sombra por divisão de fundo
    fundo = cv2.GaussianBlur(cinza, (0, 0), sigmaX=51, sigmaY=51)
    sem_sombra = cv2.divide(cinza, fundo, scale=255)

    blur = cv2.GaussianBlur(sem_sombra, (3, 3), 0)
    _, binaria = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove barra preta inferior, bloco lateral direito e ruídos por análise de componentes conectados
    inv = 255 - binaria
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    limpa = np.full_like(binaria, 255)
    h_b, w_b = binaria.shape

    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]
        # Mantém apenas componentes com dimensões compatíveis com caracteres da placa
        altura_ok = (0.30 * h_b) <= ch <= (0.92 * h_b)
        largura_ok = (0.015 * w_b) <= cw <= (0.22 * w_b)
        nao_eh_moldura = not (cw > 0.35 * w_b)  # descarta barra horizontal inferior
        if altura_ok and largura_ok and nao_eh_moldura and area >= 80:
            limpa[labels == i] = 0

    # Fecha riscos brancos finos da marca d'água dentro das letras pretas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    limpa = cv2.morphologyEx(limpa, cv2.MORPH_OPEN, kernel)

    return cv2.copyMakeBorder(limpa, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)


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

    bruto = pytesseract.image_to_string(
        binaria,
        config="--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    ).strip()
    texto = normalizar_mercosul(bruto)

    linha = f"{nome}: {texto} (bruto: {bruto})"
    resultados.append(linha)
    print(linha, flush=True)

with open(SAIDA_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(resultados))