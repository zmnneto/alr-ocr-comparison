import glob
import os
import re
from collections import defaultdict
import cv2
import numpy as np
import pytesseract

EXTENSOES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
PADRAO_MERCOSUL = [True, True, True, False, True, False, False]  # LLLNLNN
PARA_LETRA = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}
PARA_DIGITO = {
    "O": "0", "Q": "0", "D": "0", "U": "0",
    "I": "1", "L": "1", "J": "1",
    "Z": "2", "A": "4", "S": "5", "G": "6", "T": "7", "B": "8",
}


def calcular_angulo_horizontal(rect):
    box = cv2.boxPoints(rect)
    d01 = np.linalg.norm(box[1] - box[0])
    d12 = np.linalg.norm(box[2] - box[1])
    p1, p2 = (box[0], box[1]) if d01 >= d12 else (box[1], box[2])
    if p2[0] < p1[0]:
        p1, p2 = p2, p1
    return float(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])))


def remover_moldura_cinza(crop):
    """Varre de dentro da placa para fora e corta na borda preta interna, mesmo com fundo claro fora."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    if h < 15 or w < 40:
        return crop

    p75_linhas = np.percentile(g, 75, axis=1)

    # A regiao [12%, 48%] do recorte (logo abaixo da faixa azul) esta sempre dentro da area branca
    y_ini = int(h * 0.12)
    y_fim = max(y_ini + 1, int(h * 0.48))
    y_mid = y_ini + int(np.argmax(p75_linhas[y_ini:y_fim]))
    ref_v = float(p75_linhas[y_mid])
    limiar_v = min(85.0, ref_v * 0.52)

    topo = 0
    for y in range(y_mid, -1, -1):
        if p75_linhas[y] < limiar_v:
            topo = y + 1
            break

    base = h
    for y in range(y_mid, h):
        if p75_linhas[y] < limiar_v:
            base = y
            break

    if base - topo < 12:
        topo, base = 0, h

    g_v = g[topo:base, :]
    h_v, w_v = g_v.shape

    # Referencia local por lateral para suportar meia-placa na sombra
    esq_ini, esq_fim = int(w_v * 0.08), max(int(w_v * 0.08) + 1, w_v // 3)
    dir_ini, dir_fim = int(w_v * 0.58), max(int(w_v * 0.58) + 1, int(w_v * 0.88))
    ref_esq = float(np.percentile(g_v[:, esq_ini:esq_fim], 90))
    ref_dir = float(np.percentile(g_v[:, dir_ini:dir_fim], 90))
    limiar_esq = min(85.0, ref_esq * 0.52)
    limiar_dir = min(85.0, ref_dir * 0.52)

    p80_cols = np.percentile(g_v, 80, axis=0)
    faixa_h = max(2, h_v // 8)
    topo_col = g_v[:faixa_h, :].mean(axis=0)
    base_col = g_v[-faixa_h:, :].mean(axis=0)

    # Moldura lateral e escura no topo, meio e base; letras tem margem branca no topo/base
    esq = 0
    for x in range(int(w_v * 0.16), -1, -1):
        if p80_cols[x] < limiar_esq and topo_col[x] < limiar_esq * 1.25 and base_col[x] < limiar_esq * 1.25:
            esq = x + 1
            break

    dir_ = w_v
    for x in range(int(w_v * 0.78), w_v):
        if p80_cols[x] < limiar_dir and topo_col[x] < limiar_dir * 1.25 and base_col[x] < limiar_dir * 1.25:
            dir_ = x
            break

    if dir_ - esq < 30:
        esq, dir_ = 0, w_v

    pad_t = max(2, int(h_v * 0.03)) if topo > 0 else 1
    pad_b = max(2, int(h_v * 0.03)) if base < h else 1
    pad_l = max(2, int((dir_ - esq) * 0.01)) if esq > 0 else 0
    pad_r = max(2, int((dir_ - esq) * 0.01)) if dir_ < w_v else 0

    y1 = min(base - 10, topo + pad_t)
    y2 = max(y1 + 10, base - pad_b)
    x1 = min(dir_ - 25, esq + pad_l)
    x2 = max(x1 + 25, dir_ - pad_r)

    return crop[y1:y2, x1:x2]


def extrair_candidatos_placa(img, max_candidatos=3):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(img)
    h_img, w_img = img.shape[:2]

    azul_bgr = ((b.astype(np.int16) - r.astype(np.int16)) >= 20) & (
        (b.astype(np.int16) - g.astype(np.int16)) >= 5
    )

    candidatos_brutos = []
    for s_min in (95, 65):
        azul_min = np.array([96, s_min, 18])
        azul_max = np.array([134, 255, 255])
        mascara = cv2.inRange(hsv, azul_min, azul_max)
        mascara = cv2.bitwise_and(mascara, (azul_bgr.astype(np.uint8) * 255))

        kernel_w = max(25, int(w_img * 0.055))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contornos:
            rect = cv2.minAreaRect(c)
            (_, _), (rw, rh), _ = rect
            if rw == 0 or rh == 0:
                continue
            largura = max(rw, rh)
            altura = min(rw, rh)
            razao = largura / altura
            area = largura * altura
            if 4.5 <= razao <= 48.0 and area > 250:
                candidatos_brutos.append((area, largura, altura, rect))

    crops = []
    centros_vistos = []

    for _, bw, bh, rect in sorted(candidatos_brutos, key=lambda x: x[0], reverse=True):
        (cx, cy) = rect[0]
        if any(abs(cx - ox) < 35 and abs(cy - oy) < 25 for ox, oy in centros_vistos):
            continue

        angulo = calcular_angulo_horizontal(rect)
        if abs(angulo) > 35.0:
            continue

        matriz_rot = cv2.getRotationMatrix2D((cx, cy), angulo, 1.0)
        img_alinhada = cv2.warpAffine(
            img, matriz_rot, (w_img, h_img), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

        bx = cx - bw / 2.0
        by = cy - bh / 2.0

        # A faixa azul cobre ~88% da largura (sem a bandeira), entao 1.16 * bw garante o ultimo digito
        x1 = max(0, int(bx + bw * 0.08))
        x2 = min(w_img, int(bx + bw * 1.16))
        y1 = max(0, int(by + bh * 1.05))
        y2 = min(h_img, int(by + bh * 4.55))

        if x2 - x1 < 40 or y2 - y1 < 15:
            continue

        crop = img_alinhada[y1:y2, x1:x2]
        cinza_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if np.percentile(cinza_crop, 85) < 85 or np.std(cinza_crop) < 15:
            continue

        centros_vistos.append((cx, cy))
        crops.append(remover_moldura_cinza(crop))
        if len(crops) >= max_candidatos:
            break

    return crops


def limpar_binaria(binaria):
    inv = 255 - binaria
    h, w = inv.shape

    media_linhas = inv.mean(axis=1)
    topo = 0
    while topo < int(h * 0.22) and media_linhas[topo] > 140:
        topo += 1
    base = h
    while base > int(h * 0.78) and media_linhas[base - 1] > 140:
        base -= 1
    inv = inv[topo:base, :]

    media_cols = inv.mean(axis=0)
    esq = 0
    while esq < int(w * 0.12) and media_cols[esq] > 140:
        esq += 1
    dir_ = w
    while dir_ > int(w * 0.88) and media_cols[dir_ - 1] > 140:
        dir_ -= 1
    inv = inv[:, esq:dir_]

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    h_c, w_c = inv.shape
    limpa_inv = np.zeros_like(inv)

    for i in range(1, num_labels):
        cx, cy, cw, ch, area = stats[i]
        if ch < int(0.28 * h_c) or area < 70:
            continue
        if cw > int(0.32 * w_c) and ch < int(0.38 * h_c):
            continue
        # Remove resto de "BR" no canto inferior esquerdo
        if cx < int(0.10 * w_c) and ch < int(0.55 * h_c) and cy > int(0.38 * h_c):
            continue
        limpa_inv[labels == i] = 255

    ys, xs = np.where(limpa_inv > 0)
    if ys.size > 0 and (ys.max() - ys.min()) >= 16 and (xs.max() - xs.min()) >= 35:
        limpa_inv = limpa_inv[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    # Escala para altura ~48px (faixa otima para o LSTM do Tesseract 5)
    h_ink, w_ink = limpa_inv.shape
    if h_ink > 0:
        escala = 48.0 / h_ink
        novo_w = max(20, int(round(w_ink * escala)))
        interp = cv2.INTER_AREA if escala < 1 else cv2.INTER_CUBIC
        limpa_inv = cv2.resize(limpa_inv, (novo_w, 48), interpolation=interp)
        _, limpa_inv = cv2.threshold(limpa_inv, 127, 255, cv2.THRESH_BINARY)

    limpa = 255 - limpa_inv
    return cv2.copyMakeBorder(limpa, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=255)


def gerar_binarizacoes(placa):
    redim = cv2.resize(placa, (420, 96), interpolation=cv2.INTER_CUBIC)
    cinza = cv2.cvtColor(redim, cv2.COLOR_BGR2GRAY)

    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    fundo = cv2.morphologyEx(cinza, cv2.MORPH_CLOSE, kernel_bg)
    norm = cv2.divide(cinza, np.maximum(fundo, 1), scale=255)

    p2, p98 = np.percentile(norm, (2, 98))
    if p98 > p2 + 10:
        norm = np.clip((norm.astype(np.float32) - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)

    sem_marca_leve = cv2.morphologyEx(
        norm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    sem_marca_forte = cv2.morphologyEx(
        norm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    sem_marca_extra = cv2.morphologyEx(
        norm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    )

    suave_leve = cv2.GaussianBlur(sem_marca_leve, (3, 3), 0)
    suave_forte = cv2.GaussianBlur(sem_marca_forte, (3, 3), 0)
    suave_extra = cv2.GaussianBlur(sem_marca_extra, (3, 3), 0)

    _, bin_forte = cv2.threshold(suave_forte, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bin_extra = cv2.threshold(suave_extra, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bin_leve = cv2.threshold(suave_leve, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_adapt = cv2.adaptiveThreshold(
        suave_forte, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 15
    )

    # Retorna (imagem_binaria, peso_para_votacao)
    return [
        (limpar_binaria(bin_forte), 1.3),
        (limpar_binaria(bin_extra), 1.25),
        (limpar_binaria(bin_leve), 1.0),
        (limpar_binaria(bin_adapt), 1.0),
    ]


def normalizar_e_pontuar_mercosul(texto):
    limpo = re.sub(r"[^A-Z0-9]", "", texto.upper())
    if len(limpo) >= 9 and limpo.startswith("BR"):
        limpo = limpo[2:]

    if len(limpo) < 7:
        return limpo, 50 + (7 - len(limpo)) * 10

    melhor = limpo[:7]
    menor_custo = 999

    for i in range(len(limpo) - 6):
        trecho = limpo[i : i + 7]
        custo = abs(len(limpo) - 7)
        corrigido = []
        for ch, espera_letra in zip(trecho, PADRAO_MERCOSUL):
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

    return melhor, menor_custo


def ler_por_segmentacao(binaria):
    """Fallback: segmenta os 7 caracteres (dividindo letras coladas) e le com whitelist por posicao."""
    inv = 255 - binaria
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)

    caixas = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if h >= 22 and w >= 5 and area >= 40:
            caixas.append([x, y, x + w, y + h])

    if not caixas:
        return ""

    caixas.sort(key=lambda b: b[0])
    mescladas = [caixas[0]]
    for b in caixas[1:]:
        ant = mescladas[-1]
        sobrep = min(ant[2], b[2]) - max(ant[0], b[0])
        menor_w = min(ant[2] - ant[0], b[2] - b[0])
        if menor_w > 0 and sobrep / menor_w > 0.35:
            ant[0] = min(ant[0], b[0])
            ant[1] = min(ant[1], b[1])
            ant[2] = max(ant[2], b[2])
            ant[3] = max(ant[3], b[3])
        else:
            mescladas.append(b)

    # Divide caixas muito largas (2 caracteres colados) no vale minimo de tinta
    while 4 <= len(mescladas) < 7:
        larguras = [b[2] - b[0] for b in mescladas]
        med_w = float(np.median(larguras))
        idx_max = int(np.argmax(larguras))
        if larguras[idx_max] < med_w * 1.45:
            break
        x1, y1, x2, y2 = mescladas[idx_max]
        perfil = inv[y1:y2, x1:x2].sum(axis=0)
        m1 = max(1, int((x2 - x1) * 0.30))
        m2 = max(m1 + 1, int((x2 - x1) * 0.70))
        corte = x1 + m1 + int(np.argmin(perfil[m1:m2]))
        mescladas[idx_max : idx_max + 1] = [[x1, y1, corte, y2], [corte, y1, x2, y2]]

    if len(mescladas) > 7:
        mescladas = sorted(mescladas, key=lambda b: (b[3] - b[1]) * (b[2] - b[0]), reverse=True)[:7]
        mescladas.sort(key=lambda b: b[0])

    if len(mescladas) != 7:
        return ""

    chars = []
    for (x1, y1, x2, y2), espera_letra in zip(mescladas, PADRAO_MERCOSUL):
        rec = binaria[y1:y2, x1:x2]
        rec = cv2.copyMakeBorder(rec, 18, 18, 18, 18, cv2.BORDER_CONSTANT, value=255)
        wl = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if espera_letra else "0123456789"
        ch = pytesseract.image_to_string(
            rec, config=f"--oem 3 --psm 10 -c tessedit_char_whitelist={wl}"
        ).strip()
        ch = re.sub(r"[^A-Z0-9]", "", ch.upper())
        chars.append(ch[0] if ch else "")

    return "".join(chars)


def ler_melhor_placa(binarizacoes):
    votos = defaultdict(float)
    meta = {}
    melhor_fallback = (999, "", "", binarizacoes[0][0])

    for binaria, peso_bin in binarizacoes:
        for psm in (7, 8):
            bruto = pytesseract.image_to_string(
                binaria,
                config=f"--oem 3 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            ).strip()
            texto, custo = normalizar_e_pontuar_mercosul(bruto)

            if custo < melhor_fallback[0]:
                melhor_fallback = (custo, texto, bruto, binaria)

            if len(texto) == 7 and custo <= 2:
                mult = 3.0 if custo == 0 else (1.4 if custo == 1 else 0.6)
                votos[texto] += peso_bin * mult
                if texto not in meta or custo < meta[texto][0]:
                    meta[texto] = (custo, bruto, binaria)

    # Usa segmentacao se nenhuma leitura de linha fechou com boa confianca
    melhor_custo_linha = min((m[0] for m in meta.values()), default=999)
    if melhor_custo_linha > 0:
        for binaria, peso_bin in binarizacoes:
            seg = ler_por_segmentacao(binaria)
            if not seg:
                continue
            texto, custo = normalizar_e_pontuar_mercosul(seg)
            if custo < melhor_fallback[0]:
                melhor_fallback = (custo, texto, f"seg:{seg}", binaria)
            if len(texto) == 7 and custo <= 1:
                votos[texto] += peso_bin * (2.2 if custo == 0 else 1.0)
                if texto not in meta or custo < meta[texto][0]:
                    meta[texto] = (custo, f"seg:{seg}", binaria)

    if votos:
        vencedor = max(votos.items(), key=lambda kv: kv[1])[0]
        custo, bruto, binaria = meta[vencedor]
        return vencedor, bruto, binaria, custo

    custo, texto, bruto, binaria = melhor_fallback
    return texto, bruto, binaria, custo


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

    candidatos = extrair_candidatos_placa(img)
    if not candidatos:
        linha = f"{nome}: NAO_DETECTADA"
        resultados.append(linha)
        print(linha, flush=True)
        continue

    melhor_geral = (999, "", "", candidatos[0], None)
    for placa in candidatos:
        binarizacoes = gerar_binarizacoes(placa)
        texto, bruto, melhor_bin, custo = ler_melhor_placa(binarizacoes)
        if custo < melhor_geral[0]:
            melhor_geral = (custo, texto, bruto, placa, melhor_bin)
        if custo == 0:
            break

    _, texto, bruto, melhor_placa, melhor_bin = melhor_geral
    cv2.imwrite(os.path.join(DEBUG_DIR, f"crop_{nome}"), melhor_placa)
    if melhor_bin is not None:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"bin_{nome}"), melhor_bin)

    linha = f"{nome}: {texto} (bruto: {bruto})"
    resultados.append(linha)
    print(linha, flush=True)

with open(SAIDA_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(resultados))