import re
from collections import defaultdict
import cv2
import numpy as np
import pytesseract

PADRAO_MERCOSUL = [True, True, True, False, True, False, False]  # LLLNLNN
WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
PARA_LETRA = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}
PARA_DIGITO = {
    "O": "0", "Q": "0", "D": "0", "U": "0",
    "I": "1", "L": "1", "J": "1",
    "Z": "2", "A": "4", "S": "5", "G": "6", "T": "7", "B": "8",
}


def normalizar_e_pontuar_mercosul(texto):
    """
    Limpa o texto bruto e busca a janela de 7 caracteres com menor custo de
    aderencia ao padrao Mercosul (LLLNLNN). Custo 0 indica conformidade exata.
    """
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


def _tesseract_por_segmentacao(binaria):
    """Fallback do Tesseract: segmenta os 7 caracteres e le cada um com PSM 10."""
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
        rec = cv2.copyMakeBorder(binaria[y1:y2, x1:x2], 18, 18, 18, 18, cv2.BORDER_CONSTANT, value=255)
        wl = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if espera_letra else "0123456789"
        ch = pytesseract.image_to_string(
            rec, config=f"--oem 3 --psm 10 -c tessedit_char_whitelist={wl}"
        ).strip()
        ch = re.sub(r"[^A-Z0-9]", "", ch.upper())
        chars.append(ch[0] if ch else "")

    return "".join(chars)


def reconhecer_tesseract(dados_prep):
    """Aplica Tesseract OCR nas binarizações geradas com votação ponderada e fallback por caractere."""
    binarizacoes = dados_prep["binarizacoes"]
    votos = defaultdict(float)
    meta = {}
    melhor_fallback = (999, "", "", binarizacoes[0][0])

    for binaria, peso_bin in binarizacoes:
        for psm in (7, 8):
            bruto = pytesseract.image_to_string(
                binaria, config=f"--oem 3 --psm {psm} -c tessedit_char_whitelist={WHITELIST}"
            ).strip()
            texto, custo = normalizar_e_pontuar_mercosul(bruto)

            if custo < melhor_fallback[0]:
                melhor_fallback = (custo, texto, bruto, binaria)

            if len(texto) == 7 and custo <= 2:
                mult = 3.0 if custo == 0 else (1.4 if custo == 1 else 0.6)
                votos[texto] += peso_bin * mult
                if texto not in meta or custo < meta[texto][0]:
                    meta[texto] = (custo, bruto, binaria)

    melhor_custo_linha = min((m[0] for m in meta.values()), default=999)
    if melhor_custo_linha > 0:
        for binaria, peso_bin in binarizacoes:
            seg = _tesseract_por_segmentacao(binaria)
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
        custo, bruto, img_usada = meta[vencedor]
        return vencedor, bruto, img_usada, custo

    custo, texto, bruto, img_usada = melhor_fallback
    return texto, bruto, img_usada, custo


def _extrair_texto_easyocr(reader, img):
    """Executa EasyOCR ordenando blocos da esquerda para a direita e ignorando residuos pequenos."""
    deteccoes = reader.readtext(img, allowlist=WHITELIST, detail=1)
    if not deteccoes:
        return "", 0.0

    # Calcula altura e posicao horizontal de cada caixa detectada
    itens = []
    for bbox, txt, conf in deteccoes:
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        itens.append((min(xs), max(ys) - min(ys), txt, float(conf)))

    max_h = max(it[1] for it in itens)
    # Filtra elementos muito menores que a linha principal (ex: sigla 'BR')
    filtrados = [it for it in itens if it[1] >= max_h * 0.45]
    filtrados.sort(key=lambda it: it[0])

    texto_bruto = "".join(it[2] for it in filtrados)
    conf_media = float(np.mean([it[3] for it in filtrados])) if filtrados else 0.0
    return texto_bruto, conf_media


def reconhecer_easyocr(dados_prep, reader):
    """
    Aplica EasyOCR sobre as saidas do mesmo pre-processamento (cinza normalizado,
    recorte colorido padronizado e binarizacao), escolhendo o melhor resultado
    pelo mesmo criterio Mercosul + confianca do modelo.
    """
    entradas = [
        (dados_prep["cinza_norm"], 1.3),
        (dados_prep["colorida_pad"], 1.2),
        (dados_prep["binarizacoes"][1][0], 1.0),  # bin_leve
    ]

    votos = defaultdict(float)
    meta = {}
    melhor_fallback = (999, "", "", entradas[0][0])

    for img_in, peso in entradas:
        bruto, conf = _extrair_texto_easyocr(reader, img_in)
        texto, custo = normalizar_e_pontuar_mercosul(bruto)

        if custo < melhor_fallback[0]:
            melhor_fallback = (custo, texto, bruto, img_in)

        if len(texto) == 7 and custo <= 2:
            mult = 3.0 if custo == 0 else (1.4 if custo == 1 else 0.6)
            votos[texto] += peso * mult * (0.5 + conf)
            if texto not in meta or custo < meta[texto][0]:
                meta[texto] = (custo, bruto, img_in)

    if votos:
        vencedor = max(votos.items(), key=lambda kv: kv[1])[0]
        custo, bruto, img_usada = meta[vencedor]
        return vencedor, bruto, img_usada, custo

    custo, texto, bruto, img_usada = melhor_fallback
    return texto, bruto, img_usada, custo

