import cv2
import numpy as np


def calcular_angulo_horizontal(rect):
    """Calcula o angulo de inclinacao (em graus) do lado maior do retangulo minimo."""
    box = cv2.boxPoints(rect)
    d01 = np.linalg.norm(box[1] - box[0])
    d12 = np.linalg.norm(box[2] - box[1])
    p1, p2 = (box[0], box[1]) if d01 >= d12 else (box[1], box[2])
    if p2[0] < p1[0]:
        p1, p2 = p2, p1
    return float(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])))


def remover_moldura_preta(crop):
    """
    Varre a placa de dentro para fora para localizar e remover a moldura preta.
    A varredura interna evita que fundos claros externos (lataria, cromado ou
    placa antiga por baixo) impecam a remocao da borda.
    """
    cinza = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = cinza.shape
    if h < 15 or w < 40:
        return crop

    # Percentil 75 ignora os pixels brancos da marca d'agua dentro da moldura preta
    p75_linhas = np.percentile(cinza, 75, axis=1)

    # A faixa [12%, 48%] logo abaixo da barra azul sempre intercepta o miolo claro da placa
    y_ini = int(h * 0.12)
    y_fim = max(y_ini + 1, int(h * 0.48))
    y_mid = y_ini + int(np.argmax(p75_linhas[y_ini:y_fim]))
    limiar_v = min(85.0, float(p75_linhas[y_mid]) * 0.52)

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

    if base - topo < 15:
        topo, base = 0, h

    cinza_v = cinza[topo:base, :]
    h_v, w_v = cinza_v.shape

    # Referencia local por lateral para suportar placas com meia-sombra
    ref_esq = float(np.percentile(cinza_v[:, int(w_v * 0.08) : max(int(w_v * 0.08) + 1, w_v // 3)], 90))
    ref_dir = float(np.percentile(cinza_v[:, int(w_v * 0.58) : max(int(w_v * 0.58) + 1, int(w_v * 0.88))], 90))
    limiar_esq = min(85.0, ref_esq * 0.52)
    limiar_dir = min(85.0, ref_dir * 0.52)

    p80_cols = np.percentile(cinza_v, 80, axis=0)
    faixa_h = max(2, h_v // 8)
    topo_col = cinza_v[:faixa_h, :].mean(axis=0)
    base_col = cinza_v[-faixa_h:, :].mean(axis=0)

    # Moldura lateral e escura de ponta a ponta; caracteres preservam margem clara no topo/base
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

    # Pequeno recuo para descartar residuos de antialiasing da borda preta
    pad_t = max(2, int(h_v * 0.03)) if topo > 0 else 1
    pad_b = max(2, int(h_v * 0.03)) if base < h else 1
    pad_l = max(2, int((dir_ - esq) * 0.01)) if esq > 0 else 0
    pad_r = max(2, int((dir_ - esq) * 0.01)) if dir_ < w_v else 0

    y1, y2 = (topo + pad_t, base - pad_b) if (base - pad_b) - (topo + pad_t) >= 12 else (topo, base)
    x1, x2 = (esq + pad_l, dir_ - pad_r) if (dir_ - pad_r) - (esq + pad_l) >= 30 else (esq, dir_)

    return crop[y1:y2, x1:x2]


def extrair_candidatos_placa(img, max_candidatos=3):
    """
    Detecta e extrai ate `max_candidatos` regioes de placa Mercosul ordenadas por area:
    1. Segmentacao por cor da faixa superior azul (HSV + dominancia do canal B em BGR);
    2. Fechamento morfologico horizontal para unir a faixa interrompida pela palavra BRASIL;
    3. Alinhamento geometrico via transformacao afim (minAreaRect + warpAffine);
    4. Projecao da geometria da faixa azul para a area dos caracteres e remocao da moldura.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(img)
    h_img, w_img = img.shape[:2]

    # Dominancia do canal azul mantem a faixa sob sombra e rejeita superficies prata/cinza
    azul_bgr = ((b.astype(np.int16) - r.astype(np.int16)) >= 20) & (
        (b.astype(np.int16) - g.astype(np.int16)) >= 5
    )

    candidatos_brutos = []
    for s_min in (95, 65):
        mascara = cv2.inRange(hsv, np.array([96, s_min, 18]), np.array([134, 255, 255]))
        mascara = cv2.bitwise_and(mascara, (azul_bgr.astype(np.uint8) * 255))

        # Fechamento horizontal une segmentos da barra azul; abertura remove ruidos pontuais
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
        cx, cy = rect[0]
        if any(abs(cx - ox) < 35 and abs(cy - oy) < 25 for ox, oy in centros_vistos):
            continue

        angulo = calcular_angulo_horizontal(rect)
        if abs(angulo) > 35.0:
            continue

        # Corrige inclinacao da placa girando a imagem em torno do centro da barra azul
        matriz_rot = cv2.getRotationMatrix2D((cx, cy), angulo, 1.0)
        img_alinhada = cv2.warpAffine(
            img, matriz_rot, (w_img, h_img), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

        bx = cx - bw / 2.0
        by = cy - bh / 2.0

        # Projeta a regiao dos caracteres abaixo da barra azul (ignorando a sigla BR a esquerda)
        x1 = max(0, int(bx + bw * 0.08))
        x2 = min(w_img, int(bx + bw * 1.07))
        y1 = max(0, int(by + bh * 1.05))
        y2 = min(h_img, int(by + bh * 4.25))

        if x2 - x1 < 40 or y2 - y1 < 15:
            continue

        crop = img_alinhada[y1:y2, x1:x2]
        cinza_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if np.percentile(cinza_crop, 85) < 85 or np.std(cinza_crop) < 15:
            continue

        centros_vistos.append((cx, cy))
        crops.append(remover_moldura_preta(crop))
        if len(crops) >= max_candidatos:
            break

    return crops