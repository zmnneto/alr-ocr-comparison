import cv2
import numpy as np


def limpar_binaria(binaria):
    """
    Pos-processa a imagem binarizada removendo restos de borda, ruidos pequenos
    e residuos da sigla 'BR', reescalando os caracteres para altura ideal (~48 px)
    com margem branca de seguranca.
    """
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

    # Filtragem por componentes conectados: descarta ruidos e restos de borda/BR
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    h_c, w_c = inv.shape
    limpa_inv = np.zeros_like(inv)

    for i in range(1, num_labels):
        cx, cy, cw, ch, area = stats[i]
        if ch < int(0.28 * h_c) or area < 70:
            continue
        if cw > int(0.32 * w_c) and ch < int(0.38 * h_c):
            continue
        if cx < int(0.10 * w_c) and ch < int(0.55 * h_c) and cy > int(0.38 * h_c):
            continue
        limpa_inv[labels == i] = 255

    ys, xs = np.where(limpa_inv > 0)
    if ys.size > 0 and (ys.max() - ys.min()) >= 16 and (xs.max() - xs.min()) >= 35:
        limpa_inv = limpa_inv[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    # Normaliza a altura dos caracteres para ~48 px (faixa otima para o LSTM do Tesseract)
    h_ink, w_ink = limpa_inv.shape
    if h_ink > 0:
        escala = 48.0 / h_ink
        novo_w = max(20, int(round(w_ink * escala)))
        interp = cv2.INTER_AREA if escala < 1 else cv2.INTER_CUBIC
        limpa_inv = cv2.resize(limpa_inv, (novo_w, 48), interpolation=interp)
        _, limpa_inv = cv2.threshold(limpa_inv, 127, 255, cv2.THRESH_BINARY)

    limpa = 255 - limpa_inv
    return cv2.copyMakeBorder(limpa, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=255)


def preprocessar_placa(placa_bgr):
    """
    Aplica tecnicas de pre-processamento sobre a placa segmentada e retorna:
    - `colorida_pad`: recorte original padronizado com margem (util para OCR neural);
    - `cinza_norm`: escala de cinza com sombras equalizadas e contraste esticado;
    - `binarizacoes`: lista de (imagem_binaria, peso) com diferentes limiarizacoes.
    """
    redim = cv2.resize(placa_bgr, (420, 96), interpolation=cv2.INTER_CUBIC)
    cinza = cv2.cvtColor(redim, cv2.COLOR_BGR2GRAY)

    # 1. Equalizacao de iluminacao: estima o fundo por fechamento morfologico amplo
    #    e divide a imagem pelo fundo, eliminando sombras duras sem gerar halos
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    fundo = cv2.morphologyEx(cinza, cv2.MORPH_CLOSE, kernel_bg)
    norm = cv2.divide(cinza, np.maximum(fundo, 1), scale=255)

    # 2. Expansao de contraste robusta por percentis (2% a 98%): recupera letras cinza-claras
    p2, p98 = np.percentile(norm, (2, 98))
    if p98 > p2 + 10:
        norm = np.clip((norm.astype(np.float32) - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)

    # 3. Abertura morfologica em escala de cinza: reduz riscos brancos da marca d'agua
    sem_marca_leve = cv2.morphologyEx(
        norm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    sem_marca_forte = cv2.morphologyEx(
        norm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )

    # 4. Suavizacao Gaussiana seguida de binarizacao de Otsu e Adaptativa
    suave_raw = cv2.GaussianBlur(norm, (3, 3), 0)
    suave_leve = cv2.GaussianBlur(sem_marca_leve, (3, 3), 0)
    suave_forte = cv2.GaussianBlur(sem_marca_forte, (3, 3), 0)

    _, bin_raw = cv2.threshold(suave_raw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bin_leve = cv2.threshold(suave_leve, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bin_forte = cv2.threshold(suave_forte, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_adapt = cv2.adaptiveThreshold(
        suave_leve, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 15
    )

    colorida_pad = cv2.copyMakeBorder(redim, 24, 24, 24, 24, cv2.BORDER_REPLICATE)
    cinza_norm_pad = cv2.copyMakeBorder(norm, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=255)

    binarizacoes = [
        (limpar_binaria(bin_forte), 1.3),
        (limpar_binaria(bin_leve), 1.15),
        (limpar_binaria(bin_raw), 1.0),
        (limpar_binaria(bin_adapt), 1.0),
    ]

    return {
        "colorida_pad": colorida_pad,
        "cinza_norm": cinza_norm_pad,
        "binarizacoes": binarizacoes,
    }