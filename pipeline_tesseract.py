import glob
import os
import re
import cv2

from src.segmentacao import extrair_candidatos_placa
from src.preprocessamento import preprocessar_placa
from src.ocr import reconhecer_tesseract

EXTENSOES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def ordem_numerica(caminho):
    return [int(n) for n in re.findall(r"\d+", os.path.basename(caminho))] or [0]


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_glob = os.path.join(base_dir, "dataset", "*.*")
    resultados_dir = os.path.join(base_dir, "resultados")
    debug_dir = os.path.join(resultados_dir, "debug_tesseract")
    saida_txt = os.path.join(resultados_dir, "resultados_tesseract.txt")

    os.makedirs(debug_dir, exist_ok=True)

    caminhos = [
        c for c in sorted(glob.glob(dataset_glob), key=ordem_numerica)
        if os.path.splitext(c)[1].lower() in EXTENSOES
    ]

    resultados = []
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

        melhor = (999, "", "", candidatos[0], None)
        for placa in candidatos:
            dados_prep = preprocessar_placa(placa)
            texto, bruto, img_usada, custo = reconhecer_tesseract(dados_prep)
            if custo < melhor[0]:
                melhor = (custo, texto, bruto, placa, img_usada)
            if custo == 0:
                break

        _, texto, bruto, melhor_placa, melhor_bin = melhor
        cv2.imwrite(os.path.join(debug_dir, f"crop_{nome}"), melhor_placa)
        if melhor_bin is not None:
            cv2.imwrite(os.path.join(debug_dir, f"bin_{nome}"), melhor_bin)

        linha = f"{nome}: {texto} (bruto: {bruto})"
        resultados.append(linha)
        print(linha, flush=True)

    with open(saida_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(resultados))
    print(f"\nResultados salvos em: {saida_txt}")


if __name__ == "__main__":
    main()