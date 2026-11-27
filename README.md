# Reconhecimento de Placas Veiculares (ALPR): Tesseract OCR vs. EasyOCR

Projeto desenvolvido para a disciplina de **Processamento de Imagens**, com o objetivo de construir um pipeline completo de visão computacional para segmentação, extração e reconhecimento óptico de caracteres (OCR) em placas veiculares no padrão Mercosul, comparando o desempenho do **Tesseract OCR** e do **EasyOCR** sobre a mesma base de imagens.

---

## Estrutura do Projeto

```text
alr-ocr-comparison/
├── dataset/                         # Imagens do dataset (1.JPG a 20.JPG)
├── src/
│   ├── __init__.py
│   ├── segmentacao.py               # Detecção da faixa azul Mercosul, alinhamento e recorte
│   ├── preprocessamento.py          # Equalização de iluminação, contraste, morfologia e binarização
│   └── ocr.py                       # Validação Mercosul (LLLNLNN) e integração Tesseract / EasyOCR
├── resultados/
│   ├── resultados_tesseract.txt     # Saída .txt do pipeline com Tesseract OCR
│   ├── resultados_easyocr.txt       # Saída .txt do pipeline com EasyOCR
│   ├── debug_tesseract/             # Recortes e imagens binarizadas (conferência visual)
│   └── debug_easyocr/               # Recortes e imagens pré-processadas (conferência visual)
├── pipeline_tesseract.py            # Script principal — Pipeline Tesseract OCR
├── pipeline_easyocr.py              # Script principal — Pipeline EasyOCR
├── requirements.txt                 # Dependências do projeto
└── README.md                        # Estamos aqui :)