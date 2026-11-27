# Reconhecimento de Placas Veiculares (ALPR): Tesseract OCR vs. EasyOCR

Projeto desenvolvido para a disciplina de **Processamento de Imagens**, com o objetivo de construir um pipeline completo de visão computacional para segmentação, extração e reconhecimento óptico de caracteres (OCR) em placas veiculares no padrão Mercosul, comparando o desempenho do **Tesseract OCR** e do **EasyOCR** sobre a mesma base de imagens.

---

## Estrutura do Projeto

```text
.
├── dataset
│   ├── 10.JPG                              # All dataset images (1.JPG to 20.JPG)
├── main.py
├── pipeline_easyocr.py                     # The main Script of Pipeline EasyOCR
├── pipeline_tesseract.py                   # The main Script of Pipeline Tesseract
├── README.md                               # We are here, documentation.
├── requirements.txt
├── resultados
│   ├── debug_tesseract                     # Cuts and Bin Images to visual debug
│   │   ├── bin_9.JPG
│   │   └── crop_9.JPG
│   └── resultados_tesseract.txt            # Results of Tessaract OCR pipeline.txt file
└── src
    ├── __init__.py
    ├── ocr.py
    ├── preprocessamento.py
    ├── __pycache__
    │   ├── __init__.cpython-312.pyc
    │   ├── ocr.cpython-312.pyc
    │   ├── preprocessamento.cpython-312.pyc
    │   └── segmentacao.cpython-312.pyc
    └── segmentacao.py