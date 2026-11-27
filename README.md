# Reconhecimento de Placas Veiculares (ALPR): Tesseract OCR vs. EasyOCR

Projeto desenvolvido para a disciplina de **Processamento de Imagens**, com o objetivo de construir um pipeline completo de visão computacional para segmentação, extração e reconhecimento óptico de caracteres (OCR) em placas veiculares no padrão Mercosul, comparando o desempenho do **Tesseract OCR** e do **EasyOCR** sobre a mesma base de imagens.

---

## Estrutura do Projeto

```text
├── dataset
│   ├── 10.JPG              # All dataset Images.
├── debug_placas
│   ├── bin_10.JPG
│   ├── bin_11.JPG
│   ├── bin_12.JPG
│   ├── bin_13.JPG
│   ├── bin_14.JPG
│   ├── bin_15.JPG
│   ├── bin_16.JPG
│   ├── bin_17.JPG
│   ├── bin_18.JPG
│   ├── bin_19.JPG
│   ├── bin_1.JPG
│   ├── bin_20.JPG
│   ├── bin_2.JPG
│   ├── bin_3.JPG
│   ├── bin_4.JPG
│   ├── bin_5.JPG
│   ├── bin_6.JPG
│   ├── bin_7.JPG
│   ├── bin_8.JPG
│   ├── bin_9.JPG
│   ├── crop_10.JPG
│   ├── crop_11.JPG
│   ├── crop_12.JPG
│   ├── crop_13.JPG
│   ├── crop_14.JPG
│   ├── crop_15.JPG
│   ├── crop_16.JPG
│   ├── crop_17.JPG
│   ├── crop_18.JPG
│   ├── crop_19.JPG
│   ├── crop_1.JPG
│   ├── crop_20.JPG
│   ├── crop_2.JPG
│   ├── crop_3.JPG
│   ├── crop_4.JPG
│   ├── crop_5.JPG
│   ├── crop_6.JPG
│   ├── crop_7.JPG
│   ├── crop_8.JPG
│   └── crop_9.JPG
├── main.py
├── pipeline_easyocr.py
├── pipeline_tesseract.py
├── README.md
├── requirements.txt
├── resultados
│   ├── debug_easyocr
│   │   ├── crop_10.JPG  # All crop's
│   │   └── prep_9.JPG   # All prep's
│   ├── debug_placas
│   ├── debug_tesseract
│   │   ├── bin_10.JPG   # All bin's   
│   │   └── crop_9.JPG   # All crop's
│   ├── resultados_easyocr.txt
│   └── resultados_tesseract.txt
├── resultados_tesseract.txt
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