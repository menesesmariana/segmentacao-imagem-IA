# Segmentação de Vegetação usando IA

Este projeto implementa um sistema de segmentação de vegetação em imagens aéreas usando redes neurais artificiais.

## Estrutura do Projeto

```
.
├── dados/                      # Diretório com as imagens de entrada
│   └── Orthomosaico_roi.tif    # Imagem ortomosaico de entrada
├── src/                        # Código fonte
│   ├── divide_orthomosaic.py   # Script para dividir o ortomosaico em blocos
│   ├── binarize_images.py      # Script para binarização das imagens
│   ├── train_model.py          # Script para treinamento do modelo
│   ├── model_inference.py      # Script para inferência
│   └── vectorize_mask.py       # Script para vetorização (opcional)
├── requirements.txt            # Dependências do projeto
└── README.md                   # Este arquivo
```

## Instalação

1. Clone o repositório
2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

1. Dividir o ortomosaico em blocos:
```bash
python src/divide_orthomosaic.py --input dados/Orthomosaico_roi.tif --output dados/blocos/
```

2. Gerar dataset de treinamento:
```bash
python src/binarize_images.py --input dados/blocos/ --output dados/segmentados/
```

3. Treinar o modelo:
```bash
python src/train_model.py --rgb dados/blocos/ --groundtruth dados/segmentados/ --modelpath modelo/modelo.h5
```

4. Realizar inferência:
```bash
python src/model_inference.py --rgb dados/Orthomosaico_roi.tif --modelpath modelo/modelo.h5 --output dados/mascara.tif
```

5. (Opcional) Vetorizar o resultado:
```bash
python src/vectorize_mask.py --mask dados/mascara.tif --output dados/poligonos.geojson
```

## Detalhes Técnicos

- O modelo utiliza uma arquitetura U-Net simplificada para segmentação semântica
- As imagens são divididas em blocos de 256x256 pixels
- A binarização é realizada usando o índice ExG (Excess Green)
- O modelo é treinado para classificar pixels em duas classes: vegetação (1) e não-vegetação (0) 