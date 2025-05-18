import os
import argparse
import numpy as np
import rasterio
from tqdm import tqdm
from pathlib import Path
from skimage import io

def calculate_exg(rgb_image):
    """
    Calcula o índice ExG (Excess Green) para uma imagem RGB.
    
    Args:
        rgb_image (np.ndarray): Imagem RGB normalizada (0-1)
    
    Returns:
        np.ndarray: Índice ExG
    """
    r, g, b = rgb_image[:,:,0], rgb_image[:,:,1], rgb_image[:,:,2]
    exg = 2 * g - r - b
    return exg

def binarize_image(input_path, output_path, threshold=0.1):
    """
    Binariza uma imagem RGB usando o índice ExG.
    
    Args:
        input_path (str): Caminho para a imagem de entrada
        output_path (str): Caminho para salvar a imagem binarizada
        threshold (float): Limiar para binarização
    """
    # Ler a imagem
    with rasterio.open(input_path) as src:
        image = src.read()
        image = np.transpose(image, (1, 2, 0))
    
    # Normalizar para 0-1
    image = image.astype(np.float32) / 255.0
    
    # Calcular ExG
    exg = calculate_exg(image)
    
    # Binarizar
    binary = (exg > threshold).astype(np.uint8) * 255
    
    # Salvar a imagem binarizada
    with rasterio.open(
        output_path,
        'w',
        driver='PNG',
        height=binary.shape[0],
        width=binary.shape[1],
        count=1,
        dtype=binary.dtype
    ) as dst:
        dst.write(binary[np.newaxis, :, :])

def process_directory(input_dir, output_dir, threshold=0.1):
    """
    Processa todas as imagens em um diretório.
    
    Args:
        input_dir (str): Diretório com as imagens de entrada
        output_dir (str): Diretório para salvar as imagens binarizadas
        threshold (float): Limiar para binarização
    """
    # Criar diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    
    # Listar arquivos de entrada
    input_files = list(Path(input_dir).glob('*.png'))
    
    # Processar cada arquivo
    for input_file in tqdm(input_files, desc="Processando imagens"):
        output_file = Path(output_dir) / f"mask_{input_file.stem}.png"
        binarize_image(str(input_file), str(output_file), threshold)

def main():
    parser = argparse.ArgumentParser(
        description='Binariza imagens RGB para detecção de vegetação usando ExG'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Diretório com as imagens RGB'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Diretório para salvar as imagens binarizadas'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.1,
        help='Limiar para binarização (padrão: 0.1)'
    )
    
    args = parser.parse_args()
    
    # Converter caminhos para Path
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Diretório de entrada não encontrado: {input_dir}")
    
    print(f"Processando imagens de {input_dir}")
    print(f"Usando limiar de binarização: {args.threshold}")
    process_directory(str(input_dir), str(output_dir), args.threshold)
    print(f"Imagens binarizadas salvas em: {output_dir}")

if __name__ == '__main__':
    main() 