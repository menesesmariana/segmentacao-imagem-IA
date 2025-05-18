import os
import argparse
import rasterio
import numpy as np
from tqdm import tqdm
from pathlib import Path

def divide_image(input_path, output_dir, block_size=256):
    """
    Divide uma imagem grande em blocos menores.
    
    Args:
        input_path (str): Caminho para a imagem de entrada
        output_dir (str): Diretório de saída para os blocos
        block_size (int): Tamanho dos blocos (quadrados)
    """
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Abrir a imagem com rasterio
    with rasterio.open(input_path) as src:
        # Obter dimensões da imagem
        height = src.height
        width = src.width
        
        # Calcular número de blocos
        n_blocks_h = height // block_size
        n_blocks_w = width // block_size
        
        # Para cada bloco
        for i in tqdm(range(n_blocks_h), desc="Processando linhas"):
            for j in range(n_blocks_w):
                # Calcular índices do bloco
                y_start = i * block_size
                x_start = j * block_size
                
                # Ler o bloco
                window = rasterio.windows.Window(
                    x_start, y_start, block_size, block_size
                )
                block = src.read(window=window)
                
                # Garantir que temos apenas 3 canais (RGB)
                if block.shape[0] > 3:
                    block = block[:3]  # Pegar apenas os primeiros 3 canais
                
                # Transpor para formato (altura, largura, canais)
                block = np.transpose(block, (1, 2, 0))
                
                # Salvar o bloco
                output_path = os.path.join(
                    output_dir,
                    f"bloco_{i:04d}_{j:04d}.png"
                )
                
                # Salvar usando rasterio para preservar metadados
                with rasterio.open(
                    output_path,
                    'w',
                    driver='PNG',
                    height=block_size,
                    width=block_size,
                    count=3,  # Forçar 3 canais (RGB)
                    dtype=block.dtype
                ) as dst:
                    dst.write(np.transpose(block, (2, 0, 1)))

def main():
    parser = argparse.ArgumentParser(
        description='Divide um ortomosaico em blocos menores'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Caminho para o arquivo ortomosaico'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Diretório de saída para os blocos'
    )
    parser.add_argument(
        '--block-size',
        type=int,
        default=256,
        help='Tamanho dos blocos (padrão: 256)'
    )
    
    args = parser.parse_args()
    
    # Converter caminhos para Path
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_path}")
    
    print(f"Dividindo imagem {input_path} em blocos de {args.block_size}x{args.block_size}")
    divide_image(str(input_path), str(output_dir), args.block_size)
    print(f"Blocos salvos em: {output_dir}")

if __name__ == '__main__':
    main() 