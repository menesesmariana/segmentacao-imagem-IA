import os
import argparse
import numpy as np
import rasterio
import geopandas as gpd
from rasterio import features
from shapely.geometry import shape
from pathlib import Path
from tqdm import tqdm

def vectorize_mask(mask_path, output_path, min_area=100):
    """
    Vetoriza uma máscara binária em polígonos.
    
    Args:
        mask_path (str): Caminho para a máscara binária
        output_path (str): Caminho para salvar o GeoJSON
        min_area (float): Área mínima dos polígonos em pixels
    """
    with rasterio.open(mask_path) as src:
        # Ler a máscara
        mask = src.read(1)
        
        # Obter transformação de coordenadas
        transform = src.transform
        
        # Vetorizar
        shapes = features.shapes(
            mask,
            mask=(mask > 0),
            transform=transform
        )
        
        # Converter para GeoDataFrame
        polygons = []
        areas = []
        
        for geom, value in tqdm(shapes, desc="Processando polígonos"):
            # Converter para shapely
            poly = shape(geom)
            
            # Calcular área em pixels
            area = poly.area
            
            # Filtrar por área mínima
            if area >= min_area:
                polygons.append(poly)
                areas.append(area)
        
        # Criar GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {
                'geometry': polygons,
                'area_pixels': areas,
                'area_m2': [area * (transform[0] * transform[0]) for area in areas]
            },
            crs=src.crs
        )
        
        # Converter para EPSG:4326
        gdf = gdf.to_crs('EPSG:4326')
        
        # Salvar como GeoJSON
        gdf.to_file(output_path, driver='GeoJSON')
        
        # Imprimir estatísticas
        print(f"\nEstatísticas:")
        print(f"Número de polígonos: {len(gdf)}")
        print(f"Área total de vegetação: {gdf['area_m2'].sum():.2f} m²")
        print(f"Área média por polígono: {gdf['area_m2'].mean():.2f} m²")
        print(f"Área mínima: {gdf['area_m2'].min():.2f} m²")
        print(f"Área máxima: {gdf['area_m2'].max():.2f} m²")

def main():
    parser = argparse.ArgumentParser(
        description='Vetoriza uma máscara binária em polígonos'
    )
    parser.add_argument(
        '--mask',
        required=True,
        help='Caminho para a máscara binária'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Caminho para salvar o GeoJSON'
    )
    parser.add_argument(
        '--min-area',
        type=float,
        default=100,
        help='Área mínima dos polígonos em pixels (padrão: 100)'
    )
    
    args = parser.parse_args()
    
    # Verificar arquivos
    if not os.path.exists(args.mask):
        raise FileNotFoundError(f"Máscara não encontrada: {args.mask}")
    
    # Criar diretório de saída se necessário
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Vetorizar
    print(f"Vetorizando máscara: {args.mask}")
    vectorize_mask(args.mask, args.output, args.min_area)
    print(f"Polígonos salvos em: {args.output}")

if __name__ == '__main__':
    main() 