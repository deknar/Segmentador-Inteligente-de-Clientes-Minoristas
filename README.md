# Segmentador Inteligente de Clientes Minoristas

Proyecto del Modulo #7: Aprendizaje de Maquina No Supervisado.  
Sistema de segmentacion de clientes a partir de datos transaccionales de e-commerce utilizando tecnicas de clustering y reduccion dimensional.

## Descripcion

Este proyecto aplica aprendizaje no supervisado sobre el dataset `online_retail.csv` para identificar segmentos de clientes con comportamientos de compra diferenciados. El pipeline incluye:

1. **Preprocesamiento:** Limpieza de datos, ingenieria de features RFM (Recency, Frequency, Monetary), eliminacion de outliers con IQR y estandarizacion con StandardScaler
2. **Reduccion dimensional:** PCA y t-SNE para visualizacion en 2D
3. **Clusterizacion:** K-Means, DBSCAN y agrupamiento jerarquico (Ward)
4. **Evaluacion:** Coeficiente de silueta, metodo del codo y comparativa entre algoritmos
5. **Informe:** Interpretacion comercial de segmentos y recomendaciones accionables

## Resultados principales

| Metrica | Valor |
|---------|-------|
| Transacciones originales | 541,909 |
| Transacciones limpias | 397,884 |
| Clientes analizados | 3,602 |
| Varianza explicada PCA (2 comp.) | 90.84% |
| Mejor modelo | K-Means (k=3) |
| Coeficiente de silueta | 0.4530 |

### Segmentos identificados

| Cluster | Tipo | Recency | Frequency | Monetary | % Clientes |
|---------|------|---------|-----------|----------|------------|
| 0 | VIP / Champions | 37 dias | 5.7 | $1,847 | 22.8% |
| 1 | Regulares / Prometedores | 50 dias | 2.0 | $559 | 52.1% |
| 2 | En Riesgo / Dormidos | 228 dias | 1.5 | $405 | 25.2% |

## Estructura del proyecto

```
Proyecto modulo #7/
├── main.py                 # Script principal con todo el pipeline
├── online_retail.csv       # Dataset de entrada
├── informe_final.md        # Informe tecnico con conclusiones
├── README.md               # Este archivo
└── graficos/               # Visualizaciones generadas (19 PNGs)
    ├── 01_rfm_distribucion_antes.png
    ├── 02_rfm_distribucion_despues.png
    ├── 03_rfm_boxplots.png
    ├── 04_rfm_correlacion.png
    ├── 05_pca_varianza.png
    ├── 06_pca_2d_sin_clusters.png
    ├── 07_tsne_2d_sin_clusters.png
    ├── 08_comparativa_pca_tsne.png
    ├── 09_metodo_codo.png
    ├── 10_silueta_por_k.png
    ├── 11_kmeans_pca_tsne.png
    ├── 12_dbscan_kdistance.png
    ├── 13_dbscan_pca_tsne.png
    ├── 14_dendrograma.png
    ├── 15_jerarquico_pca_tsne.png
    ├── 16_comparacion_siluetas.png
    ├── 17_perfil_rfm_clusters.png
    ├── 18_heatmap_perfiles.png
    └── 19_comparacion_3_algoritmos.png
```

## Requisitos

- Python
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy

Instalar dependencias:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

## Ejecucion

```bash
python main.py
```

El script genera automaticamente la carpeta `graficos/` con las 19 visualizaciones y muestra el informe completo en consola.

## Tecnologias utilizadas

- **Python** - Lenguaje principal
- **pandas** - Manipulacion y limpieza de datos
- **NumPy** - Operaciones numericas
- **scikit-learn** - Algoritmos de ML (K-Means, DBSCAN, AgglomerativeClustering, PCA, t-SNE, StandardScaler, silhouette_score)
- **matplotlib / seaborn** - Visualizaciones
- **scipy** - Dendrograma (linkage, dendrogram)
