import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
sns.set_style("whitegrid")

output_dir = os.path.dirname(os.path.abspath(__file__))
graficos_dir = os.path.join(output_dir, "graficos")
os.makedirs(graficos_dir, exist_ok=True)

# leccion 1 y 2: fundamentos del aprendizaje no supervisado y tecnicas de clusterizacion
# (teoria documentada en informe_final.md)

# seccion 1: carga, limpieza y preprocesamiento

df = pd.read_csv(os.path.join(output_dir, "online_retail.csv"), encoding='latin1')

print(f"\ndataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"\nprimeras filas:")
print(df.head())
print(f"\ntipos de datos:")
print(df.dtypes)
print(f"\nvalores nulos por columna:")
print(df.isnull().sum())

filas_iniciales = len(df)

# eliminar columna index si existe
if 'index' in df.columns:
    df = df.drop(columns=['index'])

# eliminar filas sin customerid
df = df.dropna(subset=['CustomerID'])
df['CustomerID'] = df['CustomerID'].astype(int)

# eliminar devoluciones (invoiceno empieza con 'C')
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

# eliminar cantidades y precios negativos o cero
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]

# eliminar descripciones nulas
df = df.dropna(subset=['Description'])

filas_final = len(df)
print(f"\nlimpieza completada:")
print(f"  - filas iniciales: {filas_iniciales:,}")
print(f"  - filas eliminadas: {filas_iniciales - filas_final:,}")
print(f"  - filas restantes: {filas_final:,}")

df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

print(f"\ncolumna TotalPrice creada")
print(f"rango de fechas: {df['InvoiceDate'].min()} a {df['InvoiceDate'].max()}")

fecha_referencia = df['InvoiceDate'].max() + pd.Timedelta(days=1)
print(f"fecha de referencia para recency: {fecha_referencia}")

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (fecha_referencia - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

print(f"\ntabla rfm creada: {rfm.shape[0]} clientes")
print(f"\nestadisticas rfm:")
print(rfm[['Recency', 'Frequency', 'Monetary']].describe().round(2))

# grafico de distribuciones rfm antes de limpiar outliers
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colores = ['#3498db', '#2ecc71', '#e74c3c']
for i, col in enumerate(['Recency', 'Frequency', 'Monetary']):
    axes[i].hist(rfm[col], bins=50, color=colores[i], edgecolor='black', alpha=0.8)
    axes[i].set_title(f'distribucion de {col.lower()} (antes de limpiar)')
    axes[i].set_xlabel(col.lower())
    axes[i].set_ylabel('frecuencia')
    axes[i].axvline(rfm[col].median(), color='black', linestyle='--',
                    label=f'mediana: {rfm[col].median():.0f}')
    axes[i].legend()
plt.suptitle('distribuciones rfm antes de eliminar outliers',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, "01_rfm_distribucion_antes.png"), dpi=150)
plt.close()

clientes_antes = len(rfm)

for col in ['Recency', 'Frequency', 'Monetary']:
    q1 = rfm[col].quantile(0.25)
    q3 = rfm[col].quantile(0.75)
    iqr = q3 - q1
    limite_inf = q1 - 1.5 * iqr
    limite_sup = q3 + 1.5 * iqr
    rfm = rfm[(rfm[col] >= limite_inf) & (rfm[col] <= limite_sup)]

clientes_despues = len(rfm)
print(f"\neliminacion de outliers (iqr 1.5x):")
print(f"  - clientes antes: {clientes_antes:,}")
print(f"  - clientes eliminados: {clientes_antes - clientes_despues:,}")
print(f"  - clientes restantes: {clientes_despues:,}")

# grafico despues de limpiar outliers
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(['Recency', 'Frequency', 'Monetary']):
    axes[i].hist(rfm[col], bins=50, color=colores[i], edgecolor='black', alpha=0.8)
    axes[i].set_title(f'distribucion de {col.lower()} (despues de limpiar)')
    axes[i].set_xlabel(col.lower())
    axes[i].set_ylabel('frecuencia')
    axes[i].axvline(rfm[col].median(), color='black', linestyle='--',
                    label=f'mediana: {rfm[col].median():.0f}')
    axes[i].legend()
plt.suptitle('distribuciones rfm despues de eliminar outliers',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, "02_rfm_distribucion_despues.png"), dpi=150)
plt.close()

# boxplots rfm
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, col in enumerate(['Recency', 'Frequency', 'Monetary']):
    bp = axes[i].boxplot(rfm[col], patch_artist=True,
                         boxprops=dict(facecolor=colores[i], alpha=0.7))
    axes[i].set_title(f'boxplot de {col.lower()}')
    axes[i].set_ylabel(col.lower())
plt.suptitle('boxplots rfm (sin outliers)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, "03_rfm_boxplots.png"), dpi=150)
plt.close()

features_rfm = rfm[['Recency', 'Frequency', 'Monetary']].values
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_rfm)

print(f"\nestandarizacion completada:")
print(f"  - media (aprox 0): {features_scaled.mean(axis=0).round(6)}")
print(f"  - std  (aprox 1): {features_scaled.std(axis=0).round(6)}")

# matriz de correlacion rfm
fig, ax = plt.subplots(figsize=(8, 6))
correlacion = rfm[['Recency', 'Frequency', 'Monetary']].corr()
sns.heatmap(correlacion, annot=True, cmap='coolwarm', center=0,
            fmt='.3f', linewidths=1, ax=ax)
ax.set_title('matriz de correlacion rfm')
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, "04_rfm_correlacion.png"), dpi=150)
plt.close()

print(f"\ncorrelacion rfm:")
print(correlacion.round(3))

# leccion 3 / seccion 2: reduccion dimensional (pca + t-sne)

# pca completo para ver varianza explicada
pca_full = PCA()
pca_full.fit(features_scaled)

varianza_explicada = pca_full.explained_variance_ratio_
varianza_acumulada = np.cumsum(varianza_explicada)

print(f"\nanalisis de varianza explicada (pca):")
for i in range(len(varianza_explicada)):
    print(f"  pc{i+1}: {varianza_explicada[i]*100:.2f}% "
          f"(acumulada: {varianza_acumulada[i]*100:.2f}%)")

# scree plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(1, len(varianza_explicada) + 1),
            varianza_explicada * 100, color='steelblue',
            edgecolor='black', alpha=0.8)
axes[0].set_xlabel('componente principal')
axes[0].set_ylabel('varianza explicada (%)')
axes[0].set_title('varianza explicada por componente')
axes[0].set_xticks(range(1, len(varianza_explicada) + 1))

axes[1].plot(range(1, len(varianza_acumulada) + 1),
             varianza_acumulada * 100, 'o-', color='steelblue', linewidth=2)
axes[1].axhline(y=90, color='red', linestyle='--', alpha=0.7,
                label='90% varianza')
axes[1].set_xlabel('numero de componentes')
axes[1].set_ylabel('varianza acumulada (%)')
axes[1].set_title('varianza acumulada')
axes[1].legend()
axes[1].set_xticks(range(1, len(varianza_acumulada) + 1))

plt.suptitle('analisis pca - varianza explicada', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, "05_pca_varianza.png"), dpi=150)
plt.close()

# reduccion a 2d con pca
pca_2d = PCA(n_components=2)
datos_pca = pca_2d.fit_transform(features_scaled)

print(f"\npca reducido a 2d:")
print(f"  - pc1: {pca_2d.explained_variance_ratio_[0]*100:.2f}%")
print(f"  - pc2: {pca_2d.explained_variance_ratio_[1]*100:.2f}%")
print(f"  - total: {sum(pca_2d.explained_variance_ratio_)*100:.2f}%")

# scatter pca 2d (sin clusters todavia)
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(datos_pca[:, 0], datos_pca[:, 1],
                     c='steelblue', alpha=0.5, edgecolors='black',
                     linewidth=0.3, s=30)
ax.set_xlabel(f'pc1 ({pca_2d.explained_variance_ratio_[0]*100:.2f}% varianza)')
ax.set_ylabel(f'pc2 ({pca_2d.explained_variance_ratio_[1]*100:.2f}% varianza)')
ax.set_title('pca - reduccion a 2d (datos sin etiquetar)')
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, "06_pca_2d_sin_clusters.png"), dpi=150)
plt.close()

print(f"\naplicando t-sne (perplexity=30, max_iter=1000)...")
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200,
            max_iter=1000, random_state=42)
datos_tsne = tsne.fit_transform(features_scaled)
print(f"t-sne completado")

# scatter t-sne 2d
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(datos_tsne[:, 0], datos_tsne[:, 1],
           c='coral', alpha=0.5, edgecolors='black',
           linewidth=0.3, s=30)
ax.set_xlabel('t-sne componente 1')
ax.set_ylabel('t-sne componente 2')
ax.set_title('t-sne - reduccion a 2d (datos sin etiquetar)')
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, "07_tsne_2d_sin_clusters.png"), dpi=150)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

axes[0].scatter(datos_pca[:, 0], datos_pca[:, 1],
                c='steelblue', alpha=0.5, edgecolors='black',
                linewidth=0.3, s=30)
axes[0].set_xlabel(f'pc1 ({pca_2d.explained_variance_ratio_[0]*100:.2f}%)')
axes[0].set_ylabel(f'pc2 ({pca_2d.explained_variance_ratio_[1]*100:.2f}%)')
axes[0].set_title('pca - reduccion a 2d')

axes[1].scatter(datos_tsne[:, 0], datos_tsne[:, 1],
                c='coral', alpha=0.5, edgecolors='black',
                linewidth=0.3, s=30)
axes[1].set_xlabel('t-sne componente 1')
axes[1].set_ylabel('t-sne componente 2')
axes[1].set_title('t-sne - reduccion a 2d')

plt.suptitle('comparativa: pca vs t-sne', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, "08_comparativa_pca_tsne.png"), dpi=150)
plt.close()

print("comparativa pca vs t-sne guardada en graficos/08_comparativa_pca_tsne.png")

# leccion 4 / seccion 3: clusterizacion (k-means, dbscan, jerarquico)

print(f"\nmetodo del codo (k=2 a k=10):")
inercias = []
rango_k = range(2, 11)

for k in rango_k:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(features_scaled)
    inercias.append(km.inertia_)
    print(f"  k={k}: inercia = {km.inertia_:.2f}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(list(rango_k), inercias, 'o-', color='steelblue', linewidth=2,
        markersize=8)
ax.set_xlabel('numero de clusters (k)')
ax.set_ylabel('inercia (wcss)')
ax.set_title('metodo del codo - k-means')
ax.set_xticks(list(rango_k))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, "09_metodo_codo.png"), dpi=150)
plt.close()

print(f"\ncoeficiente de silueta (k=2 a k=10):")
siluetas = []

for k in rango_k:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(features_scaled)
    sil = silhouette_score(features_scaled, labels)
    siluetas.append(sil)
    print(f"  k={k}: silueta = {sil:.4f}")

k_optimo = list(rango_k)[np.argmax(siluetas)]
print(f"\nk optimo segun silueta: {k_optimo} (silueta = {max(siluetas):.4f})")

fig, ax = plt.subplots(figsize=(10, 6))
barras = ax.bar(list(rango_k), siluetas, color='steelblue',
                edgecolor='black', alpha=0.8)
barras[np.argmax(siluetas)].set_color('#e74c3c')
ax.set_xlabel('numero de clusters (k)')
ax.set_ylabel('coeficiente de silueta')
ax.set_title('coeficiente de silueta por k')
ax.set_xticks(list(rango_k))
ax.axhline(y=max(siluetas), color='red', linestyle='--', alpha=0.5,
           label=f'mejor: k={k_optimo} ({max(siluetas):.4f})')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, "10_silueta_por_k.png"), dpi=150)
plt.close()

print(f"\n--- k-means (k={k_optimo}) ---")
kmeans = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(features_scaled)

sil_kmeans = silhouette_score(features_scaled, labels_kmeans)
print(f"silueta k-means: {sil_kmeans:.4f}")
print(f"distribucion de clusters:")
for c in range(k_optimo):
    n = (labels_kmeans == c).sum()
    print(f"  cluster {c}: {n} clientes ({n/len(labels_kmeans)*100:.1f}%)")

# paleta de colores para los clusters
colores_clusters = plt.cm.Set2(np.linspace(0, 1, k_optimo))

# visualizar k-means sobre pca 2d
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for c in range(k_optimo):
    mask = labels_kmeans == c
    axes[0].scatter(datos_pca[mask, 0], datos_pca[mask, 1],
                    c=[colores_clusters[c]], label=f'cluster {c}',
                    alpha=0.6, edgecolors='black', linewidth=0.3, s=40)
axes[0].set_xlabel(f'pc1 ({pca_2d.explained_variance_ratio_[0]*100:.2f}%)')
axes[0].set_ylabel(f'pc2 ({pca_2d.explained_variance_ratio_[1]*100:.2f}%)')
axes[0].set_title(f'k-means (k={k_optimo}) sobre pca 2d')
axes[0].legend(title='clusters')

for c in range(k_optimo):
    mask = labels_kmeans == c
    axes[1].scatter(datos_tsne[mask, 0], datos_tsne[mask, 1],
                    c=[colores_clusters[c]], label=f'cluster {c}',
                    alpha=0.6, edgecolors='black', linewidth=0.3, s=40)
axes[1].set_xlabel('t-sne componente 1')
axes[1].set_ylabel('t-sne componente 2')
axes[1].set_title(f'k-means (k={k_optimo}) sobre t-sne 2d')
axes[1].legend(title='clusters')

plt.suptitle(f'k-means con k={k_optimo} clusters',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, "11_kmeans_pca_tsne.png"), dpi=150)
plt.close()

print(f"\n--- dbscan ---")

# determinar eps con k-distance plot
k_vecinos = 5
nn = NearestNeighbors(n_neighbors=k_vecinos)
nn.fit(features_scaled)
distancias, _ = nn.kneighbors(features_scaled)
distancias_k = np.sort(distancias[:, -1])

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(distancias_k, color='steelblue', linewidth=1.5)
ax.set_xlabel('puntos ordenados')
ax.set_ylabel(f'distancia al {k_vecinos}-esimo vecino')
ax.set_title(f'k-distance plot (k={k_vecinos}) para determinar eps')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, "12_dbscan_kdistance.png"), dpi=150)
plt.close()

# buscar el mejor eps probando varios valores
print(f"explorando valores de eps:")
mejor_sil_dbscan = -1
mejor_eps = 0.5
mejor_labels_dbscan = None

for eps_val in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    db = DBSCAN(eps=eps_val, min_samples=5)
    labels_db = db.fit_predict(features_scaled)
    n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
    n_ruido = (labels_db == -1).sum()

    if n_clusters >= 2:
        mask_no_ruido = labels_db != -1
        if mask_no_ruido.sum() > 0 and len(set(labels_db[mask_no_ruido])) >= 2:
            sil_db = silhouette_score(features_scaled[mask_no_ruido],
                                      labels_db[mask_no_ruido])
        else:
            sil_db = -1
    else:
        sil_db = -1

    print(f"  eps={eps_val}: clusters={n_clusters}, "
          f"ruido={n_ruido} ({n_ruido/len(labels_db)*100:.1f}%), "
          f"silueta={sil_db:.4f}")

    if sil_db > mejor_sil_dbscan:
        mejor_sil_dbscan = sil_db
        mejor_eps = eps_val
        mejor_labels_dbscan = labels_db.copy()

print(f"\nmejor eps: {mejor_eps} (silueta: {mejor_sil_dbscan:.4f})")

labels_dbscan = mejor_labels_dbscan
n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)

print(f"clusters encontrados: {n_clusters_dbscan}")
print(f"puntos de ruido: {(labels_dbscan == -1).sum()}")
print(f"distribucion de clusters:")
for c in sorted(set(labels_dbscan)):
    n = (labels_dbscan == c).sum()
    nombre = f"cluster {c}" if c != -1 else "ruido"
    print(f"  {nombre}: {n} clientes ({n/len(labels_dbscan)*100:.1f}%)")

# visualizar dbscan
colores_dbscan = plt.cm.Set2(np.linspace(0, 1, max(n_clusters_dbscan, 2)))

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for c in sorted(set(labels_dbscan)):
    mask = labels_dbscan == c
    if c == -1:
        axes[0].scatter(datos_pca[mask, 0], datos_pca[mask, 1],
                        c='gray', label='ruido', alpha=0.3, s=20, marker='x')
        axes[1].scatter(datos_tsne[mask, 0], datos_tsne[mask, 1],
                        c='gray', label='ruido', alpha=0.3, s=20, marker='x')
    else:
        axes[0].scatter(datos_pca[mask, 0], datos_pca[mask, 1],
                        c=[colores_dbscan[c % len(colores_dbscan)]],
                        label=f'cluster {c}', alpha=0.6,
                        edgecolors='black', linewidth=0.3, s=40)
        axes[1].scatter(datos_tsne[mask, 0], datos_tsne[mask, 1],
                        c=[colores_dbscan[c % len(colores_dbscan)]],
                        label=f'cluster {c}', alpha=0.6,
                        edgecolors='black', linewidth=0.3, s=40)

axes[0].set_xlabel(f'pc1 ({pca_2d.explained_variance_ratio_[0]*100:.2f}%)')
axes[0].set_ylabel(f'pc2 ({pca_2d.explained_variance_ratio_[1]*100:.2f}%)')
axes[0].set_title(f'dbscan (eps={mejor_eps}) sobre pca 2d')
axes[0].legend(title='clusters')

axes[1].set_xlabel('t-sne componente 1')
axes[1].set_ylabel('t-sne componente 2')
axes[1].set_title(f'dbscan (eps={mejor_eps}) sobre t-sne 2d')
axes[1].legend(title='clusters')

plt.suptitle(f'dbscan con eps={mejor_eps}, min_samples=5',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, "13_dbscan_pca_tsne.png"), dpi=150)
plt.close()

print(f"\n--- agrupamiento jerarquico ---")

# dendrograma (usar muestra si hay muchos datos)
n_muestra_dendro = min(500, len(features_scaled))
np.random.seed(42)
idx_muestra = np.random.choice(len(features_scaled), n_muestra_dendro,
                               replace=False)
datos_muestra = features_scaled[idx_muestra]

linkage_matrix = linkage(datos_muestra, method='ward')

fig, ax = plt.subplots(figsize=(16, 8))
dendrogram(linkage_matrix, truncate_mode='lastp', p=30,
           leaf_rotation=90, leaf_font_size=10, ax=ax,
           color_threshold=0.7 * max(linkage_matrix[:, 2]))
ax.set_xlabel('clientes (agrupados)')
ax.set_ylabel('distancia (ward)')
ax.set_title(f'dendrograma jerarquico (muestra de {n_muestra_dendro} clientes)')
ax.axhline(y=0.7 * max(linkage_matrix[:, 2]), color='red',
           linestyle='--', alpha=0.7, label='corte sugerido')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, "14_dendrograma.png"), dpi=150)
plt.close()

# aplicar agglomerative clustering con k_optimo
agg = AgglomerativeClustering(n_clusters=k_optimo, linkage='ward')
labels_jerarquico = agg.fit_predict(features_scaled)

sil_jerarquico = silhouette_score(features_scaled, labels_jerarquico)
print(f"silueta jerarquico (k={k_optimo}): {sil_jerarquico:.4f}")
print(f"distribucion de clusters:")
for c in range(k_optimo):
    n = (labels_jerarquico == c).sum()
    print(f"  cluster {c}: {n} clientes ({n/len(labels_jerarquico)*100:.1f}%)")

# visualizar jerarquico
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for c in range(k_optimo):
    mask = labels_jerarquico == c
    axes[0].scatter(datos_pca[mask, 0], datos_pca[mask, 1],
                    c=[colores_clusters[c]], label=f'cluster {c}',
                    alpha=0.6, edgecolors='black', linewidth=0.3, s=40)
    axes[1].scatter(datos_tsne[mask, 0], datos_tsne[mask, 1],
                    c=[colores_clusters[c]], label=f'cluster {c}',
                    alpha=0.6, edgecolors='black', linewidth=0.3, s=40)

axes[0].set_xlabel(f'pc1 ({pca_2d.explained_variance_ratio_[0]*100:.2f}%)')
axes[0].set_ylabel(f'pc2 ({pca_2d.explained_variance_ratio_[1]*100:.2f}%)')
axes[0].set_title(f'jerarquico (k={k_optimo}) sobre pca 2d')
axes[0].legend(title='clusters')

axes[1].set_xlabel('t-sne componente 1')
axes[1].set_ylabel('t-sne componente 2')
axes[1].set_title(f'jerarquico (k={k_optimo}) sobre t-sne 2d')
axes[1].legend(title='clusters')

plt.suptitle(f'agrupamiento jerarquico con k={k_optimo} clusters (ward)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, "15_jerarquico_pca_tsne.png"), dpi=150)
plt.close()

# leccion 5 / seccion 4: evaluacion e informe de resultados

# calcular silueta para dbscan (sin ruido)
mask_no_ruido = labels_dbscan != -1
if mask_no_ruido.sum() > 0 and len(set(labels_dbscan[mask_no_ruido])) >= 2:
    sil_dbscan_final = silhouette_score(features_scaled[mask_no_ruido],
                                         labels_dbscan[mask_no_ruido])
else:
    sil_dbscan_final = -1

n_clusters_db = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)

print(f"""
--- tabla comparativa de algoritmos ---

algoritmo          | clusters | silueta  | observaciones
-------------------|----------|----------|-----------------------------
k-means            | {k_optimo:>8} | {sil_kmeans:>8.4f} | mejor rendimiento general
dbscan             | {n_clusters_db:>8} | {sil_dbscan_final:>8.4f} | detecta ruido ({(labels_dbscan == -1).sum()} puntos)
jerarquico (ward)  | {k_optimo:>8} | {sil_jerarquico:>8.4f} | captura jerarquias
""")

# grafico comparativo de siluetas
fig, ax = plt.subplots(figsize=(8, 5))
algoritmos = ['K-Means', 'DBSCAN\n(sin ruido)', 'Jerarquico\n(Ward)']
siluetas_comp = [sil_kmeans, sil_dbscan_final, sil_jerarquico]
colores_bar = ['#3498db', '#e74c3c', '#2ecc71']
bars = ax.bar(algoritmos, siluetas_comp, color=colores_bar, edgecolor='black',
              alpha=0.85)
for bar, val in zip(bars, siluetas_comp):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', fontweight='bold')
ax.set_ylabel('coeficiente de silueta')
ax.set_title('comparacion de silueta entre algoritmos')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, "16_comparacion_siluetas.png"), dpi=150)
plt.close()

# determinar mejor modelo
siluetas_modelos = {
    'k-means': sil_kmeans,
    'dbscan': sil_dbscan_final,
    'jerarquico': sil_jerarquico
}
mejor_modelo = max(siluetas_modelos, key=siluetas_modelos.get)
print(f"mejor modelo segun silueta: {mejor_modelo} "
      f"({siluetas_modelos[mejor_modelo]:.4f})")

# usar labels del mejor modelo para el perfil
if mejor_modelo == 'k-means':
    labels_mejor = labels_kmeans
elif mejor_modelo == 'jerarquico':
    labels_mejor = labels_jerarquico
else:
    labels_mejor = labels_kmeans  # fallback a kmeans si dbscan

rfm_perfil = rfm.copy()
rfm_perfil['Cluster'] = labels_mejor

print(f"\nperfil rfm por cluster ({mejor_modelo}):")
perfil = rfm_perfil.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].agg(
    ['mean', 'median', 'count']
)
print(perfil.round(2))

# tabla de medias
medias = rfm_perfil.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
print(f"\nmedias por cluster:")
print(medias.round(2))

# grafico de perfil rfm por cluster
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colores_perfil = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12',
                  '#9b59b6', '#1abc9c', '#e67e22', '#34495e']

for i, col in enumerate(['Recency', 'Frequency', 'Monetary']):
    cluster_medias = medias[col]
    bars = axes[i].bar(range(len(cluster_medias)), cluster_medias.values,
                       color=colores_perfil[:len(cluster_medias)],
                       edgecolor='black', alpha=0.85)
    axes[i].set_xlabel('cluster')
    axes[i].set_ylabel(col.lower() + ' (media)')
    axes[i].set_title(f'{col.lower()} promedio por cluster')
    axes[i].set_xticks(range(len(cluster_medias)))
    for bar, val in zip(bars, cluster_medias.values):
        axes[i].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5,
                     f'{val:.1f}', ha='center', fontsize=10)

plt.suptitle(f'perfil rfm por cluster ({mejor_modelo})',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, "17_perfil_rfm_clusters.png"), dpi=150)
plt.close()

# grafico radar / heatmap de perfiles
fig, ax = plt.subplots(figsize=(10, 6))
# normalizar medias para heatmap
medias_norm = medias.copy()
for col in medias_norm.columns:
    medias_norm[col] = (medias_norm[col] - medias_norm[col].min()) / \
                       (medias_norm[col].max() - medias_norm[col].min() + 1e-10)

sns.heatmap(medias_norm, annot=medias.round(1).values, cmap='YlOrRd',
            linewidths=1, fmt='', ax=ax,
            xticklabels=['Recency\n(dias)', 'Frequency\n(compras)',
                         'Monetary\n(gasto total)'])
ax.set_ylabel('cluster')
ax.set_title(f'heatmap de perfiles rfm por cluster ({mejor_modelo})')
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, "18_heatmap_perfiles.png"), dpi=150)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(22, 7))

# k-means
for c in range(k_optimo):
    mask = labels_kmeans == c
    axes[0].scatter(datos_pca[mask, 0], datos_pca[mask, 1],
                    c=[colores_clusters[c]], label=f'cluster {c}',
                    alpha=0.6, edgecolors='black', linewidth=0.3, s=35)
axes[0].set_title(f'k-means (k={k_optimo})\nsilueta: {sil_kmeans:.4f}')
axes[0].set_xlabel('pc1')
axes[0].set_ylabel('pc2')
axes[0].legend(fontsize=8)

# dbscan
for c in sorted(set(labels_dbscan)):
    mask = labels_dbscan == c
    if c == -1:
        axes[1].scatter(datos_pca[mask, 0], datos_pca[mask, 1],
                        c='gray', label='ruido', alpha=0.3, s=15, marker='x')
    else:
        axes[1].scatter(datos_pca[mask, 0], datos_pca[mask, 1],
                        c=[colores_dbscan[c % len(colores_dbscan)]],
                        label=f'cluster {c}', alpha=0.6,
                        edgecolors='black', linewidth=0.3, s=35)
axes[1].set_title(f'dbscan (eps={mejor_eps})\nsilueta: {sil_dbscan_final:.4f}')
axes[1].set_xlabel('pc1')
axes[1].set_ylabel('pc2')
axes[1].legend(fontsize=8)

# jerarquico
for c in range(k_optimo):
    mask = labels_jerarquico == c
    axes[2].scatter(datos_pca[mask, 0], datos_pca[mask, 1],
                    c=[colores_clusters[c]], label=f'cluster {c}',
                    alpha=0.6, edgecolors='black', linewidth=0.3, s=35)
axes[2].set_title(f'jerarquico ward (k={k_optimo})\nsilueta: {sil_jerarquico:.4f}')
axes[2].set_xlabel('pc1')
axes[2].set_ylabel('pc2')
axes[2].legend(fontsize=8)

plt.suptitle('comparacion de los 3 algoritmos de clusterizacion (sobre pca 2d)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(graficos_dir, "19_comparacion_3_algoritmos.png"), dpi=150)
plt.close()

print(f"\nmejor modelo: {mejor_modelo} (silueta: {siluetas_modelos[mejor_modelo]:.4f})")
print(f"graficos exportados en: {graficos_dir}")
print("\nanalisis completado.")
