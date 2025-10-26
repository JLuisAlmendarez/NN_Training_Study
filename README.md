# NN_Training_Study

Repositorio para el estudio de entrenamiento de redes neuronales en MNIST con PyTorch.  
Incluye experimentos de baseline, ablation studies y selección de modelo final.

---

## Contenido del repositorio

- `nn_training_study.ipynb` – Notebook principal con:
  - Setup y reproducibilidad (imports, seeds, device, flags determinísticos)
  - Datasets, transforms y DataLoaders
  - Modelos: AlexNet, VGG, CustomCNN con inicialización configurable
  - Funciones de entrenamiento y validación
  - LR Range Test
  - Entrenamiento baseline y ablations
  - Plots y métricas finales
- `results/` – Carpeta donde se guardan métricas, modelos y metadata
- `data/` – Descarga automática de MNIST
- `misc/` – Scripts auxiliares para gráficas o análisis adicionales
- `README.md` – Este archivo
- `requirements.txt` – Dependencias del proyecto

---

## Requisitos

- Python ≥ 3.10
- PyTorch ≥ 2.0
- torchvision
- ipywidgets, pillow, ipython
- matplotlib, numpy, pandas, tqdm
- Jupyter Notebook / Jupyter Lab

Instalar dependencias:

```bash
pip install -r requirements.txt
```
___

## Cómo ejecutar

1.	Abrir nn_training_study.ipynb en Jupyter Notebook.
2. Seleccionar el modelo, optimizador y hyperparámetros en los widgets.
3.	Ejecutar:
   * LR Range Test: para explorar tasas de aprendizaje iniciales.
   * Entrenamiento: para correr baselines y ablations.
4.	Los resultados se guardan automáticamente en results/<modelo>/exp_<n>/:
   * metrics.json: pérdidas y accuracies por época
   * metadata.json: configuración del experimento
   * model.pth

___
## ¿Qué se puede ajustar?

En `nn_training_study.ipynb` puedes modificar y experimentar con:

- **Modelo**: `AlexNet`, `VGG`, `CustomCNN`
- **Optimización**: `SGD`, `SGD+Nesterov`, `Adam`, `AdamW`
- **Tasa de aprendizaje (LR)**: escala logarítmica o constante con warmup
- **Weight decay**: regularización L2
- **Dropout**: activar/desactivar y configurar su tasa
- **Inicialización de pesos**: `Kaiming`, `Xavier`, `Orthogonal`
- **Label smoothing**: suavizado de etiquetas
- **LR Scheduler**: `constant+warmup`, `step`, `cosine`, `one_cycle`
- **Batch size**: 32, 128, 512
- **Data augmentation**: rotación, jitter de color, perspectiva
- **Protocolo de entrenamiento**: 
  - Fixed Epochs: número fijo de épocas
  - Fixed Wall-Clock Time: limitar entrenamiento a cierto tiempo
- **Semilla (seed)**: reproducibilidad de experimentos

