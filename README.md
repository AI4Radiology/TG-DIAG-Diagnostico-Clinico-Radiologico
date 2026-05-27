# PDG — Diagnóstico Clínico de Hallazgos Radiológicos Críticos

Sistema de aprendizaje automático para la clasificación automática de hallazgos radiológicos críticos y su evaluación diagnóstica, desarrollado como Proyecto de Grado en la Universidad ICESI.

---

## ¿Qué hace este proyecto?

El sistema recibe el texto libre de un informe radiológico y determina si el hallazgo es **Crítico** o **No Crítico**. Cuando se detecta un hallazgo crítico, el caso avanza a la etapa de evaluación de diagnóstico diferencial para identificar la patología específica.

```
Informe radiológico (texto libre)
        │
        ▼
┌─────────────────────┐
│   ¿Es CRÍTICO?      │──── No Crítico ──► Flujo estándar
│   (Clasificación)   │
└─────────────────────┘
        │ Crítico
        ▼
┌─────────────────────┐
│   ¿Qué diagnóstico? │──► Evaluación de diagnóstico diferencial
│   (Identificación)  │
└─────────────────────┘
```

---

## Pipeline de notebooks

Ejecutar en orden. Cada notebook es independiente y carga su propio estado desde los archivos de datos procesados.

| # | Notebook | Descripción |
|---|---|---|
| 0.0 | `0.0_Analisis_y_Transformacion.ipynb` | Exploración, limpieza y normalización del dataset de informes clínicos |
| 0.1 | `0.1_Inferencia_RAD_ALERT.ipynb` | Filtro de casos críticos mediante inferencia externa (Triaje) |
| 1 | `1_Modelos_Tradicionales_Baseline.ipynb` | Entrenamiento y evaluación de Baseline sobre datos desbalanceados |
| 2 | `2_Modelo_LSTM_Baseline.ipynb` | Red neuronal LSTM entrenada como Baseline inicial |
| 3.0 | `3.0_Balanceo_SMOTE_ROS.ipynb` | Solución 1 al desbalance: Oversampling clásico (SMOTE y ROS) |
| 3.1 | `3.1_Augmentation_LLM.ipynb` | Solución 2 (Estado del Arte): Generación de datos sintéticos con LLM (Claude API) |
| 4 | `4_Modelo_XGBoost_Balanceado.ipynb` | Clasificador XGBoost evaluado sobre los datos aumentados |
| 5 | `5_Comparacion_Final.ipynb` | Comparación de métricas de todos los modelos y selección del mejor |
| 6 | `6_Modelo_Transformers.ipynb` | Fine-tuning de arquitecturas tipo Transformer *(requiere GPU)* |

---

## Resultado principal

El modelo seleccionado es **SVM** (Kernel RBF, validación cruzada 5-fold), por obtener el mejor equilibrio entre precisión, recall y reproducibilidad. Fue entrenado sobre el conjunto balanceado con SMOTE.

---

## Instalación y ejecución

### Requisitos previos
- Python 3.12
- `pip` actualizado: `python -m pip install --upgrade pip`

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/isabelaocampos/PDG-Diagnostico-Clinico
cd PDG-Diagnostico-Clinico

# 2. Crear y activar entorno virtual
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Abrir los notebooks
jupyter notebook
```

> ⚠️ TensorFlow ≥ 2.16 es obligatorio para Python 3.12. El notebook `7_Modelo_RoBERTa` requiere GPU para ejecutarse completo.

---

## Estructura del repositorio

```
PDG-Diagnostico-Clinico/
├── .env.example
├── requirements.txt
├── README.md
└── notebooks/
    ├── 0.0_Analisis_y_Transformacion.ipynb
    ├── 0.1_Inferencia_RAD_ALERT.ipynb
    ├── 1_Modelos_Tradicionales_Baseline.ipynb
    ├── 2_Modelo_LSTM_Baseline.ipynb
    ├── 3.0_Balanceo_SMOTE_ROS.ipynb
    ├── 3.1_Augmentation_LLM.ipynb
    ├── 4_Modelo_XGBoost_Balanceado.ipynb
    ├── 5_Comparacion_Final.ipynb
    └── 6_Modelo_Transformers.ipynb
```

---

## Licencia

Uso académico — Proyecto de Grado, Universidad ICESI.