# Predicción de Popularidad de Canciones de Spotify

**Análisis Predictivo – Examen 3 (ITBA)**
**Moira Patricia Clavin – Legajo 64959**
**Competencia Kaggle:** AP2026Q1
**Public Score Final:** 0.358

---

## Objetivo

El objetivo de este proyecto es predecir la variable **track_popularity** (escala 0-100) para canciones de Spotify utilizando características de audio y metadata asociada a cada canción.

La competencia fue desarrollada sobre un conjunto de entrenamiento con etiquetas conocidas y un conjunto de test sin etiquetas, para el cual debían generarse las predicciones enviadas a Kaggle.

---

## Estructura del repositorio

### entrenar_modelo_final.ipynb

Notebook encargado de:

* Cargar los datos de entrenamiento.
* Realizar el feature engineering.
* Entrenar los modelos finales.
* Guardar los modelos entrenados.

Al ejecutarse desde cero permite reproducir completamente el entrenamiento del modelo utilizado en la entrega final.

### aplicar_modelo_test.ipynb

Notebook encargado de:

* Cargar los modelos previamente entrenados.
* Aplicar exactamente el mismo feature engineering al conjunto de test.
* Generar las predicciones finales.
* Construir el archivo de submission enviado a Kaggle.

Al ejecutarse permite reproducir exactamente el archivo entregado en la competencia.

---

## Descripción del dataset

El dataset contiene canciones de Spotify junto con características musicales calculadas automáticamente por la plataforma.

### Conjunto de entrenamiento

* 26.266 observaciones
* 24 variables
* Incluye la variable objetivo: `track_popularity`

### Conjunto de test

* 6.567 observaciones
* 23 variables
* No contiene `track_popularity`

### Variables disponibles

#### Audio Features

* danceability
* energy
* loudness
* speechiness
* acousticness
* instrumentalness
* liveness
* valence
* tempo
* duration_ms
* key
* mode

#### Metadata

* artista
* álbum
* fecha de lanzamiento
* género
* subgénero
* playlist

---

## Hallazgos del análisis exploratorio

Durante el EDA se identificó el principal desafío de la competencia.

### Distribution Shift entre Train y Test

Los conjuntos no fueron separados aleatoriamente.

#### Géneros presentes en Train

* rap
* pop
* latin
* rock
* r&b

#### Géneros presentes en Test

* edm
* r&b

El género EDM representa aproximadamente el 92% del conjunto de test y no aparece en entrenamiento.

Esto implica que el modelo debe generalizar hacia un género completamente no observado durante el aprendizaje.

### Ordenamiento del conjunto de test

También se observó que:

* Las primeras 524 filas corresponden a r&b.
* Las siguientes 6043 filas corresponden a edm.

Este hallazgo fue utilizado posteriormente en la estrategia final de modelado.

---

## Estrategia de validación

### ¿Por qué no se utilizó KFold tradicional?

KFold mezcla observaciones aleatoriamente.

En este problema eso habría generado una validación optimista, ya que cada fold contendría ejemplos de todos los géneros.

Sin embargo, el problema real consistía en predecir un género ausente en entrenamiento.

### GroupKFold por género

Se utilizó GroupKFold agrupando por `playlist_genre`.

En cada iteración se deja un género completo fuera del entrenamiento.

Ejemplo:

* Entrena con pop, latin, rock y r&b → valida sobre rap.
* Entrena con rap, latin, rock y r&b → valida sobre pop.

De esta forma la validación reproduce de manera mucho más realista el desafío de generalización presente en la competencia.

---

## Modelo Baseline

Se utilizó un:

### DummyRegressor (strategy = "mean")

Este modelo:

* No utiliza ninguna feature.
* Predice siempre la popularidad promedio observada en train.

Resultados:

* R² = -0.0053
* RMSE = 25.06

Su función fue establecer un punto de referencia mínimo para evaluar si los modelos posteriores realmente aprendían información útil.

---

## Modelos evaluados

Se compararon distintos algoritmos utilizando exactamente el mismo esquema de validación GroupKFold.

| Modelo            | R²    |
| ----------------- | ----- |
| Ridge             | 0.014 |
| Lasso             | 0.013 |
| AdaBoost          | 0.130 |
| Gradient Boosting | 0.194 |
| Random Forest     | 0.253 |
| ExtraTrees        | 0.278 |

### Conclusiones

#### Ridge y Lasso

Modelos lineales.

Obtuvieron bajo desempeño debido a que la relación entre las características musicales y la popularidad es altamente no lineal.

#### AdaBoost y Gradient Boosting

Modelos de boosting que construyen árboles secuencialmente corrigiendo errores previos.

Mejoraron significativamente respecto a los modelos lineales, pero mostraron menor capacidad de generalización frente a géneros no vistos.

#### Random Forest

Ensamble de árboles independientes entrenados sobre subconjuntos aleatorios de datos.

Capturó relaciones complejas y mejoró considerablemente el desempeño.

#### ExtraTrees

Extremely Randomized Trees.

Introduce mayor aleatoriedad en la construcción de los árboles, generando modelos más diversos y reduciendo el riesgo de sobreajuste.

Dado que el principal desafío era generalizar hacia EDM, ExtraTrees resultó ser el algoritmo más robusto y obtuvo el mejor desempeño.

---

## Feature Engineering

A partir de las variables originales se construyeron nuevas features.

### Audio original (12)

Variables de audio provistas por Spotify.

### Fecha de lanzamiento (5)

* release_year
* release_age_days
* release_date_missing
* release_year_only
* release_jan1

### Interacciones de audio (8)

* energy_x_loudness
* valence_x_danceability
* energy_minus_acousticness
* duration_min
* loudness_norm
* instrumental_energy
* dance_energy
* acoustic_valence

### Frecuencias (2)

* artist_freq
* album_freq

### Meta-feature (1)

* genre_seen_in_train

### Total

28 features finales.

---

## Optimización de hiperparámetros

Una vez seleccionado ExtraTrees se utilizó GridSearchCV.

### ¿Qué hace GridSearchCV?

Prueba automáticamente múltiples combinaciones de hiperparámetros y evalúa cada una utilizando el esquema de validación definido.

Se evaluaron combinaciones de:

* n_estimators
* min_samples_leaf
* max_features

La configuración seleccionada fue:

```python
ExtraTreesRegressor(
    n_estimators=500,
    min_samples_leaf=2,
    max_features=0.7,
    random_state=42
)
```

Por obtener el mejor R² promedio bajo GroupKFold.

## Modelo Final

Se construyeron dos modelos:

### Modelo A – General

Entrenado con todas las observaciones disponibles.

* 26.266 canciones
* 5 géneros

R² GroupKFold:

0.2765

### Modelo B – Especialista en r&b

Entrenado únicamente con canciones r&b.

* 4.907 canciones

R² sobre r&b:

0.4222

---

## Estrategia híbrida final

Dado que r&b es el único género presente tanto en train como en test:

### Canciones r&b

Predicción final:

70% Modelo B + 30% Modelo A

### Canciones EDM

Predicción final:

100% Modelo A

Esta estrategia permitió aprovechar la especialización del modelo B sin perder capacidad de generalización.

---

## Resultados

### Public Score Kaggle

0.358

### Evolución

* Voting Ensemble: 0.339
* ExtraTrees Tuneado: 0.356
* Modelo Híbrido Final: 0.358

---

## Limitaciones

### Ausencia de EDM en entrenamiento

El 92% del test corresponde a un género completamente ausente en train.

### Falta de información externa

No se dispone de:

* Seguidores del artista
* Reproducciones históricas
* Métricas de streaming
* Popularidad histórica

### Información temporal limitada

Sólo se cuenta con la fecha de lanzamiento y no con la evolución de popularidad a través del tiempo.

---

## Posibles mejoras

* Incorporar ejemplos adicionales de EDM.
* Agregar variables externas relacionadas con artistas.
* Incorporar información histórica de reproducciones.
* Explorar estrategias avanzadas de adaptación entre dominios.

---

## Resultado final

**Modelo seleccionado:** ExtraTrees + Especialización por género

**Public Score Kaggle:** 0.358
