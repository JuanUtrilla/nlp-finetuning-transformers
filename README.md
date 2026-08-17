# NLP — fine-tuning de transformers: clasificación y question answering

Dos trabajos de *fine-tuning* de modelos tipo BERT/RoBERTa con **Hugging Face
Transformers** y PyTorch, sobre dos tareas clásicas de comprensión del lenguaje.

Desarrollado en el módulo de *Introducción al Procesamiento de Lenguaje Natural* del
Máster en Big Data, Data Science e Inteligencia Artificial (Universidad Complutense de
Madrid, curso 2025-26).

## 1. Clasificación — inferencia textual sobre MNLI (GLUE)

[`01_clasificacion_mnli.ipynb`](01_clasificacion_mnli.ipynb)

El split **MNLI** de GLUE presenta pares premisa–hipótesis que hay que clasificar en tres
relaciones: *entailment* (la hipótesis se deduce de la premisa), *neutral* (compatible
pero no deducible) y *contradiction*.

- *Fine-tuning* de un modelo preentrenado sobre el corpus filtrado a premisas cortas.
- Tokenización de pares de secuencias, con la separación premisa/hipótesis que el modelo
  necesita para distinguir ambos segmentos.
- Evaluación por *accuracy* y análisis de en qué clase se concentran los errores —
  *neutral* es sistemáticamente la más difícil, por estar entre las otras dos.

## 2. Question Answering extractivo — SQuAD 2.0

[`02_question_answering_squad2.ipynb`](02_question_answering_squad2.ipynb)

*Fine-tuning* de **RoBERTa** sobre SQuAD 2.0: dadas una pregunta y un contexto, localizar
el fragmento exacto que la responde — o determinar que **no existe respuesta**, que es lo
que diferencia SQuAD 2.0 de la versión 1.1 y donde fallan los modelos ingenuos.

- **Análisis exploratorio previo a modelar**: se estudia la distribución de longitudes
  para elegir `MAX_LENGTH` con criterio en lugar de por defecto. Con el percentil 95 del
  contexto por debajo de 295 caracteres, la longitud de tokenización se dimensiona a esa
  medida: truncar de más pierde la respuesta, truncar de menos desperdicia cómputo.
- Mapeo de las posiciones de respuesta de caracteres a *tokens*, gestionando el
  desplazamiento (`offset_mapping`) y las preguntas sin respuesta.
- Comparación de *checkpoints* por métricas **Exact Match y F1**.

## Ejecución

Ambos notebooks requieren GPU. Están preparados para **Google Colab**; en local:

```bash
pip install transformers datasets evaluate torch accelerate
jupyter lab
```

Los *checkpoints* entrenados no se incluyen en el repositorio por tamaño (varios GB); los
notebooks los regeneran desde cero.

## Stack

PyTorch · Hugging Face Transformers · Datasets · Evaluate · Python

---

**Juan Peñas Utrilla** — [LinkedIn](https://www.linkedin.com/in/juan-penas-utrilla/) · [GitHub](https://github.com/JuanUtrilla)
