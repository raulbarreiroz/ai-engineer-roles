# AI Engineer - Junior

## Definition
Implementa APIs de modelos pre-entrenados (OpenAI, Hugging Face), realiza fine-tuning básico y hace limpieza de datasets usando librerías estándar (sklearn).

## Specific Project
API REST que recibe reseñas de clientes y devuelve sentimiento (positivo/negativo/neutral) usando un modelo BERT pre-entrenado de Hugging Face.

## Core Concepts
Hugging Face pipeline, tokenización (AutoTokenizer), serialización de modelos con pickle/joblib, FastAPI (endpoints síncronos/asíncronos), manejo de requests concurrentes, límite de tasa (rate limiting).

## Recommended Modern Technologies
Hugging Face Transformers + PEFT (LoRA básico), FastAPI + Uvicorn, PyTorch 2.4, Sentence-Transformers, ChromaDB o FAISS, MLflow.
