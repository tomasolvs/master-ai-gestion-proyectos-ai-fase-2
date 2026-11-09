# Fase 2 — Priorización predictiva de mantenimiento | Grupo Bimbo (_proyecto dummy educativo_)

* Nombre: Tomás Olivares
* Nombre del curso: Gestión de proyectos de inteligencia artificial
* Nombre del profesor: Luis Ariel Vázquez Piña
* Módulo: 2 - Semana: 8
* Fecha:  8/11/2026
* Fase 2 — Priorización predictiva de mantenimiento | Grupo Bimbo

Prototipo académico de una capa de IA para priorizar señales de riesgo de falla de activos industriales. Continúa el business case de Fase 1: combina condición del activo, historial de mantenimiento y criticidad. La IA se mantiene sólo como "recomendador"; el técnico conserva la decisión final.

> La actividad "Fase 1" definió el business case, pero no entregó un modelo entrenado, dataset ni API. Este repositorio implementa un prototipo demostrativo nuevo con datos sintéticos reproducibles. No utiliza datos internos de Grupo Bimbo.

## 🔗 Links relevantes
* [Este repositorio _para fines de entrega_](https://github.com/tomasolvs/master-ai-gestion-proyectos-ai-fase-2)
* [Validación y pruebas + Manual de despliegue en nube](https://github.com/tomasolvs/master-ai-gestion-proyectos-ai-fase-2/blob/main/docs/081126%20TODDL%20Fase%202%20M%C3%B3dulo%202%20Gesti%C3%B3n%20de%20proyectos%20de%20inteligencia%20artificial.pdf)
* [Evidencia de Notebook ejecutado](https://github.com/tomasolvs/master-ai-gestion-proyectos-ai-fase-2/tree/main/evidence/notebook)
* [Evidencia de build y ejecución de Docker](https://github.com/tomasolvs/master-ai-gestion-proyectos-ai-fase-2/tree/main/evidence/docker)

## Arquitectura

Navegador (frontend HTML/JS) → API FastAPI (`POST /predict`) → pipeline Scikit-learn / Random Forest → scoring de riesgo y factores explicativos. El servicio expone `/health`, `/model/metrics` y `/docs`.

## Requisitos

- Python 3.11 o Docker Desktop / Docker Engine.
- 2 GB RAM recomendados para entrenamiento inicial.
- Puerto local 8080  disponible. (Alternativamente, se ha probado con el 8000 desde Colab, no localmente)

## Ejecución local sin Docker

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts_train.py
uvicorn app.main:app --host 127.0.0.1 --port 8080
```

Abrir http://127.0.0.1:8080 y http://127.0.0.1:8080/docs.

## Construcción y ejecución Docker

```bash
docker build -t bimbo-maintenance-ai:1.0 .
docker run --rm -p 8080:8080 --name maintenance-ai bimbo-maintenance-ai:1.0
```

Verificar:
```bash
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"asset_id":"MOTOR-001","temperature_c":92,"vibration_mm_s":8.2,"pressure_bar":6.5,"energy_kw":48,"operating_hours":18000,"work_orders_90d":5,"days_since_maintenance":220,"asset_criticality":5}'
```

## Pruebas

```bash
pytest -q
```

## Variables del modelo

Temperatura, vibración, presión, consumo energético, horas de operación, órdenes de trabajo en 90 días, días desde último mantenimiento y criticidad (1–5). Las etiquetas se generan mediante una función probabilística sintética para demostrar el flujo ML; no son fallas históricas reales.

## Limitaciones y uso responsable
- Uso meramente educativo.
- No conectar a redes OT/IT productivas ni a CMMS real.
- No usar para decisiones de seguridad industrial.
- No crear órdenes de trabajo automáticamente.
- Validar con datos históricos representativos, evaluación temporal, calibración, análisis por tipo de activo y aprobación de Operaciones/Seguridad antes de cualquier piloto real.
- Los umbrales explicativos son reglas didácticas exclusivamente.
- El score es experimental y no debe interpretarse como probabilidad operacional calibrada o como un dato fiable por si solo.

## Estructura

```text
app/                 API, esquema, modelo y frontend
tests/               pruebas de contrato y validación
docs/                manual cloud e informe de pruebas
evidence/            evidencia de notebook, construcción y ejecución de docker, y de creación de este repositorio
Dockerfile           imagen reproducible
requirements.txt     dependencias fijadas
scripts_train.py     entrenamiento explícito
```
