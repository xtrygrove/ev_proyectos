# Evaluación de Proyectos – Streamlit (MVP)

## 1) Crear entorno e instalar

### Opción A: Conda (recomendado)
Desde la raíz del repo:
```bash
conda env create -f environment.yml
conda activate eval_proyectos
```

### Opción B: venv (alternativa)
Desde la raíz del repo:
```bash
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

python -m pip install -r project_eval_app/requirements.txt
```

## 2) Ejecutar la app
Desde la raíz del repo:
```bash
streamlit run project_eval_app/app.py
```
