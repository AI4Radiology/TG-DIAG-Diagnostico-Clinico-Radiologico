import sys
from pathlib import Path
sys.path.append(str(Path(r"c:\Users\nsp1324\Documents\PDG II\TG-DIAG-Diagnostico-Clinico-Radiologico")))

from scripts.generar_orus_prueba import generar_oru

data = [
    {
        "report_id": "TCDIAG001_ACV",
        "tecnica": "Técnica estándar.",
        "hallazgos": "Área hipodensa córtico-subcortical frontoparietal izquierda con pérdida de diferenciación sustancia gris-blanca y borramiento de surcos adyacentes.",
        "opinion": "Hallazgos sugestivos de accidente cerebrovascular isquémico agudo en territorio de arteria cerebral media izquierda."
    },
    {
        "report_id": "TCDIAG002_HEMORRAGIA",
        "tecnica": "Técnica estándar.",
        "hallazgos": "Se identifica colección hiperdensa intraparenquimatosa temporal derecha compatible con hemorragia intracerebral aguda. Edema perilesional asociado.",
        "opinion": "Hallazgos compatibles con hemorragia intracraneal aguda."
    },
    {
        "report_id": "TCDIAG003_DESVIACION",
        "tecnica": "Técnica estándar.",
        "hallazgos": "Importante edema cerebral hemisférico izquierdo con efecto de masa asociado. Se observa desviación de la línea media hacia la derecha de aproximadamente 8 mm.",
        "opinion": "Desviación significativa de la línea media secundaria a efecto de masa intracraneal."
    },
    {
        "report_id": "TCDIAG004_FRACTURA",
        "tecnica": "Técnica estándar.",
        "hallazgos": "Fractura conminuta temporoparietal derecha con extensión a la base del cráneo. Pequeño neumoencéfalo asociado.",
        "opinion": "Fractura compleja de cráneo con compromiso de base craneal."
    },
    {
        "report_id": "TCDIAG005_NORMAL",
        "tecnica": "Técnica estándar.",
        "hallazgos": "Estudio tomográfico dentro de parámetros anatómicos normales. Sistema ventricular de tamaño y morfología conservados. Sin evidencia de colecciones extraaxiales ni alteraciones estructurales agudas.",
        "opinion": "Estudio tomográfico de cráneo sin alteraciones estructurales agudas."
    }
]

out_dir = Path(r"c:\Users\nsp1324\Documents\PDG II\TG-DIAG-Diagnostico-Clinico-Radiologico\data\orus_prueba")
out_dir.mkdir(parents=True, exist_ok=True)

for d in data:
    text = generar_oru(d, d["report_id"])
    file_path = out_dir / f'{d["report_id"]}.oru'
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
        
print("Archivos generados con el script nativo.")
