"""Generador de mensajes HL7 ORU^R01 sintéticos para pruebas de TC-DIAG.

Reproduce el formato real AGFA PACS de la Fundación Valle del Lili (FVL).
Para uso exclusivo en pruebas de integración con Mirth Connect.
NO contiene datos reales de pacientes.
"""

from __future__ import annotations

import csv
import os
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Código y descripción del procedimiento TC craneal
# ---------------------------------------------------------------------------
TC_CRANEAL_CODE = "870100"
TC_CRANEAL_DESC = "TOMOGRAFÍA COMPUTADA DE CRÁNEO SIMPLE"
TC_CRANEAL_SHORT = "TC DE CRANEO"

# ---------------------------------------------------------------------------
# Datos ficticios de pacientes — solo para pruebas, NO son reales
# ---------------------------------------------------------------------------
FAKE_PATIENTS = [
    {"cedula": "1234567890", "apellido": "GARCIA",    "nombre": "CARLOS",    "dob": "19800315", "sex": "M", "telefono": "3001234567"},
    {"cedula": "9876543210", "apellido": "MARTINEZ",  "nombre": "ANA",       "dob": "19750820", "sex": "F", "telefono": "3109876543"},
    {"cedula": "1122334455", "apellido": "RODRIGUEZ", "nombre": "LUIS",      "dob": "19900110", "sex": "M", "telefono": "3201122334"},
    {"cedula": "5544332211", "apellido": "LOPEZ",     "nombre": "MARIA",     "dob": "19851225", "sex": "F", "telefono": "3005544332"},
    {"cedula": "6677889900", "apellido": "HERNANDEZ", "nombre": "JORGE",     "dob": "19700505", "sex": "M", "telefono": "3156677889"},
    {"cedula": "1029384756", "apellido": "GONZALEZ",  "nombre": "SANDRA",    "dob": "19951130", "sex": "F", "telefono": "3001029384"},
    {"cedula": "2938475610", "apellido": "TORRES",    "nombre": "ANDRES",    "dob": "19880720", "sex": "M", "telefono": "3202938475"},
    {"cedula": "3847561029", "apellido": "RAMIREZ",   "nombre": "PAOLA",     "dob": "19920415", "sex": "F", "telefono": "3153847561"},
    {"cedula": "4756102938", "apellido": "VARGAS",    "nombre": "FELIPE",    "dob": "19671003", "sex": "M", "telefono": "3004756102"},
    {"cedula": "5610293847", "apellido": "CASTILLO",  "nombre": "DIANA",     "dob": "19830618", "sex": "F", "telefono": "3115610293"},
    {"cedula": "6102938475", "apellido": "MORENO",    "nombre": "RICARDO",   "dob": "19910212", "sex": "M", "telefono": "3006102938"},
    {"cedula": "7192837465", "apellido": "JIMENEZ",   "nombre": "VALENTINA", "dob": "19990801", "sex": "F", "telefono": "3207192837"},
    {"cedula": "8273645192", "apellido": "PEREZ",     "nombre": "OSCAR",     "dob": "19780925", "sex": "M", "telefono": "3158273645"},
    {"cedula": "9364752810", "apellido": "SANCHEZ",   "nombre": "CAROLINA",  "dob": "19861114", "sex": "F", "telefono": "3009364752"},
    {"cedula": "0455861729", "apellido": "REYES",     "nombre": "MIGUEL",    "dob": "19730308", "sex": "M", "telefono": "3200455861"},
    {"cedula": "1546970638", "apellido": "MENDOZA",   "nombre": "LAURA",     "dob": "19940520", "sex": "F", "telefono": "3151546970"},
    {"cedula": "2637081547", "apellido": "ORTIZ",     "nombre": "CAMILO",    "dob": "19820714", "sex": "M", "telefono": "3002637081"},
    {"cedula": "3728192456", "apellido": "SILVA",     "nombre": "NATALIA",   "dob": "19971022", "sex": "F", "telefono": "3213728192"},
    {"cedula": "4819203365", "apellido": "ROJAS",     "nombre": "DAVID",     "dob": "19691118", "sex": "M", "telefono": "3164819203"},
    {"cedula": "5900314274", "apellido": "GUERRERO",  "nombre": "SOFIA",     "dob": "20010403", "sex": "F", "telefono": "3005900314"},
]

# ---------------------------------------------------------------------------
# Radiólogos ficticios en formato HL7: CODIGO^Apellido^Nombre
# ---------------------------------------------------------------------------
FAKE_MEDICOS = [
    "RAD001^VASQUEZ^JUAN",
    "RAD002^MONTOYA^CLAUDIA",
    "RAD003^ESCOBAR^PEDRO",
    "RAD004^RIOS^MARCELA",
    "RAD005^CARDENAS^HUGO",
]


def _hl7_datetime(dt: Optional[datetime] = None) -> str:
    """Formatea una fecha en el formato HL7: YYYYMMDDHHmmss."""
    return (dt or datetime.now()).strftime("%Y%m%d%H%M%S")


def _split_sentences(texto: str) -> list[str]:
    """Divide texto en oraciones simples por punto."""
    oraciones = [s.strip() for s in str(texto or "").split(".") if s.strip()]
    return oraciones if oraciones else [str(texto or "").strip()]


def generar_oru(row: dict, report_id: str) -> str:
    """Genera un mensaje HL7 ORU^R01 en formato AGFA PACS FVL.

    Args:
        row: Diccionario con tecnica, hallazgos, opinion, patologia.
        report_id: Identificador único del reporte.

    Returns:
        Mensaje HL7 completo como cadena con saltos de línea.
    """
    paciente = random.choice(FAKE_PATIENTS)
    medico = random.choice(FAKE_MEDICOS)
    dt_str = _hl7_datetime()
    msg_id = uuid.uuid4().hex.upper()[:20]
    orden = str(random.randint(1000000000, 9999999999))
    image_id = f"AGFA{uuid.uuid4().hex[:20].upper()}"

    segmentos: list[str] = []
    obx_counter = [0]

    def add_obx(seccion: str, texto: str) -> None:
        obx_counter[0] += 1
        n = obx_counter[0]
        segmentos.append(
            f"OBX|{n}|FT|{TC_CRANEAL_CODE}&GDT^{TC_CRANEAL_DESC}^^{image_id}|"
            f"{seccion}|{texto}||||||F|||{dt_str}||{medico}"
        )

    def add_obx_separator(seccion: str) -> None:
        obx_counter[0] += 1
        n = obx_counter[0]
        segmentos.append(
            f"OBX|{n}|FT|{TC_CRANEAL_CODE}&GDT^{TC_CRANEAL_DESC}^^{image_id}|"
            f"{seccion}|||||||F|||{dt_str}||{medico}"
        )

    # MSH
    segmentos.append(
        f"MSH|^~\\&|AGFA_PACS|AGFA|AGFA_PACS|RHAP_ORU|{dt_str}||ORU^R01^ORU_R01"
        f"|{msg_id}|P|2.4||||||UNICODE UTF-8"
    )

    # PID
    segmentos.append(
        f"PID|||{paciente['cedula']}^^^SYSTEM^PI||"
        f"{paciente['apellido']}^{paciente['nombre']}^^CC^^^^A||"
        f"{paciente['dob']}|{paciente['sex']}|||NO TIENE^^CALI^^^UKW^H||"
        f"{paciente['telefono']}^PRN^PH"
    )

    # ORC
    segmentos.append(
        f"ORC|RE|{orden}^TESCANOG|{orden}||CM"
    )

    # OBR
    segmentos.append(
        f"OBR|1|{orden}^TESCANOG|{orden}|"
        f"{TC_CRANEAL_CODE}^{TC_CRANEAL_DESC}^TESCANOG^{TC_CRANEAL_CODE}^{TC_CRANEAL_SHORT}^SYSTEM"
        f"|A|{dt_str}"
    )

    # OBX — RPSEC1 (Técnica)
    for oracion in _split_sentences(row.get("tecnica", "")):
        add_obx("RPSEC1", oracion)
    add_obx_separator("RPSEC1")

    # OBX — RPSEC2 (Hallazgos)
    for oracion in _split_sentences(row.get("hallazgos", "")):
        add_obx("RPSEC2", oracion)
    add_obx_separator("RPSEC2")

    # OBX — RPSEC3 (Opinión)
    for oracion in _split_sentences(row.get("opinion", "")):
        add_obx("RPSEC3", oracion)

    return "\n".join(segmentos)


def generar_todos_los_orus(
    excel_path: str,
    output_dir: str,
    n_muestras: Optional[int] = None,
) -> None:
    """Genera un archivo .hl7 por cada fila del Excel y un CSV resumen.

    Args:
        excel_path: Ruta al archivo Excel con los reportes.
        output_dir: Directorio de salida para los archivos .hl7.
        n_muestras: Si se especifica, limita la cantidad de ORUs generados.

    Returns:
        None.
    """
    df = pd.read_excel(excel_path)
    if "report_id" not in df.columns:
        df["report_id"] = [f"TC-{i:04d}" for i in range(len(df))]
    if n_muestras is not None:
        df = df.head(n_muestras)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    resumen: list[dict] = []
    generados = 0
    errores = 0

    for _, fila in df.iterrows():
        report_id = str(fila.get("report_id", ""))
        try:
            oru = generar_oru(fila.to_dict(), report_id)
            nombre_archivo = f"{report_id}.hl7"
            ruta = Path(output_dir) / nombre_archivo
            ruta.write_text(oru, encoding="utf-8")

            n_obx = sum(1 for linea in oru.splitlines() if linea.startswith("OBX|"))
            resumen.append({
                "filename": nombre_archivo,
                "report_id": report_id,
                "patologia_real": fila.get("patologia", ""),
                "n_obx_segments": n_obx,
            })
            generados += 1
        except Exception as exc:
            print(f"Error generando ORU para '{report_id}': {exc}")
            errores += 1

    csv_path = Path(output_dir) / "resumen_orus.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "report_id", "patologia_real", "n_obx_segments"])
        writer.writeheader()
        writer.writerows(resumen)

    print(f"ORUs generados: {generados} | Errores: {errores} | Resumen: {csv_path}")


if __name__ == "__main__":
    generar_todos_los_orus(
        excel_path="data/raw/reportes_tc_craneal.xlsx",
        output_dir="data/orus_prueba",
    )
