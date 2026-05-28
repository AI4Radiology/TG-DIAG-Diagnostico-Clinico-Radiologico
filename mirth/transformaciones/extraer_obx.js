// File: mirth/transformaciones/extraer_obx.js
// Extrae campos clínicos de mensajes HL7 ORU^R01 en formato AGFA PACS (FVL).
// OBX-4 usa códigos de sección: RPSEC1 = técnica, RPSEC2 = hallazgos, RPSEC3 = opinión.
// Mantenido por el equipo clínico — comentarios en español.

// ---------------------------------------------------------------------------
// CONFIGURACIÓN: agregar aquí los códigos de procedimiento de TC de cráneo
// de la institución. Si se deja vacío, se filtra solo por descripción.
// Ejemplo: var CODIGOS_TC_CRANEAL = ["870100", "870101"];
// ---------------------------------------------------------------------------
var CODIGOS_TC_CRANEAL = ["879111"];

// ---------------------------------------------------------------------------
// Función auxiliar: elimina tildes para comparación de texto
// ---------------------------------------------------------------------------
function normalizeStr(s) {
    return String(s)
        .replace(/á/g, 'a').replace(/é/g, 'e').replace(/í/g, 'i')
        .replace(/ó/g, 'o').replace(/ú/g, 'u').replace(/ñ/g, 'n')
        .replace(/Á/g, 'a').replace(/É/g, 'e').replace(/Í/g, 'i')
        .replace(/Ó/g, 'o').replace(/Ú/g, 'u').replace(/Ñ/g, 'n')
        .toLowerCase();
}

try {
    logger.info('TC-DIAG: Iniciando extracción de segmentos HL7 ORU^R01');

    // --- Extraer identificador único del reporte (MSH-10) ---
    var reportId = msg['MSH']['MSH.10']['MSH.10.1'].toString().trim();
    logger.info('TC-DIAG: report_id = ' + reportId);

    // --- Extraer código y descripción del procedimiento (OBR-4.1 y OBR-4.2) ---
    var codigoProcedimiento = '';
    var descripcionProcedimiento = '';
    try {
        codigoProcedimiento = msg['OBR']['OBR.4']['OBR.4.1'].toString().trim();
        descripcionProcedimiento = msg['OBR']['OBR.4']['OBR.4.2'].toString().trim();
    } catch (e) {
        logger.warn('TC-DIAG: No se pudo leer OBR-4: ' + e);
    }

    var descNorm = normalizeStr(descripcionProcedimiento);
    logger.info('TC-DIAG: codigo=' + codigoProcedimiento + ' | descripcion=' + descripcionProcedimiento);

    // --- Filtro 1: por código de procedimiento (si la lista está configurada) ---
    var esPorCodigo = false;
    if (CODIGOS_TC_CRANEAL.length > 0) {
        for (var c = 0; c < CODIGOS_TC_CRANEAL.length; c++) {
            if (codigoProcedimiento === CODIGOS_TC_CRANEAL[c]) {
                esPorCodigo = true;
                break;
            }
        }
    }

    // --- Filtro 2: por descripción normalizada (sin tildes) ---
    var terminosTcCraneal = ['craneo', 'craneal', 'cerebral', 'encefalo', 'tc craneo', 'tac craneo'];
    var esPorDescripcion = false;
    for (var t = 0; t < terminosTcCraneal.length; t++) {
        if (descNorm.indexOf(terminosTcCraneal[t]) >= 0) {
            esPorDescripcion = true;
            break;
        }
    }

    // Si no coincide ningún filtro, marcar como no craneal y detener
    if (!esPorCodigo && !esPorDescripcion) {
        logger.info('TC-DIAG: Estudio no corresponde a TC craneal. Descripcion: ' + descripcionProcedimiento);
        channelMap.put('es_tc_craneal', 'false');
        return false;
    }

    // --- Extraer segmentos OBX usando códigos de sección RPSEC1/RPSEC2/RPSEC3 ---
    // Formato AGFA PACS FVL: OBX-4 = código de sección, OBX-5 = texto
    var tecnica    = '';
    var hallazgos  = '';
    var opinion    = '';

    var numObx = msg['OBX'].length();
    logger.info('TC-DIAG: Procesando ' + numObx + ' segmentos OBX');

    for (var i = 0; i < numObx; i++) {
        // OBX-4 contiene el código de sección (RPSEC1, RPSEC2, RPSEC3)
        var seccion = msg['OBX'][i]['OBX.4']['OBX.4.1'].toString().trim().toUpperCase();

        // OBX-5 contiene el texto clínico
        var valor = msg['OBX'][i]['OBX.5']['OBX.5.1'].toString().trim();

        // Omitir líneas vacías o separadores vacíos (formato AGFA PACS)
        if (!valor || valor === '""') {
            continue;
        }

        if (seccion === 'RPSEC1') {
            // RPSEC1 = Técnica radiológica
            tecnica += ' ' + valor;
        } else if (seccion === 'RPSEC2') {
            // RPSEC2 = Hallazgos
            hallazgos += ' ' + valor;
        } else if (seccion === 'RPSEC3') {
            // RPSEC3 = Opinión / Conclusión
            opinion += ' ' + valor;
        }
    }

    // --- Guardar resultados en channelMap para el siguiente transformador ---
    channelMap.put('report_id',                reportId);
    channelMap.put('tecnica',                  tecnica.trim());
    channelMap.put('hallazgos',                hallazgos.trim());
    channelMap.put('opinion',                  opinion.trim());
    channelMap.put('es_tc_craneal',            'true');
    channelMap.put('codigo_procedimiento',     codigoProcedimiento);
    channelMap.put('descripcion_procedimiento', descripcionProcedimiento);

    logger.info('TC-DIAG: Extracción completada para report_id=' + reportId +
                ' | hallazgos=' + hallazgos.trim().substring(0, 80) + '...');

} catch (error) {
    logger.error('TC-DIAG: Error al extraer segmentos OBX: ' + error);
    throw error;
}
