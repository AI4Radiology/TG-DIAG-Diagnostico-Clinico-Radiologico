// File: mirth/transformaciones/construir_json.js
// Construye el payload JSON que será enviado a la API FastAPI de TC-DIAG.
// Se ejecuta después de extraer_obx.js en el canal de Mirth Connect.
// Mantenido por el equipo clínico — comentarios en español.

try {
    logger.info('TC-DIAG: Iniciando construcción del payload JSON');

    // --- Verificar si el mensaje corresponde a TC craneal ---
    // Si no es TC craneal, se eliminan todos los destinos para no enviar a la API
    var esTcCraneal = String(channelMap.get('es_tc_craneal') || 'false');
    if (esTcCraneal === 'false') {
        logger.info('TC-DIAG: Mensaje no corresponde a TC craneal. Descartando envío a la API.');
        destinationSet.removeAll();
        return;
    }

    // --- Recuperar campos extraídos por extraer_obx.js ---
    var reportId               = String(channelMap.get('report_id')                || '');
    var hallazgos              = String(channelMap.get('hallazgos')                || '');
    var opinion                = String(channelMap.get('opinion')                  || '');
    var tecnica                = String(channelMap.get('tecnica')                  || '');
    var codigoProcedimiento    = String(channelMap.get('codigo_procedimiento')     || '');
    var descripcionProcedimiento = String(channelMap.get('descripcion_procedimiento') || '');

    // --- Validar campo hallazgos (obligatorio para el clasificador) ---
    if (!hallazgos || hallazgos.trim() === '') {
        logger.warn('TC-DIAG: Campo hallazgos vacío para report_id=' + reportId +
                    '. Se enviará como cadena vacía — el clasificador puede rechazarlo.');
        hallazgos = '';
    }

    // --- Construir el objeto JSON según el esquema ReporteRequest de la API ---
    var payload = {
        report_id:                  reportId,
        hallazgos:                  hallazgos,
        opinion:                    opinion,
        tecnica:                    tecnica,
        datos_clinicos:             '',
        fuente:                     'HL7',
        codigo_procedimiento:       codigoProcedimiento,
        descripcion_procedimiento:  descripcionProcedimiento
    };

    var jsonPayload = JSON.stringify(payload);
    channelMap.put('jsonPayload', jsonPayload);

    // --- Log resumido para depuración (máximo 100 caracteres de hallazgos) ---
    var hallazgosLog = hallazgos.length > 100
        ? hallazgos.substring(0, 100) + '...'
        : hallazgos;

    logger.info('TC-DIAG: Payload construido para report_id=' + reportId +
                ' | hallazgos=' + hallazgosLog);

} catch (error) {
    logger.error('TC-DIAG: Error al construir el JSON de salida: ' + error);
    throw error;
}
