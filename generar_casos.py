import os

template = """MSH|^~\&|AGFA_PACS|AGFA|AGFA_PACS|RHAP_ORU|20250428064248||ORU^R01^ORU_R01|{id}|P|2.4||||||UNICODE UTF-8
PID|||1114953164^^^SYSTEM^PI||PACIENTE DE PRUEBA {id}^^^^A||20011123|M||||||||||||||||||||0
PV1||E|EURGADUL|A|||||||||||||0||||||||||||||||||||||||20250428000601
ORC|RE|0028469121^TESCANOG|0028469121||CM||^^^20250428003119^20250428003219^A||||||||||EURGADUL^UE Urgencias adulto^SYSTEM
OBR|1|0317272486^TESCANOG|0028469121|879301^TOMOGRAFIA COMPUTADA DE CRANEO^TESCANOG^879301^TAC DE CRANEO^SYSTEM|A|20250428000929|20250428000929||||||PRUEBA IA|||||0317272486|0317272486|0317272486||20250428064247||CT|F||^^^20250428003119^20250428003219^A||||^""||||||||||||||PRDCT5802
NTE|||""|PC
OBX|1|FT|879301&GDT^TOMOGRAFIA COMPUTADA DE CRANEO^^AGFA000000048717436|RPSEC204|TOMOGRAFIA COMPUTADA DE CRANEO||||||F|||20250428013441||SYSTEM
OBX|2|FT|879301&GDT^TOMOGRAFIA COMPUTADA DE CRANEO^^AGFA000000048717436|RPSEC1|Tecnica: TC de craneo simple.||||||F|||20250428013441||SYSTEM
OBX|3|FT|879301&GDT^TOMOGRAFIA COMPUTADA DE CRANEO^^AGFA000000048717436|RPSEC2|Hallazgos||||||F|||20250428013441||SYSTEM
OBX|4|FT|879301&GDT^TOMOGRAFIA COMPUTADA DE CRANEO^^AGFA000000048717436|RPSEC2|{hallazgos}||||||F|||20250428013441||SYSTEM
OBX|5|FT|879301&GDT^TOMOGRAFIA COMPUTADA DE CRANEO^^AGFA000000048717436|RPSEC3|Opinion||||||F|||20250428013441||SYSTEM
OBX|6|FT|879301&GDT^TOMOGRAFIA COMPUTADA DE CRANEO^^AGFA000000048717436|RPSEC3|{opinion}||||||F|||20250428013441||SYSTEM
"""

casos = [
    ('1001', 'hemorragia', 'Se observa extensa coleccion hiperdensa intraparenquimatosa a nivel de ganglios basales derechos con volumen aproximado de 40cc. No hay signos de isquemia aguda.', 'Hemorragia intracraneal aguda en ganglios basales derechos.'),
    ('1002', 'acv', 'Se evidencia pérdida de la diferenciación sustancia blanca y gris a nivel del territorio de la arteria cerebral media izquierda. Borramiento de surcos corticales.', 'Signos tomográficos sugestivos de evento isquémico agudo en territorio de la ACM izquierda.'),
    ('1003', 'desviacion', 'Presencia de lesión ocupante de espacio en hemisferio derecho que produce marcado efecto de masa. Se observa un desplazamiento de la línea media hacia la izquierda de 12 mm.', 'Efecto de masa severo con desviación de línea media.'),
    ('1004', 'fractura', 'Solución de continuidad ósea a nivel del hueso parietal derecho con hundimiento de 5 mm y neumoencéfalo asociado. No se observan hematomas epidurales.', 'Fractura deprimida parietal derecha con neumoencéfalo.')
]

output_dir = r'C:\Users\nsp1324\Documents\PDG II\TG-DIAG-Diagnostico-Clinico-Radiologico\data\HL7_Entrada'

for id_num, nombre, hallazgos, opinion in casos:
    content = template.format(id=id_num, hallazgos=hallazgos, opinion=opinion).replace('\n', '\r')
    file_path = os.path.join(output_dir, f'prueba_{nombre}.oru')
    with open(file_path, 'wb') as f:
        f.write(content.encode('utf-8'))
