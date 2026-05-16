return
/**
 * Gmail MD5 Tagger
 * ─────────────────────────────────────────────────────────────────
 * Busca correos con una etiqueta dada, calcula el MD5 del subject
 * de cada uno y, si hay coincidencia en el mapa md5→etiquetas,
 * aplica esas etiquetas y añade una estrella al correo.
 *
 * CONFIGURACIÓN:
 *   1. Edita MD5_TAG_MAP con tus pares <md5, [etiquetas]>.
 *   2. Edita SOURCE_LABEL con el nombre de la etiqueta de búsqueda.
 *   3. Ejecuta la función `procesarCorreos()`.
 * ─────────────────────────────────────────────────────────────────
 */

// ── CONFIGURACIÓN ────────────────────────────────────────────────

/**
 * Mapa de md5 → array de etiquetas a aplicar.
 * Las etiquetas que no existan en Gmail se crearán automáticamente.
 *
 * Ejemplo:
 *   "a1b2c3d4e5f6...": ["proyecto/alpha", "urgente"],
 */
const MD5_TAG_MAP = {
  "5d41402abc4b2a76b9719d911017c592": ["proyecto/alpha", "urgente"],
  "7215ee9c7d9dc229d2921a40e899ec5f": ["newsletter", "revisar"],
  // ← añade aquí tus propios pares
};

/** Etiqueta cuya bandeja se recorrerá. */
const SOURCE_LABEL = "mi-etiqueta-origen";

/** Máximo de hilos por página (Gmail devuelve hasta 500). */
const PAGE_SIZE = 500;

// ── PUNTO DE ENTRADA ─────────────────────────────────────────────

function procesarCorreos() {
  const label = obtenerOCrearEtiqueta_(SOURCE_LABEL);
  if (!label) {
    Logger.log(`ERROR: No se encontró la etiqueta "${SOURCE_LABEL}".`);
    return;
  }

  let inicio = 0;
  let procesados = 0;
  let modificados = 0;

  Logger.log(`Iniciando procesamiento para la etiqueta: "${SOURCE_LABEL}"`);

  while (true) {
    const hilos = GmailApp.search(
      `label:${SOURCE_LABEL}`,
      inicio,
      PAGE_SIZE
    );

    if (hilos.length === 0) break;

    for (const hilo of hilos) {
      const mensajes = hilo.getMessages();

      for (const mensaje of mensajes) {
        procesados++;
        const subject = mensaje.getSubject() || "";
        const hash = md5_(subject);

        if (MD5_TAG_MAP.hasOwnProperty(hash)) {
          const etiquetasDestino = MD5_TAG_MAP[hash];
          Logger.log(
            `✓ Coincidencia | Subject: "${subject}" | MD5: ${hash} | Etiquetas: [${etiquetasDestino.join(", ")}]`
          );

          // Aplicar etiquetas al hilo completo
          for (const nombreEtiqueta of etiquetasDestino) {
            const etiqueta = obtenerOCrearEtiqueta_(nombreEtiqueta);
            hilo.addLabel(etiqueta);
          }

          // Marcar con estrella
          mensaje.star();
          modificados++;
        }
      }
    }

    inicio += hilos.length;
    if (hilos.length < PAGE_SIZE) break; // última página
  }

  Logger.log(
    `Finalizado. Mensajes procesados: ${procesados} | Modificados: ${modificados}`
  );
}

// ── UTILIDADES ───────────────────────────────────────────────────

/**
 * Obtiene una etiqueta de Gmail por nombre; la crea si no existe.
 * @param {string} nombre
 * @returns {GmailLabel}
 */
function obtenerOCrearEtiqueta_(nombre) {
  let etiqueta = GmailApp.getUserLabelByName(nombre);
  if (!etiqueta) {
    Logger.log(`Creando etiqueta nueva: "${nombre}"`);
    etiqueta = GmailApp.createLabel(nombre);
  }
  return etiqueta;
}

/**
 * Calcula el hash MD5 de un string y lo devuelve como hex lowercase.
 * Usa la API nativa de Apps Script (Utilities.computeDigest).
 *
 * @param {string} texto
 * @returns {string} hash MD5 en hexadecimal (32 caracteres)
 */
function md5_(texto) {
  const bytes = Utilities.computeDigest(
    Utilities.DigestAlgorithm.MD5,
    texto,
    Utilities.Charset.UTF_8
  );
  return bytes
    .map(b => {
      // computeDigest devuelve bytes con signo (-128..127)
      const hex = (b & 0xff).toString(16);
      return hex.length === 1 ? "0" + hex : hex;
    })
    .join("");
}

// ── HELPER DE PRUEBA ─────────────────────────────────────────────

/**
 * Función auxiliar para verificar el MD5 de un subject en particular.
 * Úsala desde el editor de Apps Script para generar hashes a incluir
 * en MD5_TAG_MAP.
 *
 * Ejemplo de uso:
 *   probarMd5("Re: Factura de enero");
 */
function probarMd5(subject) {
  const hash = md5_(subject || "");
  Logger.log(`Subject: "${subject}"\nMD5:     ${hash}`);
  return hash;
}
