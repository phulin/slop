import { asyncBufferFromUrl, cachedAsyncBuffer, parquetReadObjects } from "hyparquet";
import { compressors } from "hyparquet-compressors";

let docFile = null;
let sparseFile = null;
const docRowCache = new Map();
let allDocRowsPromise = null;
const sparseRowCache = new Map();
const activationCache = new Map();
const promptActivationCache = new Map();

function numeric(value, fallback = 0) {
  if (typeof value === "bigint") return Number(value);
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

async function openFile(url) {
  return cachedAsyncBuffer(await asyncBufferFromUrl({ url }));
}

async function readDocRow(rowIndex) {
  if (allDocRowsPromise) {
    const rows = await allDocRowsPromise;
    return rows[rowIndex] ?? null;
  }
  const key = String(rowIndex);
  if (!docRowCache.has(key)) {
    docRowCache.set(key, (async () => {
      const rows = await parquetReadObjects({
        file: docFile,
        compressors,
        rowStart: rowIndex,
        rowEnd: rowIndex + 1,
      });
      return rows[0] ?? null;
    })());
  }
  return docRowCache.get(key);
}

async function preloadDocRows() {
  if (!allDocRowsPromise) {
    allDocRowsPromise = parquetReadObjects({
      file: docFile,
      compressors,
    }).then((rows) => {
      rows.forEach((row, index) => {
        docRowCache.set(String(index), Promise.resolve(row));
      });
      return rows;
    });
  }
  return allDocRowsPromise;
}

async function readSparseRows(start, end) {
  const key = `${start}:${end}`;
  if (!sparseRowCache.has(key)) {
    sparseRowCache.set(key, parquetReadObjects({
      file: sparseFile,
      compressors,
      columns: ["latent_ids", "activations", "token_index"],
      rowStart: start,
      rowEnd: end,
    }));
  }
  return sparseRowCache.get(key);
}

function activationForLatent(row, latentId) {
  const ids = row.latent_ids ?? [];
  const values = row.activations ?? [];
  for (let index = 0; index < ids.length; index += 1) {
    if (numeric(ids[index]) === latentId) return numeric(values[index]);
  }
  return 0;
}

async function readActivationMap({ docId, latentId, start, end }) {
  const key = `${latentId}:${docId}`;
  if (!activationCache.has(key)) {
    activationCache.set(key, (async () => {
      const sparseRows = await readSparseRows(start, end);
      const entries = [];
      let maxActivation = 0;
      let maxTokenIndex = -1;
      for (const sparseRow of sparseRows) {
        const tokenIndex = numeric(sparseRow.token_index);
        const value = activationForLatent(sparseRow, latentId);
        entries.push([tokenIndex, value]);
        if (value > maxActivation) {
          maxActivation = value;
          maxTokenIndex = tokenIndex;
        }
      }
      return { entries, maxActivation, maxTokenIndex };
    })());
  }
  return activationCache.get(key);
}

async function readPromptActivationMap({ turnId, latentId, start, end, docs }) {
  const key = `${latentId}:${turnId}`;
  if (!promptActivationCache.has(key)) {
    promptActivationCache.set(key, (async () => {
      const sparseRows = await readSparseRows(start, end);
      const sortedDocs = [...(docs ?? [])]
        .map((doc) => ({
          docId: doc.docId,
          start: numeric(doc.start),
          end: numeric(doc.end),
        }))
        .filter((doc) => doc.docId && doc.end > doc.start)
        .sort((a, b) => a.start - b.start);
      const byDoc = Object.fromEntries(sortedDocs.map((doc) => [
        doc.docId,
        { entries: [], maxActivation: 0, maxTokenIndex: -1 },
      ]));
      let docOffset = 0;
      for (let rowOffset = 0; rowOffset < sparseRows.length; rowOffset += 1) {
        const globalRow = start + rowOffset;
        while (docOffset < sortedDocs.length && globalRow >= sortedDocs[docOffset].end) {
          docOffset += 1;
        }
        const doc = sortedDocs[docOffset];
        if (!doc || globalRow < doc.start || globalRow >= doc.end) continue;
        const sparseRow = sparseRows[rowOffset];
        const tokenIndex = numeric(sparseRow.token_index);
        const value = activationForLatent(sparseRow, latentId);
        const result = byDoc[doc.docId];
        result.entries.push([tokenIndex, value]);
        if (value > result.maxActivation) {
          result.maxActivation = value;
          result.maxTokenIndex = tokenIndex;
        }
      }
      return { docs: byDoc };
    })());
  }
  return promptActivationCache.get(key);
}

self.onmessage = async (event) => {
  const { id, type, payload } = event.data;
  try {
    let result;
    if (type === "init") {
      docFile = await openFile(payload.docTokensUrl);
      sparseFile = await openFile(payload.sparseTopkUrl);
      const rows = await preloadDocRows();
      result = { ok: true, docRows: rows.length };
    } else if (type === "readDoc") {
      result = await readDocRow(payload.row);
    } else if (type === "readActivation") {
      result = await readActivationMap(payload);
    } else if (type === "readPromptActivation") {
      result = await readPromptActivationMap(payload);
    } else if (type === "prefetchActivations") {
      let prefetched = 0;
      for (const task of payload.tasks ?? []) {
        if (task.turnId) await readPromptActivationMap(task);
        else await readActivationMap(task);
        prefetched += 1;
      }
      result = { prefetched };
    } else {
      throw new Error(`unknown worker request ${type}`);
    }
    self.postMessage({ id, result });
  } catch (error) {
    self.postMessage({ id, error: String(error?.message ?? error) });
  }
};
