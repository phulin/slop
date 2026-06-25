import React, { startTransition, useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  BookOpenText,
  FileText,
  Flame,
  LoaderCircle,
  MessageSquareText,
  Search,
  SlidersHorizontal,
} from "lucide-react";
import "./styles.css";

function numeric(value, fallback = 0) {
  if (typeof value === "bigint") return Number(value);
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

const LATENT_LABELS_STORAGE_KEY = "slop.latentExplorer.latentLabels.v1";
const SELECTED_RUN_STORAGE_KEY = "slop.latentExplorer.selectedRun.v1";
const LATENT_LABELS_API = "/api/latent-labels";

function normalizeManifest(payload) {
  const fallbackRun = {
    id: payload.id ?? "default",
    name: payload.name ?? "SAE run",
    browserIndexJson: payload.browserIndexJson,
    docTokensParquet: payload.docTokensParquet,
    sparseTopkParquet: payload.sparseTopkParquet,
  };
  const rawRuns = Array.isArray(payload.runs) && payload.runs.length ? payload.runs : [fallbackRun];
  const runs = rawRuns.map((run, index) => ({
    ...fallbackRun,
    ...run,
    id: String(run.id ?? `run-${index + 1}`),
    name: String(run.name ?? run.id ?? `Run ${index + 1}`),
  }));
  const defaultRunId = String(payload.defaultRunId ?? payload.activeRunId ?? runs[0]?.id ?? "");
  return { ...payload, runs, defaultRunId };
}

function latentLabelKey(runId, latentId) {
  return `${runId || "default"}:${latentId}`;
}

function loadCachedLatentLabels() {
  try {
    return JSON.parse(window.localStorage.getItem(LATENT_LABELS_STORAGE_KEY) ?? "{}");
  } catch (_error) {
    return {};
  }
}

function cacheLatentLabels(labels) {
  try {
    window.localStorage.setItem(LATENT_LABELS_STORAGE_KEY, JSON.stringify(labels));
  } catch (_error) {
    // Ignore storage errors; labeling should not break exploration.
  }
}

async function fetchLatentLabels() {
  const response = await fetch(LATENT_LABELS_API, {
    headers: { accept: "application/json" },
    cache: "no-store",
  });
  if (!response.ok) throw new Error(`Label fetch failed: ${response.status}`);
  const payload = await response.json();
  return payload.labels ?? {};
}

async function persistLatentLabel(latentId, label) {
  const response = await fetch(LATENT_LABELS_API, {
    method: "PUT",
    headers: {
      accept: "application/json",
      "content-type": "application/json",
    },
    body: JSON.stringify({ latentId: String(latentId), label }),
  });
  if (!response.ok) throw new Error(`Label save failed: ${response.status}`);
  return response.json();
}

function canonicalModelId(value) {
  const text = String(value ?? "");
  const names = {
    "accounts/fireworks/models/glm-5p2": "glm-5.2",
    "zai-org/GLM-5.2-FP8": "glm-5.2",
    "neuralwatt/glm-5.2-short": "glm-5.2",
    "qwen3.6-35b-fast": "qwen3.6-35b",
  };
  return names[text] ?? text;
}

function modelName(value) {
  const text = canonicalModelId(value);
  const names = {
    human: "Human",
    "gpt-5.5": "GPT-5.5",
    "gemini-3.5-flash": "Gemini 3.5 Flash",
    "glm-5.2": "GLM-5.2",
    "qwen3.6-35b": "Qwen3.6 35B",
  };
  return (
    names[text] ??
    text
      .replace(/^accounts\/fireworks\/models\//, "")
      .replace(/^zai-org\//, "")
      .replace(/^neuralwatt\//, "")
      .replaceAll("-", " ")
  );
}

function routeModelName(value) {
  const text = canonicalModelId(value);
  const names = {
    human: "human",
    "gpt-5.5": "gpt-5.5",
    "gemini-3.5-flash": "gemini-3.5-flash",
    "glm-5.2": "glm-5.2",
    "qwen3.6-35b": "qwen3.6-35b",
  };
  return names[text] ?? encodeURIComponent(text);
}

function modelFromRoute(value) {
  const text = decodeURIComponent(String(value ?? ""));
  const names = {
    human: "human",
    "gpt-5.5": "gpt-5.5",
    "gemini-3.5-flash": "gemini-3.5-flash",
    "glm-5.2": "glm-5.2",
    "qwen3.6-35b": "qwen3.6-35b",
  };
  return canonicalModelId(names[text] ?? text);
}

function shortDocId(value) {
  const text = String(value ?? "");
  return text.length > 10 ? `${text.slice(0, 10)}…` : text || "-";
}

function activationColor(value, maxValue) {
  if (value <= 0 || maxValue <= 0) return "transparent";
  const intensity = Math.min(1, value / maxValue);
  const alpha = 0.12 + intensity * 0.76;
  return `rgba(229, 72, 77, ${alpha})`;
}

function modelSortKey(model) {
  return {
    human: 0,
    "gpt-5.5": 1,
    "gemini-3.5-flash": 2,
    "glm-5.2": 3,
    "qwen3.6-35b": 4,
  }[canonicalModelId(model)] ?? 99;
}

function activationAt(activationMap, tokenIndex) {
  return numeric(activationMap?.[tokenIndex], 0);
}

function tokenRowsForDoc(doc, activationMap) {
  const tokens = doc?.tokens ?? [];
  const tokenIndices = doc?.token_indices ?? [];
  return tokens.map((token, index) => {
    const tokenIndex = numeric(tokenIndices[index], index);
    const activation = activationAt(activationMap, tokenIndex);
    return {
      id: `${tokenIndex}-${index}`,
      text: String(token),
      tokenIndex,
      tokenIndices: [tokenIndex],
      activation,
    };
  });
}

function routeSelectionFromPath(pathname, loadedIndex) {
  const parts = pathname.split("/").filter(Boolean);
  if (parts.length !== 3) return null;
  const turnId = decodeURIComponent(parts[1]);
  const model = modelFromRoute(parts[2]);
  const promptDocs = loadedIndex.promptGroups?.[turnId] ?? [];
  const documentId = promptDocs.find((docId) => canonicalModelId(loadedIndex.documents?.[docId]?.model) === model);
  if (!documentId) return null;
  if (parts[0] === "D") {
    return {
      viewMode: "document",
      documentId,
    };
  }
  if (!/^L\d+$/.test(parts[0])) return null;
  const latentId = parts[0].slice(1);
  if (!loadedIndex.latentDocs?.[latentId]) return null;
  const example = loadedIndex.latentDocs[latentId].find((row) => {
    return loadedIndex.documents?.[row.doc_id]?.turn_id === turnId;
  });
  if (!example) return null;
  return {
    viewMode: "latent",
    latentId,
    exampleDocId: example.doc_id,
    documentId,
  };
}

function routePathForSelection({ latentId, turnId, model }) {
  if (!latentId || !turnId || !model) return "/";
  return `/L${encodeURIComponent(String(latentId))}/${encodeURIComponent(String(turnId))}/${routeModelName(model)}`;
}

function routePathForDocument({ turnId, model }) {
  if (!turnId || !model) return "/";
  return `/D/${encodeURIComponent(String(turnId))}/${routeModelName(model)}`;
}

function causalImpactForDoc(indexData, latentId, docId) {
  const rows = indexData?.latentDocs?.[String(latentId)] ?? [];
  const match = rows.find((row) => row.doc_id === docId);
  if (match?.causal_logit_drop !== undefined) return numeric(match.causal_logit_drop, null);
  const generationImpact = indexData?.generationImpacts?.[String(latentId)]?.[docId];
  if (generationImpact?.causal_logit_drop !== undefined) {
    return numeric(generationImpact.causal_logit_drop, null);
  }
  return null;
}

function App() {
  const [manifest, setManifest] = useState(null);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [indexData, setIndexData] = useState(null);
  const [workerOnline, setWorkerOnline] = useState(false);
  const [workerReady, setWorkerReady] = useState(false);
  const [selectedLatent, setSelectedLatent] = useState(null);
  const [selectedExampleDocId, setSelectedExampleDocId] = useState(null);
  const [selectedDocumentId, setSelectedDocumentId] = useState(null);
  const [viewMode, setViewMode] = useState("latent");
  const [latentLabels, setLatentLabels] = useState(loadCachedLatentLabels);
  const [query, setQuery] = useState("");
  const [source, setSource] = useState("all");
  const [tokenDisplayMode, setTokenDisplayMode] = useState("activation");
  const [loadedDocs, setLoadedDocs] = useState({});
  const [docActivations, setDocActivations] = useState({});
  const [documentLatents, setDocumentLatents] = useState({});
  const [generationRows, setGenerationRows] = useState([]);
  const [hoveredToken, setHoveredToken] = useState(null);
  const [loadingGeneration, setLoadingGeneration] = useState(false);
  const [loadingActivations, setLoadingActivations] = useState(false);
  const [loadingDocumentLatents, setLoadingDocumentLatents] = useState(false);
  const [loadingDocs, setLoadingDocs] = useState(true);
  const [loadStage, setLoadStage] = useState("Starting");
  const [loadError, setLoadError] = useState("");
  const workerRef = useRef(null);
  const requestIdRef = useRef(0);
  const dataLoadIdRef = useRef(0);
  const pendingRequestsRef = useRef(new Map());
  const prefetchRequestedRef = useRef(new Set());
  const pendingLabelSavesRef = useRef(new Map());
  const labelSaveTimeoutRef = useRef(null);
  const runs = manifest?.runs ?? [];
  const activeRun = useMemo(() => {
    return runs.find((run) => run.id === selectedRunId) ?? runs[0] ?? null;
  }, [runs, selectedRunId]);

  function workerCall(type, payload) {
    const worker = workerRef.current;
    if (!worker) return Promise.reject(new Error("Parquet worker is not ready"));
    const id = requestIdRef.current + 1;
    requestIdRef.current = id;
    return new Promise((resolve, reject) => {
      pendingRequestsRef.current.set(id, { resolve, reject });
      worker.postMessage({ id, type, payload });
    });
  }

  async function flushPendingLabelSaves() {
    const entries = Array.from(pendingLabelSavesRef.current.entries());
    pendingLabelSavesRef.current.clear();
    await Promise.allSettled(
      entries.map(([latentId, label]) => persistLatentLabel(latentId, label)),
    );
  }

  function scheduleLabelSave(latentId, label) {
    pendingLabelSavesRef.current.set(String(latentId), label.trim());
    if (labelSaveTimeoutRef.current) window.clearTimeout(labelSaveTimeoutRef.current);
    labelSaveTimeoutRef.current = window.setTimeout(() => {
      labelSaveTimeoutRef.current = null;
      flushPendingLabelSaves().catch((error) => {
        console.warn("label save failed", error);
      });
    }, 350);
  }

  useEffect(() => {
    let cancelled = false;
    async function loadLabels() {
      const cachedLabels = loadCachedLatentLabels();
      try {
        const remoteLabels = await fetchLatentLabels();
        if (cancelled) return;
        const mergedLabels = { ...cachedLabels, ...remoteLabels };
        setLatentLabels(mergedLabels);
        cacheLatentLabels(mergedLabels);
        const missingRemoteEntries = Object.entries(cachedLabels).filter(
          ([latentId, label]) => label && remoteLabels[latentId] === undefined,
        );
        await Promise.allSettled(
          missingRemoteEntries.map(([latentId, label]) => persistLatentLabel(latentId, label)),
        );
      } catch (error) {
        console.warn("label load failed; using cached labels", error);
      }
    }
    loadLabels();
    return () => {
      cancelled = true;
      if (labelSaveTimeoutRef.current) {
        window.clearTimeout(labelSaveTimeoutRef.current);
        labelSaveTimeoutRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    const worker = new Worker(new URL("./parquetWorker.js", import.meta.url), { type: "module" });
    workerRef.current = worker;
    setWorkerOnline(true);
    worker.onmessage = (event) => {
      const { id, result, error } = event.data;
      const pending = pendingRequestsRef.current.get(id);
      if (!pending) return;
      pendingRequestsRef.current.delete(id);
      if (error) pending.reject(new Error(error));
      else pending.resolve(result);
    };
    return () => {
      setWorkerOnline(false);
      worker.terminate();
      workerRef.current = null;
      pendingRequestsRef.current.forEach(({ reject }) => reject(new Error("Parquet worker terminated")));
      pendingRequestsRef.current.clear();
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function loadManifest() {
      try {
        setLoadStage("Loading manifest");
        const loadedManifest = normalizeManifest(
          await fetch("/data/manifest.json", { cache: "no-store" }).then((response) => response.json()),
        );
        if (cancelled) return;
        const cachedRunId = window.localStorage.getItem(SELECTED_RUN_STORAGE_KEY);
        const initialRunId = loadedManifest.runs.some((run) => run.id === cachedRunId)
          ? cachedRunId
          : loadedManifest.defaultRunId;
        startTransition(() => {
          setManifest(loadedManifest);
          setSelectedRunId(initialRunId);
        });
      } catch (error) {
        console.error(error);
        if (!cancelled) {
          setLoadError(String(error));
          setLoadingDocs(false);
        }
      }
    }
    loadManifest();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!workerOnline || !activeRun) return;
    let cancelled = false;
    const loadId = dataLoadIdRef.current + 1;
    dataLoadIdRef.current = loadId;
    startTransition(() => {
      setIndexData(null);
      setWorkerReady(false);
      setSelectedLatent(null);
      setSelectedExampleDocId(null);
      setSelectedDocumentId(null);
      setViewMode("latent");
      setLoadedDocs({});
      setDocActivations({});
      setDocumentLatents({});
      setGenerationRows([]);
      setHoveredToken(null);
      setSource("all");
      setLoadingDocs(true);
      setLoadingGeneration(false);
      setLoadingActivations(false);
      setLoadingDocumentLatents(false);
      setLoadError("");
    });
    prefetchRequestedRef.current.clear();
    try {
      window.localStorage.setItem(SELECTED_RUN_STORAGE_KEY, activeRun.id);
    } catch (_error) {
      // Ignore storage errors; run selection should still work.
    }
    async function loadRun() {
      setLoadStage("Loading index");
      const loadedIndex = await fetch(activeRun.browserIndexJson, { cache: "no-store" }).then((response) => response.json());
      if (cancelled || dataLoadIdRef.current !== loadId) return;
      setLoadStage("Opening parquet roots");
      await workerCall("init", {
        docTokensUrl: activeRun.docTokensParquet ?? loadedIndex.docTokensParquet,
        sparseTopkUrl: loadedIndex.sparseTopkParquet ?? activeRun.sparseTopkParquet,
      });
      if (cancelled || dataLoadIdRef.current !== loadId) return;
      setLoadStage("Ready");
      const routeSelection = routeSelectionFromPath(window.location.pathname, loadedIndex);
      const firstLatent = routeSelection?.latentId ?? String(loadedIndex.latents?.[0]?.latent_id ?? "");
      startTransition(() => {
        setIndexData(loadedIndex);
        setWorkerReady(true);
        setLoadingDocs(false);
        setViewMode(routeSelection?.viewMode ?? "latent");
        setSelectedLatent(firstLatent);
        setSelectedExampleDocId(routeSelection?.exampleDocId ?? loadedIndex.latentDocs?.[firstLatent]?.[0]?.doc_id ?? null);
        setSelectedDocumentId(routeSelection?.documentId ?? null);
      });
    }
    loadRun().catch((error) => {
      console.error(error);
      if (!cancelled && dataLoadIdRef.current === loadId) {
        setLoadError(String(error));
        setLoadingDocs(false);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [activeRun, workerOnline]); // eslint-disable-line react-hooks/exhaustive-deps

  const latents = useMemo(() => {
    return (indexData?.latents ?? [])
      .map((latent) => ({
        ...latent,
        latent_id: String(latent.latent_id),
        rank: numeric(latent.rank, 999999),
        causal_max_abs_target_logit_change: numeric(latent.causal_max_abs_target_logit_change),
        causal_max_target_logit_drop: numeric(latent.causal_max_target_logit_drop),
        causal_mean_abs_target_logit_change: numeric(latent.causal_mean_abs_target_logit_change),
        causal_mean_target_logit_drop: numeric(latent.causal_mean_target_logit_drop),
        ai_human_log_odds: numeric(latent.ai_human_log_odds),
        ai_mass_share: numeric(latent.ai_mass_share),
        total_mass: numeric(latent.total_mass),
      }))
      .sort((a, b) => a.rank - b.rank);
  }, [indexData]);

  const latentExamples = useMemo(() => {
    const examples = indexData?.latentDocs?.[String(selectedLatent)] ?? [];
    return examples
      .filter((example) => source === "all" || canonicalModelId(example.source) === source)
      .sort((a, b) => numeric(a.example_rank) - numeric(b.example_rank));
  }, [indexData, selectedLatent, source]);

  useEffect(() => {
    if (!latentExamples.length) {
      setSelectedExampleDocId(null);
      return;
    }
    if (!latentExamples.some((example) => example.doc_id === selectedExampleDocId)) {
      setSelectedExampleDocId(latentExamples[0].doc_id);
    }
  }, [latentExamples, selectedExampleDocId]);

  const sources = useMemo(() => {
    const allExamples = Object.values(indexData?.latentDocs ?? {}).flat();
    return ["all", ...Array.from(new Set(allExamples.map((example) => canonicalModelId(example.source)))).sort()];
  }, [indexData]);

  const filteredLatents = useMemo(() => {
    const q = query.trim().toLowerCase();
    return latents.filter((latent) => {
      if (!q) return true;
      return (
        latent.latent_id.includes(q) ||
        String(latent.rank).includes(q) ||
        String(latent.source_token_totals ?? "").toLowerCase().includes(q)
      );
    });
  }, [latents, query]);

  const latentById = useMemo(
    () => Object.fromEntries(latents.map((latent) => [String(latent.latent_id), latent])),
    [latents],
  );

  const rootLoading = loadingDocs && !indexData;
  const rootLoadFailed = Boolean(loadError) && !indexData;
  const activeLatent = latents.find((latent) => latent.latent_id === String(selectedLatent));
  const activeExample = latentExamples.find((example) => example.doc_id === selectedExampleDocId) ?? latentExamples[0];
  const selectedLatentId = numeric(selectedLatent);
  const selectedMeta = activeExample ? indexData?.documents?.[activeExample.doc_id] : null;
  const selectedDocumentMeta = selectedDocumentId ? indexData?.documents?.[selectedDocumentId] : null;
  const promptTurnId = selectedDocumentMeta?.turn_id ?? selectedMeta?.turn_id ?? null;
  const siblingDocIds = useMemo(
    () => (promptTurnId ? (indexData?.promptGroups?.[promptTurnId] ?? []) : []),
    [indexData, promptTurnId],
  );

  useEffect(() => {
    setHoveredToken(null);
  }, [selectedLatent, selectedExampleDocId, selectedDocumentId]);

  useEffect(() => {
    setDocActivations({});
    prefetchRequestedRef.current.clear();
  }, [selectedLatent]);

  useEffect(() => {
    if (!indexData || !selectedDocumentId || !selectedDocumentMeta) return;
    const nextPath = viewMode === "document"
      ? routePathForDocument({
        turnId: selectedDocumentMeta.turn_id,
        model: selectedDocumentMeta.model,
      })
      : routePathForSelection({
        latentId: selectedLatent,
        turnId: selectedDocumentMeta.turn_id,
        model: selectedDocumentMeta.model,
      });
    const currentPath = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    if (currentPath !== nextPath) {
      window.history.replaceState(null, "", nextPath);
    }
  }, [indexData, selectedDocumentId, selectedDocumentMeta, selectedLatent, viewMode]);

  useEffect(() => {
    let cancelled = false;
    async function loadGenerationGroup() {
      if (!workerReady || !indexData || !siblingDocIds.length) return;
      setLoadingGeneration(true);
      setLoadError("");
      try {
        const docs = {};
        const rows = [];
        for (const docId of siblingDocIds) {
          const meta = indexData.documents[docId];
          if (!meta) continue;
          const doc = loadedDocs[docId] ?? await workerCall("readDoc", { row: numeric(meta.row) });
          docs[docId] = doc;
          const maxSummary = indexData.generationMaxes?.[String(selectedLatentId)]?.[docId] ?? {};
          rows.push({
            docId,
            doc,
            model: canonicalModelId(doc?.model ?? meta.model),
            maxActivation: numeric(maxSummary.max_activation),
            maxTokenIndex: numeric(maxSummary.max_token_index, -1),
            scoreImpact: causalImpactForDoc(indexData, selectedLatentId, docId),
          });
        }
        if (cancelled) return;
        rows.sort((a, b) => modelSortKey(a.model) - modelSortKey(b.model));
        const best = rows.reduce(
          (current, row) => {
            const currentImpact = current.scoreImpact === null ? -Infinity : Math.abs(current.scoreImpact);
            const rowImpact = row.scoreImpact === null ? -Infinity : Math.abs(row.scoreImpact);
            return rowImpact > currentImpact ? row : current;
          },
          rows[0] ?? null,
        );
        startTransition(() => {
          setLoadedDocs((previous) => ({ ...previous, ...docs }));
          setGenerationRows(rows);
          setSelectedDocumentId((current) => (
            current && rows.some((row) => row.docId === current) ? current : best?.docId ?? selectedDocumentId
          ));
        });
      } catch (error) {
        console.error(error);
        if (!cancelled) setLoadError(String(error));
      } finally {
        if (!cancelled) setLoadingGeneration(false);
      }
    }
    loadGenerationGroup();
    return () => {
      cancelled = true;
    };
  }, [indexData, selectedDocumentId, selectedLatentId, siblingDocIds, workerReady]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    let cancelled = false;
    async function loadSelectedActivations() {
      if (!workerReady || !indexData || !selectedDocumentId) return;
      if (docActivations[selectedDocumentId]) return;
      const meta = indexData.documents?.[selectedDocumentId];
      if (!meta) return;
      const turnId = meta.turn_id;
      const turnRange = indexData.turnRanges?.[turnId];
      const promptDocs = (indexData.promptGroups?.[turnId] ?? [])
        .map((docId) => {
          const docMeta = indexData.documents?.[docId];
          if (!docMeta) return null;
          return {
            docId,
            start: numeric(docMeta.sparse_row_start),
            end: numeric(docMeta.sparse_row_end),
          };
        })
        .filter(Boolean);
      setLoadingActivations(true);
      try {
        const result = turnRange
          ? await workerCall("readPromptActivation", {
            turnId,
            latentId: selectedLatentId,
            start: numeric(turnRange.row_start),
            end: numeric(turnRange.row_end),
            docs: promptDocs,
          })
          : await workerCall("readActivation", {
            selectedDocumentId,
            docId: selectedDocumentId,
            latentId: selectedLatentId,
            start: numeric(meta.sparse_row_start),
            end: numeric(meta.sparse_row_end),
          });
        if (cancelled) return;
        const activationMaps = result.docs
          ? Object.fromEntries(
            Object.entries(result.docs).map(([docId, docResult]) => [
              docId,
              Object.fromEntries(docResult.entries ?? []),
            ]),
          )
          : { [selectedDocumentId]: Object.fromEntries(result.entries ?? []) };
        startTransition(() => {
          setDocActivations((previous) => ({
            ...previous,
            ...activationMaps,
          }));
        });
      } catch (error) {
        console.error(error);
        if (!cancelled) setLoadError(String(error));
      } finally {
        if (!cancelled) setLoadingActivations(false);
      }
    }
    loadSelectedActivations();
    return () => {
      cancelled = true;
    };
  }, [docActivations, indexData, selectedDocumentId, selectedLatentId, workerReady]);

  useEffect(() => {
    let cancelled = false;
    async function loadDocumentLatents() {
      if (!workerReady || !indexData || !selectedDocumentId || viewMode !== "document") return;
      if (documentLatents[selectedDocumentId]) return;
      const meta = indexData.documents?.[selectedDocumentId];
      if (!meta) return;
      setLoadingDocumentLatents(true);
      setLoadError("");
      try {
        const result = await workerCall("readDocumentLatents", {
          docId: selectedDocumentId,
          start: numeric(meta.sparse_row_start),
          end: numeric(meta.sparse_row_end),
          limit: 240,
        });
        if (cancelled) return;
        startTransition(() => {
          setDocumentLatents((previous) => ({
            ...previous,
            [selectedDocumentId]: result.rows ?? [],
          }));
        });
      } catch (error) {
        console.error(error);
        if (!cancelled) setLoadError(String(error));
      } finally {
        if (!cancelled) setLoadingDocumentLatents(false);
      }
    }
    loadDocumentLatents();
    return () => {
      cancelled = true;
    };
  }, [documentLatents, indexData, selectedDocumentId, viewMode, workerReady]);

  useEffect(() => {
    if (!workerReady || !indexData || !activeExample || !latentExamples.length) return;
    if (!selectedDocumentId || !docActivations[selectedDocumentId]) return;
    const activeIndex = latentExamples.findIndex((example) => example.doc_id === activeExample.doc_id);
    if (activeIndex < 0) return;
    const windowExamples = latentExamples.slice(
      Math.max(0, activeIndex - 2),
      Math.min(latentExamples.length, activeIndex + 3),
    );
    const tasks = [];
    for (const example of windowExamples) {
      const meta = indexData.documents?.[example.doc_id];
      const turnId = meta?.turn_id;
      if (!turnId) continue;
      const promptKey = `${selectedLatentId}:${turnId}`;
      if (prefetchRequestedRef.current.has(promptKey)) continue;
      prefetchRequestedRef.current.add(promptKey);
      const turnRange = indexData.turnRanges?.[turnId];
      if (turnRange) {
        tasks.push({
          turnId,
          latentId: selectedLatentId,
          start: numeric(turnRange.row_start),
          end: numeric(turnRange.row_end),
          docs: (indexData.promptGroups?.[turnId] ?? [])
            .map((docId) => {
              const docMeta = indexData.documents?.[docId];
              if (!docMeta) return null;
              return {
                docId,
                start: numeric(docMeta.sparse_row_start),
                end: numeric(docMeta.sparse_row_end),
              };
            })
            .filter(Boolean),
        });
      } else {
        for (const docId of indexData.promptGroups?.[turnId] ?? []) {
          const docMeta = indexData.documents?.[docId];
          if (!docMeta) continue;
          tasks.push({
            docId,
            latentId: selectedLatentId,
            start: numeric(docMeta.sparse_row_start),
            end: numeric(docMeta.sparse_row_end),
          });
        }
      }
    }
    if (!tasks.length) return;
    const schedule = window.requestIdleCallback ?? ((callback) => window.setTimeout(callback, 50));
    const cancelSchedule = window.cancelIdleCallback ?? window.clearTimeout;
    const handle = schedule(() => {
      workerCall("prefetchActivations", { tasks }).catch((error) => {
        console.warn("activation prefetch failed", error);
      });
    });
    return () => cancelSchedule(handle);
  }, [activeExample, docActivations, indexData, latentExamples, selectedDocumentId, selectedLatentId, workerReady]);

  const activeDocument = selectedDocumentId ? loadedDocs[selectedDocumentId] : null;
  const activeActivationMap = selectedDocumentId ? docActivations[selectedDocumentId] : null;
  const tokenTextByIndex = useMemo(() => {
    const tokens = activeDocument?.tokens ?? [];
    const tokenIndices = activeDocument?.token_indices ?? [];
    return Object.fromEntries(tokens.map((token, index) => [
      String(numeric(tokenIndices[index], index)),
      String(token),
    ]));
  }, [activeDocument]);
  const documentLatentRows = useMemo(() => {
    const rows = documentLatents[selectedDocumentId] ?? [];
    return rows.map((row, index) => {
      const latentId = String(row.latentId);
      const latent = latentById[latentId];
      return {
        ...row,
        latentId,
        displayRank: index + 1,
        latent,
        label: labelForLatent(latentId),
        peakToken: tokenTextByIndex[String(row.peakTokenIndex)] ?? "",
      };
    });
  }, [documentLatents, latentById, selectedDocumentId, tokenTextByIndex, latentLabels, selectedRunId]);
  const rawTokenRows = useMemo(
    () => tokenRowsForDoc(activeDocument, activeActivationMap),
    [activeDocument, activeActivationMap],
  );
  const tokenRows = useDeferredValue(rawTokenRows);
  const displayTokenRows = useMemo(
    () => tokenRows.map((token, index) => {
      const previousActivation = index > 0 ? numeric(tokenRows[index - 1]?.activation) : numeric(token.activation);
      const positiveDelta = Math.max(0, numeric(token.activation) - previousActivation);
      const displayValue = tokenDisplayMode === "positive-delta" ? positiveDelta : numeric(token.activation);
      return {
        ...token,
        positiveDelta,
        displayValue,
      };
    }),
    [tokenDisplayMode, tokenRows],
  );
  const maxTokenDisplayValue = Math.max(...displayTokenRows.map((token) => token.displayValue), 0);
  const peakToken = displayTokenRows.reduce(
    (current, token) => (token.displayValue > current.displayValue ? token : current),
    { activation: 0, positiveDelta: 0, displayValue: 0, tokenIndex: "-", text: "" },
  );
  const inspectedToken = hoveredToken ?? peakToken;
  const displayMetricLabel = tokenDisplayMode === "positive-delta" ? "delta" : "activation";

  function labelForLatent(latentId) {
    const key = latentLabelKey(selectedRunId, latentId);
    return latentLabels[key] ?? latentLabels[String(latentId)] ?? "";
  }

  function updateLatentLabel(latentId, value) {
    const label = value.trimStart();
    const key = latentLabelKey(selectedRunId, latentId);
    setLatentLabels((previous) => {
      const next = { ...previous };
      if (label.trim()) next[key] = label;
      else delete next[key];
      cacheLatentLabels(next);
      return next;
    });
    scheduleLabelSave(key, label);
  }

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">
          <Flame size={22} />
          <div>
            <h1>Latent Explorer</h1>
            <p>{activeRun?.name ?? manifest?.name ?? "Loading artifacts"}</p>
          </div>
        </div>

        <label className="runSelect">
          <span>SAE</span>
          <select
            value={selectedRunId}
            onChange={(event) => setSelectedRunId(event.target.value)}
            disabled={!runs.length}
          >
            {runs.map((run) => (
              <option key={run.id} value={run.id}>
                {run.name}
              </option>
            ))}
          </select>
        </label>

        <label className="search">
          <Search size={16} />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search latents"
            disabled={rootLoading || rootLoadFailed}
          />
        </label>

        <div className="latentList">
          {rootLoading ? (
            Array.from({ length: 12 }, (_, index) => (
              <div className="latentSkeleton" key={index}>
                <span />
                <span />
                <span />
              </div>
            ))
          ) : null}
          {!rootLoading && filteredLatents.slice(0, 120).map((latent) => (
            <button
              key={latent.latent_id}
              className={latent.latent_id === String(selectedLatent) ? "latent active" : "latent"}
              onClick={() => {
                const latentId = String(latent.latent_id);
                if (latentId === String(selectedLatent)) return;
                setSelectedLatent(latentId);
                setSelectedExampleDocId(indexData?.latentDocs?.[latentId]?.[0]?.doc_id ?? null);
                setSelectedDocumentId(null);
              }}
            >
              <span className="rank">#{latent.rank}</span>
              <span className="latentTitle">
                <span className="latentId">L{latent.latent_id}</span>
                {labelForLatent(latent.latent_id) ? (
                  <span className="latentLabel">{labelForLatent(latent.latent_id)}</span>
                ) : null}
              </span>
              <span className="score">
                {(latent.causal_max_abs_target_logit_change || latent.ai_human_log_odds).toFixed(2)}
              </span>
            </button>
          ))}
        </div>
      </aside>

      <section className="workspace">
        <header className="toolbar">
          <div>
            <div className="latentHeading">
              <h2>{selectedLatent ? `Latent ${selectedLatent}` : "Latent"}</h2>
              {selectedLatent ? (
                <input
                  className="latentLabelInput"
                  value={labelForLatent(selectedLatent)}
                  onChange={(event) => updateLatentLabel(selectedLatent, event.target.value)}
                  placeholder="Label"
                />
              ) : null}
              <div className="viewSwitch" role="group" aria-label="Explorer view">
                <button
                  type="button"
                  className={viewMode === "latent" ? "viewSwitchButton active" : "viewSwitchButton"}
                  onClick={() => setViewMode("latent")}
                >
                  Latent
                </button>
                <button
                  type="button"
                  className={viewMode === "document" ? "viewSwitchButton active" : "viewSwitchButton"}
                  onClick={() => setViewMode("document")}
                  disabled={!selectedDocumentId}
                >
                  Document
                </button>
              </div>
              <label className="tokenModeSlider">
                <span>Act</span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="1"
                  value={tokenDisplayMode === "positive-delta" ? "1" : "0"}
                  onChange={(event) => {
                    setTokenDisplayMode(event.target.value === "1" ? "positive-delta" : "activation");
                  }}
                  aria-label="Token highlight mode"
                />
                <span>Δ+</span>
              </label>
            </div>
            <p>
              Rank {activeLatent?.rank ?? "-"} · impact{" "}
              {activeLatent?.causal_max_target_logit_drop?.toFixed(3) ?? "-"} · log-odds{" "}
              {activeLatent?.ai_human_log_odds?.toFixed(3) ?? "-"} ·
              mass share {activeLatent ? `${(activeLatent.ai_mass_share * 100).toFixed(1)}%` : "-"}
            </p>
          </div>
          <label className="sourceSelect">
            <SlidersHorizontal size={16} />
            <select
              value={source}
              onChange={(event) => setSource(event.target.value)}
              disabled={rootLoading || rootLoadFailed}
            >
              {sources.map((item) => (
                <option key={item} value={item}>
                  {item === "all" ? "all" : modelName(item)}
                </option>
              ))}
            </select>
          </label>
        </header>

        {rootLoading || rootLoadFailed ? (
          <div className="rootState">
            <div className={rootLoadFailed ? "rootStatePanel error" : "rootStatePanel"}>
              {rootLoadFailed ? <FileText size={24} /> : <LoaderCircle className="spin" size={24} />}
              <div>
                <h3>{rootLoadFailed ? "Data Load Failed" : "Loading Root Data"}</h3>
                <p>{rootLoadFailed ? loadError : `${activeRun?.name ?? "Selected SAE"} · ${loadStage}`}</p>
              </div>
            </div>
          </div>
        ) : (
        <div className="grid">
          {viewMode === "document" ? (
            <section className="examples">
              <div className="sectionTitle">
                <FileText size={17} />
                <h3>Document Latents</h3>
              </div>
              <div className="documentLatentSummary">
                <span title={selectedDocumentId ?? ""}>doc {shortDocId(selectedDocumentId)}</span>
                <span>{loadingDocumentLatents ? "loading latents" : `${documentLatentRows.length} shown`}</span>
              </div>
              <div className="exampleList">
                {documentLatentRows.map((row) => (
                  <button
                    key={`${selectedDocumentId}-${row.latentId}`}
                    className={String(selectedLatent) === row.latentId ? "example documentLatent active" : "example documentLatent"}
                    onClick={() => {
                      setSelectedLatent(row.latentId);
                      const latentDocs = indexData?.latentDocs?.[row.latentId] ?? [];
                      const samePromptExample = latentDocs.find((example) => (
                        indexData?.documents?.[example.doc_id]?.turn_id === promptTurnId
                      ));
                      if (samePromptExample) setSelectedExampleDocId(samePromptExample.doc_id);
                    }}
                  >
                    <span className="exampleMeta">
                      #{row.displayRank} · L{row.latentId}
                      {row.latent?.rank ? ` · rank ${row.latent.rank}` : ""} · peak {numeric(row.maxActivation).toFixed(3)}
                    </span>
                    <span className="documentLatentTitle">
                      {row.label || (row.latent ? "ranked latent" : "uncurated latent")}
                    </span>
                    <span className="documentLatentStats">
                      mass {numeric(row.mass).toFixed(2)} · tokens {numeric(row.tokenCount)} · index {numeric(row.peakTokenIndex, "-")}
                      {row.peakToken ? ` · ${row.peakToken}` : ""}
                    </span>
                  </button>
                ))}
                {!loadingDocumentLatents && !documentLatentRows.length ? (
                  <div className="emptyState">No sparse latent activations found for this document.</div>
                ) : null}
              </div>
            </section>
          ) : (
            <section className="examples">
              <div className="sectionTitle">
                <FileText size={17} />
                <h3>Top Docs</h3>
              </div>
              <div className="exampleList">
                {latentExamples.map((example) => (
                  <button
                    key={`${example.doc_id}-${example.example_rank}`}
                    className={activeExample?.doc_id === example.doc_id ? "example active" : "example"}
                    onClick={() => {
                      if (activeExample?.doc_id === example.doc_id) return;
                      setSelectedExampleDocId(example.doc_id);
                      setSelectedDocumentId(null);
                    }}
                  >
                    <span className="exampleMeta">
                      #{example.example_rank} · {modelName(example.source)} · impact{" "}
                      {(example.causal_logit_drop !== undefined
                        ? numeric(example.causal_logit_drop)
                        : numeric(example.peak_activation)).toFixed(3)}
                    </span>
                    <span className="exampleText">{example.context}</span>
                  </button>
                ))}
              </div>
            </section>
          )}

          <section className="documentPane">
            <div className="documentBlock promptBlock">
              <div className="sectionTitle">
                <MessageSquareText size={17} />
                <h3>Prompt</h3>
              </div>
              <div className="documentText promptText">
                {activeDocument?.prompt || "Prompt unavailable for this document."}
              </div>
            </div>

            <div className="documentBlock">
              <div className="sectionTitle">
                <BookOpenText size={17} />
                <h3>Generations</h3>
              </div>
              <div className="modelTabs" role="tablist" aria-label="Prompt completions">
                {generationRows.map((row) => (
                  <button
                    key={row.docId}
                    className={row.docId === selectedDocumentId ? "modelTab active" : "modelTab"}
                    onClick={() => {
                      if (row.docId !== selectedDocumentId) setSelectedDocumentId(row.docId);
                    }}
                    title={row.docId}
                  >
                    <span>{modelName(row.model)}</span>
                    <span className="tabScore">
                      {row.scoreImpact === null ? "-" : row.scoreImpact.toFixed(3)}
                    </span>
                  </button>
                ))}
              </div>
              <div className="documentStats">
                <span>{modelName(activeDocument?.model)}</span>
                <span>{activeDocument?.category ?? "-"}</span>
                <span title={selectedDocumentId ?? ""}>doc {shortDocId(selectedDocumentId)}</span>
                <span>{numeric(activeDocument?.word_count, "-")} words</span>
                <span>{numeric(activeDocument?.token_len, "-")} tokens</span>
                {loadingDocs ? <span>preloading docs</span> : null}
                {loadingGeneration ? <span>loading prompt group</span> : null}
                {loadingActivations ? <span>loading token activations</span> : null}
                {loadError ? <span className="errorText">{loadError}</span> : null}
              </div>
              <div className="hoverReadout" aria-live="polite">
                <span className="readoutLabel">{hoveredToken ? "hover" : "peak"}</span>
                <span>{displayMetricLabel} {numeric(inspectedToken.displayValue).toFixed(6)}</span>
                <span>activation {numeric(inspectedToken.activation).toFixed(6)}</span>
                <span>delta {numeric(inspectedToken.positiveDelta).toFixed(6)}</span>
                <span>index {inspectedToken.tokenIndex ?? "-"}</span>
              </div>
              <div className="tokenView fullDocTokenView">
                {displayTokenRows.map((token) => (
                  <span
                    key={token.id}
                    className={token.displayValue === maxTokenDisplayValue && maxTokenDisplayValue > 0 ? "token peak" : "token"}
                    style={{ backgroundColor: activationColor(token.displayValue, maxTokenDisplayValue) }}
                    title={`activation ${token.activation.toFixed(4)} · delta ${token.positiveDelta.toFixed(4)}`}
                    onMouseEnter={() => setHoveredToken(token)}
                    onMouseLeave={() => setHoveredToken(null)}
                  >
                    {token.text}
                  </span>
                ))}
              </div>
            </div>
          </section>
        </div>
        )}
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
