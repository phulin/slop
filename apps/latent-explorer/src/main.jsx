import React, { startTransition, useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { BookOpenText, FileText, Flame, MessageSquareText, Search, SlidersHorizontal } from "lucide-react";
import "./styles.css";

function numeric(value, fallback = 0) {
  if (typeof value === "bigint") return Number(value);
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function modelName(value) {
  const text = String(value ?? "");
  const names = {
    human: "Human",
    "gpt-5.5": "GPT-5.5",
    "gemini-3.5-flash": "Gemini 3.5 Flash",
    "accounts/fireworks/models/glm-5p2": "GLM-5.2",
  };
  return names[text] ?? text.replace(/^accounts\/fireworks\/models\//, "").replaceAll("-", " ");
}

function routeModelName(value) {
  const text = String(value ?? "");
  const names = {
    human: "human",
    "gpt-5.5": "gpt-5.5",
    "gemini-3.5-flash": "gemini-3.5-flash",
    "accounts/fireworks/models/glm-5p2": "glm-5.2",
  };
  return names[text] ?? encodeURIComponent(text);
}

function modelFromRoute(value) {
  const text = decodeURIComponent(String(value ?? ""));
  const names = {
    human: "human",
    "gpt-5.5": "gpt-5.5",
    "gemini-3.5-flash": "gemini-3.5-flash",
    "glm-5.2": "accounts/fireworks/models/glm-5p2",
  };
  return names[text] ?? text;
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
    "accounts/fireworks/models/glm-5p2": 3,
  }[model] ?? 99;
}

function tokenRowsForDoc(doc, activationMap) {
  const tokens = doc?.tokens ?? [];
  const tokenIndices = doc?.token_indices ?? [];
  return tokens.map((token, index) => {
    const tokenIndex = numeric(tokenIndices[index], index);
    const activation = numeric(activationMap?.[tokenIndex], 0);
    return {
      id: `${tokenIndex}-${index}`,
      text: String(token),
      tokenIndex,
      activation,
    };
  });
}

function routeSelectionFromPath(pathname, loadedIndex) {
  const parts = pathname.split("/").filter(Boolean);
  if (parts.length !== 3 || !/^L\d+$/.test(parts[0])) return null;
  const latentId = parts[0].slice(1);
  const turnId = decodeURIComponent(parts[1]);
  const model = modelFromRoute(parts[2]);
  const promptDocs = loadedIndex.promptGroups?.[turnId] ?? [];
  const documentId = promptDocs.find((docId) => loadedIndex.documents?.[docId]?.model === model);
  if (!documentId || !loadedIndex.latentDocs?.[latentId]) return null;
  const example = loadedIndex.latentDocs[latentId].find((row) => {
    return loadedIndex.documents?.[row.doc_id]?.turn_id === turnId;
  });
  if (!example) return null;
  return {
    latentId,
    exampleDocId: example.doc_id,
    documentId,
  };
}

function routePathForSelection({ latentId, turnId, model }) {
  if (!latentId || !turnId || !model) return "/";
  return `/L${encodeURIComponent(String(latentId))}/${encodeURIComponent(String(turnId))}/${routeModelName(model)}`;
}

function App() {
  const [manifest, setManifest] = useState(null);
  const [indexData, setIndexData] = useState(null);
  const [workerReady, setWorkerReady] = useState(false);
  const [selectedLatent, setSelectedLatent] = useState(null);
  const [selectedExampleDocId, setSelectedExampleDocId] = useState(null);
  const [selectedDocumentId, setSelectedDocumentId] = useState(null);
  const [query, setQuery] = useState("");
  const [source, setSource] = useState("all");
  const [loadedDocs, setLoadedDocs] = useState({});
  const [docActivations, setDocActivations] = useState({});
  const [generationRows, setGenerationRows] = useState([]);
  const [hoveredToken, setHoveredToken] = useState(null);
  const [loadingGeneration, setLoadingGeneration] = useState(false);
  const [loadingActivations, setLoadingActivations] = useState(false);
  const [loadingDocs, setLoadingDocs] = useState(true);
  const [loadError, setLoadError] = useState("");
  const workerRef = useRef(null);
  const requestIdRef = useRef(0);
  const pendingRequestsRef = useRef(new Map());
  const prefetchRequestedRef = useRef(new Set());

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

  useEffect(() => {
    const worker = new Worker(new URL("./parquetWorker.js", import.meta.url), { type: "module" });
    workerRef.current = worker;
    worker.onmessage = (event) => {
      const { id, result, error } = event.data;
      const pending = pendingRequestsRef.current.get(id);
      if (!pending) return;
      pendingRequestsRef.current.delete(id);
      if (error) pending.reject(new Error(error));
      else pending.resolve(result);
    };
    async function load() {
      const loadedManifest = await fetch("/data/manifest.json").then((response) => response.json());
      const loadedIndex = await fetch(loadedManifest.browserIndexJson).then((response) => response.json());
      await workerCall("init", {
        docTokensUrl: loadedManifest.docTokensParquet ?? loadedIndex.docTokensParquet,
        sparseTopkUrl: loadedIndex.sparseTopkParquet ?? loadedManifest.sparseTopkParquet,
      });
      startTransition(() => {
        setManifest(loadedManifest);
        setIndexData(loadedIndex);
        setWorkerReady(true);
        setLoadingDocs(false);
      });
      const routeSelection = routeSelectionFromPath(window.location.pathname, loadedIndex);
      const firstLatent = routeSelection?.latentId ?? String(loadedIndex.latents?.[0]?.latent_id ?? "");
      startTransition(() => {
        setSelectedLatent(firstLatent);
        setSelectedExampleDocId(routeSelection?.exampleDocId ?? loadedIndex.latentDocs?.[firstLatent]?.[0]?.doc_id ?? null);
        setSelectedDocumentId(routeSelection?.documentId ?? null);
      });
    }
    load().catch((error) => {
      console.error(error);
      setLoadError(String(error));
      setLoadingDocs(false);
    });
    return () => {
      worker.terminate();
      workerRef.current = null;
      pendingRequestsRef.current.forEach(({ reject }) => reject(new Error("Parquet worker terminated")));
      pendingRequestsRef.current.clear();
    };
  }, []);

  const latents = useMemo(() => {
    return (indexData?.latents ?? [])
      .map((latent) => ({
        ...latent,
        latent_id: String(latent.latent_id),
        rank: numeric(latent.rank, 999999),
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
      .filter((example) => source === "all" || example.source === source)
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
    return ["all", ...Array.from(new Set(allExamples.map((example) => example.source))).sort()];
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

  const activeLatent = latents.find((latent) => latent.latent_id === String(selectedLatent));
  const activeExample = latentExamples.find((example) => example.doc_id === selectedExampleDocId) ?? latentExamples[0];
  const selectedLatentId = numeric(selectedLatent);
  const selectedMeta = activeExample ? indexData?.documents?.[activeExample.doc_id] : null;
  const selectedDocumentMeta = selectedDocumentId ? indexData?.documents?.[selectedDocumentId] : null;
  const siblingDocIds = useMemo(
    () => (selectedMeta?.turn_id ? (indexData?.promptGroups?.[selectedMeta.turn_id] ?? []) : []),
    [indexData, selectedMeta?.turn_id],
  );

  useEffect(() => {
    setHoveredToken(null);
  }, [selectedLatent, selectedExampleDocId, selectedDocumentId]);

  useEffect(() => {
    setDocActivations({});
    prefetchRequestedRef.current.clear();
  }, [selectedLatent]);

  useEffect(() => {
    if (!indexData || !selectedLatent || !selectedDocumentId || !selectedDocumentMeta) return;
    const nextPath = routePathForSelection({
      latentId: selectedLatent,
      turnId: selectedDocumentMeta.turn_id,
      model: selectedDocumentMeta.model,
    });
    const currentPath = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    if (currentPath !== nextPath) {
      window.history.replaceState(null, "", nextPath);
    }
  }, [indexData, selectedDocumentId, selectedDocumentMeta, selectedLatent]);

  useEffect(() => {
    let cancelled = false;
    async function loadGenerationGroup() {
      if (!workerReady || !indexData || !activeExample || !siblingDocIds.length) return;
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
            model: doc?.model ?? meta.model,
            maxActivation: numeric(maxSummary.max_activation),
            maxTokenIndex: numeric(maxSummary.max_token_index, -1),
          });
        }
        if (cancelled) return;
        rows.sort((a, b) => modelSortKey(a.model) - modelSortKey(b.model));
        const best = rows.reduce(
          (current, row) => (row.maxActivation > current.maxActivation ? row : current),
          rows[0] ?? null,
        );
        startTransition(() => {
          setLoadedDocs((previous) => ({ ...previous, ...docs }));
          setGenerationRows(rows);
          setSelectedDocumentId((current) => (
            current && rows.some((row) => row.docId === current) ? current : best?.docId ?? activeExample.doc_id
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
  }, [activeExample, indexData, selectedLatentId, siblingDocIds, workerReady]); // eslint-disable-line react-hooks/exhaustive-deps

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
  const rawTokenRows = useMemo(
    () => tokenRowsForDoc(activeDocument, activeActivationMap),
    [activeDocument, activeActivationMap],
  );
  const tokenRows = useDeferredValue(rawTokenRows);
  const maxTokenActivation = Math.max(...tokenRows.map((token) => token.activation), 0);
  const peakToken = tokenRows.reduce(
    (current, token) => (token.activation > current.activation ? token : current),
    { activation: 0, tokenIndex: "-", text: "" },
  );
  const inspectedToken = hoveredToken ?? peakToken;

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">
          <Flame size={22} />
          <div>
            <h1>Latent Explorer</h1>
            <p>{manifest?.name ?? "Loading artifacts"}</p>
          </div>
        </div>

        <label className="search">
          <Search size={16} />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search latents"
          />
        </label>

        <div className="latentList">
          {filteredLatents.slice(0, 120).map((latent) => (
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
              <span className="latentId">L{latent.latent_id}</span>
              <span className="score">
                {(latent.causal_mean_abs_target_logit_change || latent.ai_human_log_odds).toFixed(2)}
              </span>
            </button>
          ))}
        </div>
      </aside>

      <section className="workspace">
        <header className="toolbar">
          <div>
            <h2>{activeLatent ? `Latent ${activeLatent.latent_id}` : "Latent"}</h2>
            <p>
              Rank {activeLatent?.rank ?? "-"} · impact{" "}
              {activeLatent?.causal_mean_target_logit_drop?.toFixed(3) ?? "-"} · log-odds{" "}
              {activeLatent?.ai_human_log_odds?.toFixed(3) ?? "-"} ·
              mass share {activeLatent ? `${(activeLatent.ai_mass_share * 100).toFixed(1)}%` : "-"}
            </p>
          </div>
          <label className="sourceSelect">
            <SlidersHorizontal size={16} />
            <select value={source} onChange={(event) => setSource(event.target.value)}>
              {sources.map((item) => (
                <option key={item} value={item}>
                  {item === "all" ? "all" : modelName(item)}
                </option>
              ))}
            </select>
          </label>
        </header>

        <div className="grid">
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
                    <span className="tabScore">{row.maxActivation.toFixed(3)}</span>
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
                <span>activation {numeric(inspectedToken.activation).toFixed(6)}</span>
                <span>index {inspectedToken.tokenIndex ?? "-"}</span>
              </div>
              <div className="tokenView fullDocTokenView">
                {tokenRows.map((token) => (
                  <span
                    key={token.id}
                    className={token.activation === maxTokenActivation && maxTokenActivation > 0 ? "token peak" : "token"}
                    style={{ backgroundColor: activationColor(token.activation, maxTokenActivation) }}
                    title={`activation ${token.activation.toFixed(4)}`}
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
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
