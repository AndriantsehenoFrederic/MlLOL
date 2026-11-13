"use client";

import { useEffect, useMemo, useState, type CSSProperties } from "react";

type Side = "blue" | "red";
type Slot = { side: Side; role: number };
type Champion = { name: string; portrait_url: string | null; matches: number };
type Picks = Record<Side, (string | null)[]>;
type Roster = { champions: Champion[]; model_ready: boolean; roles: string[] };
type Prediction = { blue_probability: number; red_probability: number };

const ROLES = ["TOP", "JUNGLE", "MID", "BOT", "SUPPORT"];
const EMPTY_PICKS = (): Picks => ({ blue: Array(5).fill(null), red: Array(5).fill(null) });
const ALL_SLOTS: Slot[] = [
  ...ROLES.map((_, role) => ({ side: "blue" as const, role })),
  ...ROLES.map((_, role) => ({ side: "red" as const, role })),
];

function normalizeSearch(value: string) {
  return value.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase().replace(/[^a-z0-9]/g, "");
}

function nextEmptySlot(picks: Picks, current: Slot): Slot {
  const start = ALL_SLOTS.findIndex((slot) => slot.side === current.side && slot.role === current.role);
  for (let offset = 1; offset <= ALL_SLOTS.length; offset += 1) {
    const candidate = ALL_SLOTS[(start + offset) % ALL_SLOTS.length];
    if (!picks[candidate.side][candidate.role]) return candidate;
  }
  return current;
}

function Crest() {
  return (
    <svg viewBox="0 0 64 64" fill="none" aria-hidden="true" className="crest-icon">
      <path d="M32 4 55 14v20c0 15-10 23-23 27C19 57 9 49 9 34V14L32 4Z" stroke="currentColor" strokeWidth="1.5" />
      <path d="m19 22 7 8 6-13 6 13 7-8-3 19H22l-3-19Z" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
      <path d="M23 46h18" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle cx="10.8" cy="10.8" r="6.3" stroke="currentColor" strokeWidth="1.6" />
      <path d="m16 16 4.6 4.6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

function TeamPanel({
  side,
  picks,
  activeSlot,
  championByName,
  onActivate,
  onClear,
}: {
  side: Side;
  picks: Picks;
  activeSlot: Slot;
  championByName: Map<string, Champion>;
  onActivate: (slot: Slot) => void;
  onClear: (slot: Slot) => void;
}) {
  return (
    <section className={`team-panel ${side}`} aria-label={`Composition de l'équipe ${side === "blue" ? "bleue" : "rouge"}`}>
      <div className="team-heading">
        <span className="team-marker" />
        <div>
          <p className="overline">COMPOSITION</p>
          <h2>Équipe {side === "blue" ? "bleue" : "rouge"}</h2>
        </div>
        <span className="team-count">{picks[side].filter(Boolean).length}/5</span>
      </div>
      <div className="slots">
        {ROLES.map((role, roleIndex) => {
          const championName = picks[side][roleIndex];
          const champion = championName ? championByName.get(championName) : null;
          const isActive = activeSlot.side === side && activeSlot.role === roleIndex;
          return (
            <div className={`slot-row ${isActive ? "active" : ""} ${champion ? "filled" : ""}`} key={`${side}-${role}`}>
              <button
                type="button"
                className="slot-main"
                onClick={() => onActivate({ side, role: roleIndex })}
                aria-label={`${role} de l'équipe ${side === "blue" ? "bleue" : "rouge"} : ${championName ?? "choisir un champion"}`}
                aria-current={isActive ? "true" : undefined}
              >
                <span className="slot-portrait">
                  {champion?.portrait_url ? <img src={champion.portrait_url} alt="" width="54" height="54" /> : <span>{String(roleIndex + 1).padStart(2, "0")}</span>}
                </span>
                <span className="slot-copy">
                  <span className="slot-role">{role}</span>
                  <span className="slot-name">{championName ?? "Choisir un champion"}</span>
                </span>
                {!championName && <span className="slot-plus" aria-hidden="true">+</span>}
              </button>
              {championName && (
                <button type="button" className="slot-clear" onClick={() => onClear({ side, role: roleIndex })} aria-label={`Retirer ${championName}`}>
                  ×
                </button>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

export default function Home() {
  const [roster, setRoster] = useState<Roster | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [picks, setPicks] = useState<Picks>(EMPTY_PICKS);
  const [activeSlot, setActiveSlot] = useState<Slot>({ side: "blue", role: 0 });
  const [selectedTeam, setSelectedTeam] = useState<Side>("blue");
  const [query, setQuery] = useState("");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [predictError, setPredictError] = useState<string | null>(null);
  const [predicting, setPredicting] = useState(false);

  async function loadRoster() {
    setLoadError(null);
    try {
      const response = await fetch("/api/champions", { cache: "no-store" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? "Impossible de charger les champions.");
      setRoster(payload as Roster);
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : "Impossible de charger les champions.");
    }
  }

  useEffect(() => {
    void loadRoster();
  }, []);

  const championByName = useMemo(
    () => new Map((roster?.champions ?? []).map((champion) => [champion.name, champion])),
    [roster],
  );
  const used = useMemo(
    () => new Set([...picks.blue, ...picks.red].filter((name): name is string => Boolean(name))),
    [picks],
  );
  const filteredChampions = useMemo(() => {
    const term = normalizeSearch(query);
    return (roster?.champions ?? []).filter((champion) => normalizeSearch(champion.name).includes(term));
  }, [roster, query]);
  const filledCount = used.size;
  const complete = filledCount === 10;
  const activeChampion = picks[activeSlot.side][activeSlot.role];
  const selectedProbability = prediction
    ? selectedTeam === "blue" ? prediction.blue_probability : prediction.red_probability
    : null;

  function selectChampion(champion: Champion) {
    if (used.has(champion.name) && activeChampion !== champion.name) return;
    const updated: Picks = { blue: [...picks.blue], red: [...picks.red] };
    updated[activeSlot.side][activeSlot.role] = champion.name;
    setPicks(updated);
    setActiveSlot(nextEmptySlot(updated, activeSlot));
    setPrediction(null);
    setPredictError(null);
  }

  function clearSlot(slot: Slot) {
    const updated: Picks = { blue: [...picks.blue], red: [...picks.red] };
    updated[slot.side][slot.role] = null;
    setPicks(updated);
    setActiveSlot(slot);
    setPrediction(null);
    setPredictError(null);
  }

  function resetDraft() {
    setPicks(EMPTY_PICKS());
    setActiveSlot({ side: "blue", role: 0 });
    setSelectedTeam("blue");
    setPrediction(null);
    setPredictError(null);
    setQuery("");
  }

  async function calculate() {
    if (!complete || !roster?.model_ready || predicting) return;
    setPredicting(true);
    setPredictError(null);
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ blue: picks.blue, red: picks.red, team: selectedTeam }),
      });
      const payload = await response.json();
      if (!response.ok) {
        const detail = typeof payload.detail === "string" ? payload.detail : "La prédiction a échoué.";
        throw new Error(detail);
      }
      setPrediction(payload as Prediction);
    } catch (error) {
      setPredictError(error instanceof Error ? error.message : "La prédiction a échoué.");
    } finally {
      setPredicting(false);
    }
  }

  return (
    <main className="app-shell">
      <div className="hero-art" aria-hidden="true" />
      <div className="ambient ambient-one" aria-hidden="true" />
      <div className="ambient ambient-two" aria-hidden="true" />

      <header className="site-header">
        <div className="brand">
          <Crest />
          <div className="brand-wordmark"><span>RIFT</span><strong>ORACLE</strong></div>
        </div>
        <div className="header-right">
          <span className="header-divider" />
          <span className="edition">PRO MATCH DATA <span>·</span> 2026</span>
          <span className={`status-pill ${roster?.model_ready ? "online" : "offline"}`}>
            <span className="status-dot" />{roster?.model_ready ? "MODÈLE PRÊT" : "MODÈLE HORS LIGNE"}
          </span>
        </div>
      </header>

      <div className="content">
        <section className="intro">
          <div className="intro-line"><span /> ANALYSE DE COMPOSITION <span /></div>
          <h1>Chaque choix <em>change la partie.</em></h1>
          <p>Compose les deux équipes. Explore les champions. Découvre les chances de victoire estimées par ton modèle.</p>
        </section>

        <section className="draft-section" aria-label="Sélection des équipes">
          <div className="section-kicker"><span>01</span><span>LA DRAFT</span><i /></div>
          <div className="draft-layout">
            <TeamPanel side="blue" picks={picks} activeSlot={activeSlot} championByName={championByName} onActivate={setActiveSlot} onClear={clearSlot} />

            <section className="prediction-panel" aria-label="Probabilité de victoire">
              <div className="prediction-topline"><span>PRÉDICTION</span><span>01 / 01</span></div>
              <div className="prediction-orb-wrap">
                <div className={`prediction-orb ${prediction ? "has-result" : ""} ${selectedTeam === "red" ? "red-focus" : ""}`} style={{ "--probability": `${(selectedProbability ?? 0) * 100}%` } as CSSProperties}>
                  <div className="orb-inner">
                    <span className="orb-label">CHANCES DE VICTOIRE</span>
                    <span className="orb-value">{selectedProbability === null ? "—" : `${(selectedProbability * 100).toFixed(1)}%`}</span>
                    <span className="orb-team">ÉQUIPE {selectedTeam === "blue" ? "BLEUE" : "ROUGE"}</span>
                  </div>
                </div>
                <span className="orb-flourish left" /><span className="orb-flourish right" />
              </div>
              <div className="focus-control" aria-label="Équipe à analyser">
                <span>MON ÉQUIPE</span>
                <div className="focus-buttons">
                  <button type="button" className={selectedTeam === "blue" ? "selected blue-choice" : ""} onClick={() => setSelectedTeam("blue")}>Bleue</button>
                  <button type="button" className={selectedTeam === "red" ? "selected red-choice" : ""} onClick={() => setSelectedTeam("red")}>Rouge</button>
                </div>
              </div>
              <div className="draft-progress">
                <div className="progress-label"><span>COMPOSITION</span><strong>{filledCount} / 10</strong></div>
                <div className="progress-track"><span style={{ width: `${filledCount * 10}%` }} /></div>
              </div>
              <button type="button" className="predict-button" disabled={!complete || !roster?.model_ready || predicting} onClick={calculate}>
                <span>{predicting ? "ANALYSE EN COURS…" : "CALCULER LES CHANCES"}</span><span aria-hidden="true">→</span>
              </button>
              {predictError && <p className="inline-error" role="alert">{predictError}</p>}
              {!roster?.model_ready && roster && <p className="panel-hint">Entraîne le modèle, puis relance l’API Python.</p>}
              {prediction && <p className="panel-hint result-hint">Estimation basée sur les compositions complètes de matchs professionnels.</p>}
            </section>

            <TeamPanel side="red" picks={picks} activeSlot={activeSlot} championByName={championByName} onActivate={setActiveSlot} onClear={clearSlot} />
          </div>
        </section>

        <section className="champion-section" aria-label="Liste des champions">
          <div className="section-kicker"><span>02</span><span>LE CHOIX DU CHAMPION</span><i /></div>
          <div className="champion-panel">
            <div className="champion-panel-head">
              <div>
                <p className="overline">LA SÉLECTION</p>
                <h2>Choisis ton champion<span>.</span></h2>
                <p className="champion-panel-subtitle">Sélection active : <strong>{activeSlot.side === "blue" ? "Équipe bleue" : "Équipe rouge"} · {ROLES[activeSlot.role]}</strong></p>
              </div>
              <button type="button" className="reset-button" onClick={resetDraft}>↻ <span>Réinitialiser la draft</span></button>
            </div>
            <div className="champion-toolbar">
              <div className="roster-tab"><span className="tab-diamond" /> TOUS LES CHAMPIONS <span className="roster-total">{roster?.champions.length ?? "—"}</span></div>
              <label className="search-box">
                <SearchIcon />
                <span className="sr-only">Rechercher un champion</span>
                <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Rechercher un champion..." autoComplete="off" />
                {query && <button type="button" aria-label="Effacer la recherche" onClick={() => setQuery("")}>×</button>}
              </label>
            </div>

            {loadError ? (
              <div className="roster-message" role="alert"><p>{loadError}</p><button type="button" onClick={() => void loadRoster()}>Réessayer</button></div>
            ) : !roster ? (
              <div className="roster-message">Chargement des champions…</div>
            ) : filteredChampions.length === 0 ? (
              <div className="roster-message">Aucun champion ne correspond à « {query} ».</div>
            ) : (
              <div className="champion-grid">
                {filteredChampions.map((champion) => {
                  const taken = used.has(champion.name);
                  const inActiveSlot = activeChampion === champion.name;
                  return (
                    <button
                      type="button"
                      className={`champion-card ${taken ? "taken" : ""} ${inActiveSlot ? "current" : ""}`}
                      key={champion.name}
                      onClick={() => selectChampion(champion)}
                      disabled={taken && !inActiveSlot}
                      title={taken && !inActiveSlot ? `${champion.name} est déjà sélectionné` : `Choisir ${champion.name}`}
                      aria-label={`${champion.name}${taken ? " — déjà sélectionné" : ""}`}
                    >
                      <span className="champion-image">
                        <span className="champion-initial">{champion.name.slice(0, 1)}</span>
                        {champion.portrait_url && <img src={champion.portrait_url} alt="" width="96" height="96" loading="lazy" />}
                      </span>
                      <span className="champion-name">{champion.name}</span>
                      {taken && <span className="taken-check" aria-hidden="true">✓</span>}
                    </button>
                  );
                })}
              </div>
            )}
            <div className="roster-footer"><span>CLIQUE SUR UN RÔLE, PUIS SUR UN CHAMPION</span><span>LA SÉLECTION AVANCE AUTOMATIQUEMENT</span></div>
          </div>
        </section>

        <footer className="site-footer">
          <span>RIFT ORACLE <i>◆</i> UN PROJET MLLOL</span>
          <p>Prédictions expérimentales sur des matchs professionnels de 2026. Visuels des champions © Riot Games. Projet non affilié à Riot Games.</p>
        </footer>
      </div>
    </main>
  );
}
