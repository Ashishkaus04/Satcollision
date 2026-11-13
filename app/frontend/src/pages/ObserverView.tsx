import { useMemo, useState } from "react";
import Globe, { GlobePoint } from "../components/Globe";
import { Legend, Panel, Pill, RepChart, Stat, Tier } from "../components/ui";
import { colorFor, fmtDur, fmtMiss, Session } from "../api";
import Compare from "./Compare";

export default function ObserverView({ view, session }: { view: any; session: Session }) {
  const [comparing, setComparing] = useState<string | null>(null);
  const ops: string[] = view.world.operators;
  const c = view.counters;
  const tr = view.tracks;
  const total = Object.values(view.fleet as Record<string, number>).reduce((a, b) => a + b, 0);

  const points = useMemo<GlobePoint[]>(() => tr.points.map((p: number[]) => ({
    lat: p[0], lon: p[1], alt_km: p[5], color: p[3] === 2 ? "#ff5468" : colorFor(tr.operators[p[2]], ops), highlight: p[3] === 2,
  })), [view]);

  if (comparing) {
    return <Compare world={view.world.name} encounterId={comparing} session={session} version={view.version} onClose={() => setComparing(null)} />;
  }

  const rows = [...view.upcoming, ...view.passed];
  return (
    <>
      <div className="role-banner observer">
        <b>Observer: ground truth.</b> {view.note} Pick any encounter and press <i>Compare</i> to see it as each operator and the server hold it.
        {view.processes && <div className="small proc">Separate processes: {Object.entries(view.processes).map(([k, pid]: any) =>
          <span key={k} className="mono"> {k === "app" ? "sky + app" : k} {pid}</span>)}</div>}
      </div>
      <section className="tiles">
        <Stat value={total.toLocaleString()} label={`objects · ${view.world.source}`} />
        <Stat value={c.encounters_shared} label="cross-operator encounters" />
        <Stat value={c.maneuvers_agreed} label="joint plans agreed" tone="good" />
        <Stat value={c.unilateral} label="partner stayed silent" tone={c.unilateral ? "bad" : undefined} />
        <Stat value={c.internal + c.private_watch} label="kept private / in-house" />
        <Stat value={c.federation_rounds} label="federation rounds" />
      </section>
      <div className="grid">
        <div className="stack">
          <Panel title="Everything, as it really is" extra={tr.points.length < total ? `(${tr.points.length.toLocaleString()} of ${total.toLocaleString()} drawn)` : ""}>
            <Globe points={points} />
            <Legend items={[...tr.operators.map((op: string) => ({ color: colorFor(op, ops), label: `${op} (${(view.fleet[op] ?? 0).toLocaleString()})` })),
              { color: "#ff5468", label: "in an upcoming encounter" }]} />
          </Panel>
          <Panel title="Encounters" extra={`${view.upcoming.length} upcoming`}>
            <div className="scroll">
              <table>
                <thead><tr><th>When</th><th>Pair</th><th>Miss</th><th>Pc</th><th>Handling</th><th /></tr></thead>
                <tbody>
                  {rows.map((e: any) => (
                    <tr key={e.id} className={e.status !== "upcoming" ? "dim" : ""}>
                      <td className="num">{e.status === "upcoming" ? `in ${fmtDur(e.time_to_tca_s)}` : "passed"}</td>
                      <td>{e.labels[0]} × {e.labels[1]}<div className="muted small">{e.geometry}{e.engineered ? " · engineered" : ""}</div></td>
                      <td className="num">{fmtMiss(e.miss_km)}
                        {e.truth && !e.observer_only && <div className="small">true {fmtMiss(e.truth.miss_km)}
                          {e.truth.miss_km_without_maneuver != null && <span className="good"> (was {fmtMiss(e.truth.miss_km_without_maneuver)} before the burns)</span>}</div>}
                        {e.observer_only && <div className="muted small">true miss; no twin flagged it</div>}
                        {e.avoidance?.total_dv_m_s != null && <div className="muted small">burns {(e.avoidance.total_dv_m_s * 100).toFixed(1)} cm/s in all · Pc {e.avoidance.check?.pc_precise_before?.toExponential(1)} → {e.avoidance.check?.pc_precise_after?.toExponential(1)} at the precise covariance{e.avoidance.check?.met ? "" : " (target NOT met)"}</div>}
                        {e.avoidance?.cannot_avoid && <div className="bad small">neither satellite can burn</div>}</td>
                      <td className="num">{e.pc.toExponential(1)}</td>
                      <td><Pill kind={e.handling}>{e.handling}</Pill></td>
                      <td><button className="btn small" onClick={() => setComparing(e.id)}>Compare</button></td>
                    </tr>
                  ))}
                  {!rows.length && <tr><td colSpan={6} className="muted">No encounters yet.</td></tr>}
                </tbody>
              </table>
            </div>
          </Panel>
        </div>
        <div className="stack">
          <Panel title="Reputation">
            <div className="reps">
              {ops.map((op) => (
                <div className="rep" key={op}>
                  <span style={{ color: colorFor(op, ops) }}>{op}{view.behaviours[op] !== "honest" && <span className="muted small"> ({view.behaviours[op]})</span>}</span>
                  <span className="bar"><i style={{ width: `${view.reputation.scores[op] * 100}%`, background: colorFor(op, ops) }} /></span>
                  <span className="mono">{view.reputation.scores[op].toFixed(2)}</span>
                  <Tier tier={view.reputation.tiers[op]} />
                </div>
              ))}
            </div>
            <RepChart operators={ops} history={view.reputation.history} />
          </Panel>
          <Panel title="What happened">
            <ul className="feed">
              {view.events.map((e: any, i: number) => (
                <li key={i}>
                  <div className="t"><Pill kind={e.kind}>{e.kind.replaceAll("_", " ")}</Pill>{e.sim_time.replace("T", " ").replace("Z", "")}</div>
                  <div className="x">{e.text}</div>
                </li>
              ))}
            </ul>
          </Panel>
        </div>
      </div>
    </>
  );
}
