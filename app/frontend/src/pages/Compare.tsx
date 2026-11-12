import { ReactNode, useEffect, useState } from "react";
import { api, colorFor, fmtDur, fmtMiss, Session } from "../api";
import { Pill } from "../components/ui";

const Locked = ({ why = "never received" }: { why?: string }) => <span className="locked">🔒 {why}</span>;

type Row = { label: string; op: (side: any) => ReactNode; server: (srv: any) => ReactNode };

const ROWS: Row[] = [
  {
    label: "Own satellite, exact position",
    op: (s) => s.own_satellite
      ? <span className="mono">{s.own_satellite.label}: {s.own_satellite.lat.toFixed(3)}°, {s.own_satellite.lon.toFixed(3)}°, {s.own_satellite.alt_km.toFixed(2)} km, {s.own_satellite.speed_km_s.toFixed(3)} km/s</span>
      : <span className="muted">retired</span>,
    server: () => <Locked />,
  },
  {
    label: "The other object's position",
    op: (s) => s.partner_as_seen
      ? <span className="mono">{s.partner_as_seen.label}: {s.partner_as_seen.lat}°, {s.partner_as_seen.lon}°, ~{s.partner_as_seen.alt_km} km <span className="muted">(public catalogue, ~50 km)</span></span>
      : <span className="muted">not in view</span>,
    server: () => <Locked />,
  },
  {
    label: "Miss distance",
    op: (s) => s.encounter ? <b>{fmtMiss(s.encounter.miss_km)}</b> : <Locked />,
    server: () => <Locked />,
  },
  {
    label: "Collision probability",
    op: (s) => s.encounter ? <b className="mono">{s.encounter.pc.toExponential(2)}</b> : <Locked />,
    server: () => <Locked why="only a coarse tier" />,
  },
  {
    label: "Severity tier",
    op: (s) => s.alerts[0]?.severity_tier
      ? <>partner says <Pill kind={s.alerts[0].severity_tier}>{s.alerts[0].severity_tier}</Pill></>
      : s.alerts[0] ? <Locked why={`held back at tier ${s.alerts[0].tier}`} /> : <span className="muted">no alert from partner</span>,
    server: (v) => v.signals.length
      ? <>{v.signals.map((x: any) => <Pill key={x.ledger_index} kind={x.severity_tier}>{x.sender_operator}: {x.severity_tier}</Pill>)}</>
      : <Locked why="never reported" />,
  },
  {
    label: "Geometry",
    op: (s) => s.encounter ? <>{s.encounter.geometry}, {s.encounter.rel_speed_km_s} km/s</> : <Locked />,
    server: (v) => v.signals[0] ? <>{v.signals[0].geometry_class} <span className="muted">(class only)</span></> : <Locked why="never reported" />,
  },
  {
    label: "Closest approach",
    op: (s) => s.encounter ? <>{s.encounter.time_to_tca_s > 0 ? `in ${fmtDur(s.encounter.time_to_tca_s)}` : `passed ${fmtDur(s.encounter.time_to_tca_s)}`}</> : <Locked />,
    server: (v) => v.signals[0] ? <>was {fmtDur(v.signals[0].tca_offset_s)} away when signalled</> : <Locked why="never reported" />,
  },
  {
    label: "Signed signal",
    op: (s) => s.encounter
      ? s.encounter.handling !== "federated" ? <span className="muted">not shared ({s.encounter.handling})</span>
      : s.encounter.i_signalled ? <span className="good">sent and signed</span> : <span className="bad">stayed silent</span>
      : <Locked />,
    server: (v) => v.signals.length
      ? <>{v.signals.map((x: any) => <div key={x.ledger_index} className="mono small">{x.sender_operator} · sig {x.signature} · ledger #{x.ledger_index}</div>)}</>
      : <Locked why="nothing arrived" />,
  },
  {
    label: "Manoeuvre",
    op: (s) => s.encounter?.plan
      ? <span className="good">move {s.encounter.plan.my_satellite} along {s.encounter.plan.direction === "+" ? "+" : "−"} axis</span>
      : <span className="muted">{s.encounter?.handling === "federated" ? "no joint plan: partner silent" : "own decision"}</span>,
    server: () => <span className="muted">not computed by the server: both sides derive it from the encounter id</span>,
  },
];

export default function Compare({ world, encounterId, session, version, onClose }: {
  world: string; encounterId: string; session: Session; version: number; onClose: () => void;
}) {
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api(`/api/worlds/${world}/compare/${encounterId}`, session).then(setData).catch((e) => setErr(e.message));
  }, [world, encounterId, version]);

  if (err) return <div className="panel bad">Couldn't load this encounter: {err}</div>;
  if (!data) return <div className="panel muted">Loading…</div>;
  const t = data.truth;
  const ops: string[] = data.world.operators;
  const cols = [...data.operators.map((s: any) => ({ kind: "op", side: s })), { kind: "server", side: data.server }];

  return (
    <section className="panel compare">
      <div className="compare-head">
        <div>
          <h2>One encounter, three views</h2>
          <div className="mono small">{data.encounter_id}</div>
        </div>
        <button className="btn ghost" onClick={onClose}>Back to the world</button>
      </div>
      <div className="truth">
        <b>Ground truth (observer only):</b> {t.labels[0]} × {t.labels[1]}, {fmtMiss(t.miss_km)} apart at {t.tca.replace("T", " ").replace("Z", " UTC")},
        Pc {t.pc.toExponential(2)}, {t.geometry}. Handled as <Pill kind={t.handling}>{t.handling}</Pill>.
      </div>
      <div className="compare-scroll">
        <table className="compare-table">
          <thead>
            <tr>
              <th />
              {cols.map((c: any) => (
                <th key={c.side.party} style={{ color: c.kind === "op" ? colorFor(c.side.party, ops) : "var(--warn)" }}>
                  {c.kind === "op" ? `${c.side.party}'s twin` : "Federation server"}
                  {c.kind === "op" && <div className="muted small">reputation {c.side.reputation.score.toFixed(2)} · {c.side.reputation.tier}</div>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ROWS.map((row) => (
              <tr key={row.label}>
                <th scope="row">{row.label}</th>
                {cols.map((c: any) => <td key={c.side.party}>{c.kind === "op" ? row.op(c.side) : row.server(c.side)}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="note">Every cell comes from the same server-side filter that each party's own login uses, so this page cannot show a party anything its own view would not.</p>
    </section>
  );
}
