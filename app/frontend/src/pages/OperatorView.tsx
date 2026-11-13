import { useEffect, useMemo, useState } from "react";
import Globe, { GlobePoint, GlobeTrack } from "../components/Globe";
import { Legend, Panel, Pill, Stat, Tier, Withheld } from "../components/ui";
import { api, colorFor, fmtDur, fmtMiss, Session } from "../api";

export default function OperatorView({ view, session }: { view: any; session: Session }) {
  const ops: string[] = view.world.operators;
  const me: string = view.operator;
  const [selected, setSelected] = useState<number | null>(null);
  const [track, setTrack] = useState<GlobeTrack | null>(null);
  const [hideOthers, setHideOthers] = useState(false);

  useEffect(() => {
    if (selected == null) { setTrack(null); return; }
    api(`/api/worlds/${view.world.name}/satellites/${selected}/track`, session)
      .then((t) => setTrack({ points: t.points, color: "#ffffff" }))
      .catch(() => setTrack(null));
  }, [selected, view.world.name, Math.floor(view.world.tick / 10)]);

  const points = useMemo<GlobePoint[]>(() => {
    const pts: GlobePoint[] = [];
    if (!hideOthers) {
      for (const o of view.public_catalogue) {
        pts.push({ lat: o.lat, lon: o.lon, alt_km: o.alt_km, color: o.operator === "DEBRIS" ? "#4a5270" : "#56607e" });
      }
    }
    for (const s of view.own_satellites) {
      pts.push({ lat: s.lat, lon: s.lon, alt_km: s.alt_km, color: s.in_encounter ? "#ff5468" : colorFor(me, ops),
                 highlight: s.in_encounter || s.id === selected });
    }
    return pts;
  }, [view, hideOthers, selected]);

  const upcoming = view.encounters.filter((e: any) => e.status === "upcoming");
  const passed = view.encounters.filter((e: any) => e.status !== "upcoming");
  const plans = view.encounters.filter((e: any) => e.plan).length;

  return (
    <>
      <div className="role-banner operator">
        <b>{me}'s digital twin.</b> Its own satellites in full; everyone else only as the public catalogue shows them
        (rounded to {view.public_grid_deg}°, about 50 km). Exact miss distances and Pc are computed here and never leave.
        {view.process?.separate && <div className="small proc">This twin is its own process (pid {view.process.pid}) and holds {me}'s private signing key;
          no other process has it.</div>}
      </div>
      <section className="tiles">
        <Stat value={view.own_count.toLocaleString()} label="my satellites (full state)" />
        <Stat value={view.public_count.toLocaleString()} label="other objects (public catalogue only)" />
        <Stat value={upcoming.length} label="close approaches coming up" tone={upcoming.length ? "warn" : undefined} />
        <Stat value={plans} label="joint manoeuvre plans" tone="good" />
        <Stat value={view.alerts.length} label="alerts received from the federation" />
        <Stat value={<>{view.reputation.score.toFixed(2)} <Tier tier={view.reputation.tier} /></>} label="my reputation" />
      </section>

      <div className="grid">
        <div className="stack">
          <Panel title="My sky" extra={view.own_satellites.length < view.own_count ? `(${view.own_satellites.length.toLocaleString()} of mine drawn)` : ""}>
            <Globe points={points} track={track} />
            <Legend items={[
              { color: colorFor(me, ops), label: `${me} (precise)` },
              { color: "#56607e", label: "others (public catalogue, ~50 km)" },
              { color: "#ff5468", label: "in an upcoming close approach" },
              ...(track ? [{ color: "#ffffff", label: "selected orbit, next 100 min" }] : []),
            ]} />
            <label className="check"><input type="checkbox" checked={hideOthers} onChange={(e) => setHideOthers(e.target.checked)} /> hide other operators</label>
          </Panel>

          <Panel title="My close approaches" extra={`look-ahead ${fmtDur(view.world.lookahead_s)}`}>
            <div className="scroll">
              <table>
                <thead><tr><th>When</th><th>My satellite</th><th>Other object</th><th>Miss</th><th>Pc</th><th>Handling</th><th>My move</th></tr></thead>
                <tbody>
                  {[...upcoming, ...passed].map((e: any) => (
                    <tr key={e.id} className={e.status !== "upcoming" ? "dim" : ""}>
                      <td className="num">{e.status === "upcoming" ? `in ${fmtDur(e.time_to_tca_s)}` : "passed"}</td>
                      <td><button className="link" onClick={() => setSelected(Number(e.my_satellite.split("-").pop()))}>{e.my_satellite}</button></td>
                      <td>{e.other_object}<div className="muted small">{e.geometry}, {e.rel_speed_km_s} km/s</div></td>
                      <td className="num">{fmtMiss(e.miss_km)}
                        {e.public_miss_km != null && <div className="muted small">public {fmtMiss(e.public_miss_km)}</div>}</td>
                      <td className="num">{e.pc.toExponential(1)}
                        {e.other_tle_age_h != null && <div className="muted small">other: {e.other_tle_age_h} h old TLE, σT {e.sigma_other_rtn_km[1]} km</div>}</td>
                      <td><Pill kind={e.handling}>{e.handling}</Pill>{e.handling === "federated" && !e.i_signalled && <div className="bad small">I stayed silent</div>}</td>
                      <td>{e.plan ? <span className="good">move {e.plan.direction === "+" ? "+" : "−"} (partner {e.plan.partner_direction === "+" ? "+" : "−"})</span>
                        : e.handling === "federated" ? <span className="bad">no joint plan</span> : <span className="muted">own call</span>}</td>
                    </tr>
                  ))}
                  {!view.encounters.length && <tr><td colSpan={7} className="muted">Nothing involving {me} yet.</td></tr>}
                </tbody>
              </table>
            </div>
          </Panel>
        </div>

        <div className="stack">
          <Panel title="Alerts from the federation" extra={`delivered at my tier: ${view.reputation.tier}`}>
            <ul className="feed">
              {view.alerts.map((a: any, i: number) => (
                <li key={a.encounter_id + i}>
                  <div className="t"><Pill kind="alert">{a.tier}</Pill>{a.received.replace("T", " ").replace("Z", "")}</div>
                  <div className="x mono small">{a.encounter_id}</div>
                  <div className="kv">
                    {"severity_tier" in a ? <span>severity <b>{a.severity_tier}</b></span> : <span className="locked">severity held back</span>}
                    {"tca_offset_s" in a ? <span>TCA was <b>{fmtDur(a.tca_offset_s)}</b> away when sent</span> : <span className="locked">timing held back</span>}
                    {"geometry_class" in a ? <span>geometry <b>{a.geometry_class}</b></span> : <span className="locked">geometry held back</span>}
                    {"counterparty_operator" in a ? <span>from <b>{a.counterparty_operator}</b></span> : <span className="locked">sender held back</span>}
                    {a.safety_override && <span className="warn">critical: sent in full regardless of tier</span>}
                  </div>
                </li>
              ))}
              {!view.alerts.length && <li className="muted">No alerts yet.</li>}
            </ul>
          </Panel>

          <Panel title="My satellites">
            <div className="scroll short">
              <table>
                <thead><tr><th>Satellite</th><th>Lat</th><th>Lon</th><th>Alt</th><th>Speed</th></tr></thead>
                <tbody>
                  {view.own_satellites.slice(0, 300).map((s: any) => (
                    <tr key={s.id} className={s.id === selected ? "sel" : ""} onClick={() => setSelected(s.id === selected ? null : s.id)}>
                      <td>{s.label}{s.in_encounter && <span className="bad"> ●</span>}</td>
                      <td className="num">{s.lat.toFixed(2)}°</td><td className="num">{s.lon.toFixed(2)}°</td>
                      <td className="num">{s.alt_km.toFixed(1)} km</td><td className="num">{s.speed_km_s.toFixed(3)} km/s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="note">Click a satellite to draw its orbit. Only its owner may request it: the API refuses anyone else.</p>
          </Panel>

          <Panel title="Reports I signed">
            <table>
              <thead><tr><th>Sent</th><th className="num">threshold / elevated / critical</th></tr></thead>
              <tbody>
                {view.my_reports.map((r: any) => <tr key={r.received}><td>{r.received.replace("T", " ").replace("Z", "")}</td><td className="num">{r.counts.join(" / ")}</td></tr>)}
                {!view.my_reports.length && <tr><td colSpan={2} className="muted">No federation round yet.</td></tr>}
              </tbody>
            </table>
            {view.behaviour !== "honest" && <p className="note bad">This operator is scripted as a <b>{view.behaviour}</b> in this world.</p>}
          </Panel>

          <Withheld items={view.withheld} who={me} />
        </div>
      </div>
    </>
  );
}
