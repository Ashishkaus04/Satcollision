import { useEffect, useMemo, useState } from "react";
import Globe, { GlobePoint } from "../components/Globe";
import { Legend, Panel } from "../components/ui";
import { pathLayers, pathLegend } from "../components/paths";
import { api, fmtDur, fmtMiss, Session } from "../api";

/** One encounter's avoidance path, on its own: only the satellites of that
 *  encounter (an operator only its own), their true track and the burn's
 *  effect, with the burn, the closest approach and "now" marked. Fetched on
 *  demand and refreshed every world tick. */
export default function PathView({ world, encounterId, session, version, onClose }: {
  world: string; encounterId: string; session: Session; version: number; onClose: () => void;
}) {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api(`/api/worlds/${world}/encounters/${encodeURIComponent(encounterId)}/path`, session)
      .then((d) => { if (!cancelled) { setData(d); setError(null); } })
      .catch((e) => { if (!cancelled) setError(String(e.message ?? e)); });
    return () => { cancelled = true; };
  }, [world, encounterId, session.token, version]);

  const layers = useMemo(() => pathLayers(data?.paths ?? []), [data]);
  const sats = useMemo<GlobePoint[]>(() => (data?.paths ?? []).flatMap((p: any) => {
    const k = Math.max(0, Math.min(p.now_index ?? 0, p.original.length - 1));
    const [lat, lon, alt] = p.original[k];
    return [{ lat, lon, alt_km: alt, color: "#3ec6ff", highlight: true }];
  }), [data]);

  const info = data?.info ?? {};
  const observer = data?.role === "observer";
  return (
    <>
      <div className="role-banner">
        <button className="btn small" onClick={onClose}>← Back to the dashboard</button>{" "}
        <b>Avoidance path</b> <span className="mono small">{encounterId}</span>
        <div className="small">Only this encounter's satellites are drawn{observer ? " (both, as flown)" : ": your own satellite, from your own plan"}.</div>
      </div>
      {error && <div className="banner">{error.includes("404") || error.toLowerCase().includes("no avoidance path")
        ? "No avoidance path for this encounter (no manoeuvre was planned, or it is not yours to see)." : error}</div>}
      {data && (
        <div className="grid">
          <div className="stack">
            <Panel title={observer ? (info.labels ?? []).join(" × ") : `${info.my_satellite} × ${info.other_object}`}
                   extra={info.status === "upcoming" ? `closest approach in ${fmtDur(info.time_to_tca_s)}` : "passed"}>
              <Globe points={[...sats, ...layers.markers]} tracks={layers.tracks} />
              <Legend items={[{ color: "#3ec6ff", label: "the satellite now" }, ...pathLegend(data.paths)]} />
            </Panel>
          </div>
          <div className="stack">
            <Panel title="The plan">
              <table>
                <thead><tr><th>Satellite</th><th>Burn</th><th>Closest approach</th><th className="num">Moved at TCA</th></tr></thead>
                <tbody>
                  {data.paths.map((p: any) => (
                    <tr key={p.satellite}>
                      <td>{p.satellite}</td>
                      <td className="mono small">{p.burn_time.replace("T", " ").replace("Z", "")}</td>
                      <td className="mono small">{p.tca.replace("T", " ").replace("Z", "")}</td>
                      <td className="num">{fmtMiss(p.shift_at_tca_km)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {observer ? (
                <div className="kv">
                  {info.truth && <span>true miss <b>{fmtMiss(info.truth.miss_km_without_maneuver ?? info.truth.miss_km)}</b> → <b className="good">{fmtMiss(info.truth.miss_km)}</b></span>}
                  {info.avoidance?.check?.pc_precise_after != null && <span>Pc (precise covariance) <b>{info.avoidance.check.pc_precise_before.toExponential(1)}</b> → <b className="good">{info.avoidance.check.pc_precise_after.toExponential(1)}</b></span>}
                  {info.avoidance?.total_dv_m_s != null && <span>total burn <b>{(info.avoidance.total_dv_m_s * 100).toFixed(1)} cm/s</b></span>}
                  {info.burns && Object.values(info.burns).map((b: any) => (
                    <span key={b.satellite}>{b.satellite}: <b>{(b.dv_m_s * 100).toFixed(1)} cm/s</b>{b.finite_burn_s != null ? ` (low thrust, ${b.finite_burn_s} s)` : ""}</span>))}
                </div>
              ) : (
                <div className="kv">
                  <span>my miss before <b>{fmtMiss(info.miss_km)}</b>{info.public_miss_km != null && <> (public {fmtMiss(info.public_miss_km)})</>}</span>
                  {info.my_burn && <span>my burn <b>{(info.my_burn.dv_m_s * 100).toFixed(1)} cm/s</b> at {info.my_burn.burn_time.slice(11, 16)}{info.my_burn.finite_burn_s != null ? ` (low thrust, ${info.my_burn.finite_burn_s} s)` : ""}</span>}
                  {info.my_burn && <span>Pc <b>{info.my_burn.pc_before.toExponential(1)}</b> → ≤ <b>{info.my_burn.target_pc.toExponential(0)}</b> {info.my_burn.target_met ? "(secure check: met)" : "(secure check: NOT met)"}</span>}
                </div>
              )}
              <p className="note">The satellite's dot rides the grey true track. The green track is the burn's effect, with the offset exaggerated so a kilometre-scale shift is visible. The part already flown is dimmed.</p>
            </Panel>
          </div>
        </div>
      )}
      {!data && !error && <div className="panel muted">Loading the path…</div>}
    </>
  );
}
