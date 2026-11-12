import Globe from "../components/Globe";
import { Panel, Pill, RepChart, Stat, Tier, Withheld } from "../components/ui";
import { colorFor, fmtDur } from "../api";

export default function FederationView({ view }: { view: any }) {
  const ops: string[] = view.world.operators;
  const r = view.rounds[0];
  const oneSided = view.encounters.filter((e: any) => e.senders.length < 2).length;
  return (
    <>
      <div className="role-banner federation">
        <b>Federation server.</b> This is everything the central server holds. It has never received a satellite position,
        orbit, miss distance or exact probability: only signed abstract signals, count reports, and the reputation it works out from them.
        {view.process?.separate && <div className="small proc">Running as its own process (pid {view.process.pid}). It has no orbit code loaded at all,
          and accepts a message only if it verifies against the public key that operator pinned when it first connected
          {view.registry && <>: {Object.entries(view.registry).map(([op, k]: any) => <span key={op} className="mono"> {op} {k}…</span>)}</>}.</div>}
      </div>
      <section className="tiles">
        <Stat value={0} label="satellite positions received" tone="good" />
        <Stat value={view.signals.length} label="signed signals (latest)" />
        <Stat value={view.encounters.length} label="encounters reported" />
        <Stat value={oneSided} label="reported by one side only" tone={oneSided ? "bad" : undefined} />
        <Stat value={view.ledger.entries.toLocaleString()} label={`ledger entries · ${view.ledger.verified_on_load ? "chain intact" : "TAMPERED"}`}
              tone={view.ledger.verified_on_load ? "good" : "bad"} />
      </section>

      <div className="grid">
        <div className="stack">
          <Panel title="The server's map">
            <Globe points={[]} height={300}
                   emptyMessage="No positions. The federation decides who manoeuvres without ever learning where anything is." />
          </Panel>

          <Panel title="Signed signals as received" extra="newest first">
            <div className="scroll">
              <table>
                <thead><tr><th>Received</th><th>Encounter</th><th>From</th><th>Severity</th><th>Geometry</th><th>TCA in</th><th>Signature</th></tr></thead>
                <tbody>
                  {view.signals.map((s: any) => (
                    <tr key={s.ledger_index}>
                      <td className="num">{s.received.slice(11, 19)}</td>
                      <td className="mono small">{s.encounter_id}</td>
                      <td style={{ color: colorFor(s.sender_operator, ops) }}>{s.sender_operator}</td>
                      <td><Pill kind={s.severity_tier}>{s.severity_tier}</Pill></td>
                      <td>{s.geometry_class}</td>
                      <td className="num">{fmtDur(s.tca_offset_s)}</td>
                      <td className="mono small muted">{s.signature}</td>
                    </tr>
                  ))}
                  {!view.signals.length && <tr><td colSpan={7} className="muted">No signals yet. In-house and below-threshold approaches never reach the server.</td></tr>}
                </tbody>
              </table>
            </div>
            <p className="note">Each row is an Ed25519 signature checked against the sender's registered key, chained into the tamper-evident ledger (head {view.ledger.tail_hash.slice(0, 12)}…).</p>
          </Panel>
        </div>

        <div className="stack">
          <Panel title="Reputation">
            <div className="reps">
              {ops.map((op) => (
                <div className="rep" key={op}>
                  <span style={{ color: colorFor(op, ops) }}>{op}</span>
                  <span className="bar"><i style={{ width: `${view.reputation.scores[op] * 100}%`, background: colorFor(op, ops) }} /></span>
                  <span className="mono">{view.reputation.scores[op].toFixed(2)}</span>
                  <Tier tier={view.reputation.tiers[op]} />
                </div>
              ))}
            </div>
            <RepChart operators={ops} history={view.reputation.history} />
            <p className="note">Worked out only from what operators sent: whether each one reported the encounters it is party to, invented ones nobody owns, or drifted from the consensus.</p>
          </Panel>

          <Panel title="Encounters reported" extra="by encounter id">
            <div className="scroll short">
              <table>
                <thead><tr><th>Encounter</th><th>Signed by</th><th>Status</th></tr></thead>
                <tbody>
                  {view.encounters.map((e: any) => (
                    <tr key={e.encounter_id}>
                      <td className="mono small">{e.encounter_id}</td>
                      <td>{e.senders.join(", ")}</td>
                      <td className={e.senders.length < 2 ? "bad" : "good"}>{e.status}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>

          <Panel title="Last federation round">
            {r ? (
              <>
                <div className="muted small">Round {r.round} · {r.sim_time.replace("T", " ").replace("Z", "")} · {r.n_signals} signals</div>
                <table>
                  <thead><tr><th>Operator</th><th className="num">Report</th><th className="num">Matched</th><th className="num">Unbacked</th></tr></thead>
                  <tbody>
                    {ops.map((op) => (
                      <tr key={op}><td>{op}</td><td className="num">{r.reports[op].join(" / ")}</td>
                        <td className="num">{r.evidence[op].matched}/{r.evidence[op].involved}</td>
                        <td className={`num ${r.evidence[op].unbacked ? "bad" : ""}`}>{r.evidence[op].unbacked}</td></tr>
                    ))}
                  </tbody>
                </table>
                <p className="note">Aggregate: plain mean [{r.aggregates.mean.join(", ")}], reputation-weighted [{r.aggregates.reputation_weighted.join(", ")}].
                  {r.errors && "mean" in r.errors && <> Error vs the honest operators: mean <b>{r.errors.mean}</b>, reputation-weighted <b>{r.errors.reputation_weighted}</b>.</>}</p>
              </>
            ) : <p className="muted">No round yet. Rounds happen every {fmtDur(view.world.federation_period_s)} of simulated time.</p>}
          </Panel>

          <Withheld items={view.withheld} who="the server" />
        </div>
      </div>
    </>
  );
}
