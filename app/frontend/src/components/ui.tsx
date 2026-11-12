import { ReactNode } from "react";
import { colorFor } from "../api";

export function Panel({ title, extra, children, className = "" }: { title: string; extra?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`panel ${className}`}>
      <h2>{title} {extra && <span className="muted">{extra}</span>}</h2>
      {children}
    </section>
  );
}

export function Stat({ value, label, tone }: { value: ReactNode; label: string; tone?: "good" | "bad" | "warn" }) {
  return (
    <div className={`tile ${tone ?? ""}`}>
      <div className="v">{value}</div>
      <div className="l">{label}</div>
    </div>
  );
}

export function Pill({ kind, children }: { kind: string; children: ReactNode }) {
  return <span className={`pill p-${kind}`}>{children}</span>;
}

export function Tier({ tier }: { tier: string }) {
  return <span className={`tier t-${tier}`}>{tier}</span>;
}

export function Withheld({ items, who }: { items: string[]; who: string }) {
  if (!items.length) return null;
  return (
    <Panel title={`What ${who} never receives`}>
      <ul className="withheld">
        {items.map((w) => <li key={w}><span className="lock" aria-hidden>🔒</span>{w}</li>)}
      </ul>
    </Panel>
  );
}

export function RepChart({ operators, history }: { operators: string[]; history: Record<string, number>[] }) {
  const W = 420, H = 150, L = 30, B = 18, T = 6, R = 8;
  if (!history.length) return <p className="muted small">Reputation history appears after the first federation round.</p>;
  const n = history.length;
  const x = (i: number) => L + (n === 1 ? (W - L - R) / 2 : (i * (W - L - R)) / (n - 1));
  const y = (v: number) => T + (1 - v) * (H - T - B);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Reputation over federation rounds" className="chart">
      {[0, 0.25, 0.45, 0.7, 1].map((v) => (
        <g key={v}>
          <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="var(--border-soft)" />
          <text x={0} y={y(v) + 3}>{v}</text>
        </g>
      ))}
      <text x={L} y={H - 3}>first round shown</text>
      <text x={W - R - 26} y={H - 3}>now</text>
      {operators.map((op) => {
        const pts = history.map((h, j) => `${x(j).toFixed(1)},${y(Number.isFinite(h[op]) ? h[op] : 0.5).toFixed(1)}`).join(" ");
        return n === 1
          ? <circle key={op} cx={x(0)} cy={y(history[0][op] ?? 0.5)} r={3} fill={colorFor(op, operators)} />
          : <polyline key={op} fill="none" stroke={colorFor(op, operators)} strokeWidth={2} points={pts} />;
      })}
    </svg>
  );
}

export function Legend({ items }: { items: { color: string; label: string }[] }) {
  return (
    <div className="legend">
      {items.map((i) => <span key={i.label}><i className="sw" style={{ background: i.color }} />{i.label}</span>)}
    </div>
  );
}
