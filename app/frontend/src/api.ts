import { useEffect, useState } from "react";

export type Session = { token: string; role: "operator" | "federation" | "observer"; operator?: string | null; username: string };

const KEY = "satcollision.session";

export function loadSession(): Session | null {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as Session) : null;
  } catch {
    return null;
  }
}

export function saveSession(s: Session | null) {
  try {
    if (s) localStorage.setItem(KEY, JSON.stringify(s));
    else localStorage.removeItem(KEY);
  } catch {
    /* private window: session just won't survive a reload */
  }
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function api<T = any>(path: string, session: Session | null, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (session) headers.Authorization = `Bearer ${session.token}`;
  const r = await fetch(path, { ...init, headers: { ...headers, ...(init.headers as any) } });
  if (!r.ok) {
    let msg = r.statusText;
    try {
      msg = (await r.json()).detail ?? msg;
    } catch {
      /* not JSON */
    }
    throw new ApiError(r.status, msg);
  }
  return r.json();
}

/** Live role view of one world: WebSocket push, falling back to polling.
 *
 * Every run of the effect owns its socket, poll and retry timer, and its own
 * ``cancelled`` flag. (A flag shared between runs was the bug behind views
 * flipping between worlds and logins: the next run reset it before the old
 * socket's close event arrived, so the old socket started polling and
 * reconnecting with the old world and the old token, for good.) Anything that
 * arrives for another world or another party is dropped as well. */
export function useLiveView<T = any>(world: string | null, session: Session | null) {
  const [data, setData] = useState<T | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setData(null);
    setConnected(false);
    setError(null);
    if (!world || !session) return;
    let cancelled = false;
    let ws: WebSocket | null = null;
    let poll: number | undefined;
    let retry: number | undefined;

    const accept = (d: any) => {
      if (cancelled || !d || d.world?.name !== world) return;
      if (session.role === "operator" ? d.role !== "operator" || d.operator !== session.operator : d.role !== session.role) return;
      setData(d);
      setError(null);
    };

    const startPolling = () => {
      if (poll || cancelled) return;
      const tick = () =>
        api<T>(`/api/worlds/${world}/view`, session)
          .then(accept)
          .catch((e) => { if (!cancelled) setError(String(e.message ?? e)); });
      tick();
      poll = window.setInterval(tick, 3000);
    };

    const connect = () => {
      if (cancelled) return;
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const sock = new WebSocket(`${proto}://${location.host}/api/worlds/${world}/live?token=${encodeURIComponent(session.token)}`);
      ws = sock;
      sock.onopen = () => {
        if (cancelled) { sock.close(); return; }
        setConnected(true); setError(null);
        if (poll) { clearInterval(poll); poll = undefined; }
      };
      sock.onmessage = (ev) => accept(JSON.parse(ev.data));
      sock.onclose = (ev) => {
        if (cancelled || ws !== sock) return;
        setConnected(false);
        if (ev.code === 4401 || ev.code === 4403) { setError("Not allowed to view this world."); return; }
        startPolling();
        retry = window.setTimeout(connect, 5000);
      };
    };
    connect();
    return () => {
      cancelled = true;
      ws?.close();
      if (poll) clearInterval(poll);
      if (retry) clearTimeout(retry);
    };
  }, [world, session?.token]);

  return { data, connected, error };
}

export const fmtDur = (s: number) => {
  const neg = s < 0;
  s = Math.abs(Math.round(s));
  let out: string;
  if (s >= 86400) out = (s / 86400).toFixed(1) + " d";
  else if (s >= 3600) out = `${Math.floor(s / 3600)} h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")} m`;
  else if (s >= 60) out = `${Math.floor(s / 60)} m ${String(s % 60).padStart(2, "0")} s`;
  else out = `${s} s`;
  return neg ? `${out} ago` : out;
};
export const fmtMiss = (km: number) => (km < 1 ? `${Math.round(km * 1000)} m` : `${km.toFixed(2)} km`);
export const fmtTime = (iso: string) => iso.replace("T", " ").replace("Z", " UTC");
export const PALETTE = ["#3ec6ff", "#31d6a8", "#ffc247", "#b48cff", "#ff8fb1", "#7dd3fc"];
export const colorFor = (op: string, operators: string[]) =>
  op === "DEBRIS" ? "#66708f" : PALETTE[Math.max(0, operators.indexOf(op)) % PALETTE.length];
