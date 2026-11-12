import { FormEvent, useEffect, useState } from "react";
import { api, ApiError, fmtTime, loadSession, saveSession, Session, useLiveView } from "./api";
import OperatorView from "./pages/OperatorView";
import FederationView from "./pages/FederationView";
import ObserverView from "./pages/ObserverView";

type DemoUser = { username: string; role: string; operator?: string };

function Login({ onLogin }: { onLogin: (s: Session) => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [demo, setDemo] = useState<{ enabled: boolean; password?: string; users: DemoUser[] } | null>(null);

  useEffect(() => { api("/api/demo-users", null).then(setDemo).catch(() => setDemo(null)); }, []);

  const submit = async (u = username, p = password, e?: FormEvent) => {
    e?.preventDefault();
    setErr(null);
    try {
      onLogin(await api<Session>("/api/login", null, { method: "POST", body: JSON.stringify({ username: u, password: p }) }));
    } catch (x) {
      setErr(x instanceof ApiError ? x.message : "Can't reach the server.");
    }
  };

  const groups: [string, (u: DemoUser) => boolean, string][] = [
    ["Operators (synthetic worlds)", (u) => ["alpha", "beta", "gamma"].includes(u.username), "Their own digital twin"],
    ["Operators (real LEO)", (u) => u.role === "operator" && !["alpha", "beta", "gamma"].includes(u.username), "Real catalogue fleets"],
    ["Federation and demo", (u) => u.role !== "operator", "Server view, and ground truth"],
  ];

  return (
    <div className="login-wrap">
      <div className="login panel">
        <div className="eyebrow">Federated digital twins</div>
        <h1>Conjunction Watch</h1>
        <p className="muted">Sign in as an operator, the federation server, or the observer. Each login sees only what that party is allowed to see.</p>
        <form onSubmit={(e) => submit(username, password, e)}>
          <label>Username<input value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" /></label>
          <label>Password<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" /></label>
          <button className="btn" type="submit">Sign in</button>
          {err && <div className="bad small">{err}</div>}
        </form>
        {demo?.enabled && (
          <div className="demo-users">
            <div className="muted small">Demo accounts (password <span className="mono">{demo.password}</span>):</div>
            {groups.map(([title, pick, hint]) => (
              <div key={title}>
                <div className="small">{title} <span className="muted">· {hint}</span></div>
                <div className="chips">
                  {demo.users.filter(pick).map((u) => (
                    <button key={u.username} className="chip" onClick={() => submit(u.username, demo.password!)}>
                      {u.operator ?? u.username}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [session, setSession] = useState<Session | null>(loadSession());
  const [worlds, setWorlds] = useState<any[]>([]);
  const [world, setWorld] = useState<string | null>(null);
  const live = useLiveView(world, session);

  useEffect(() => {
    if (!session) return;
    api<any[]>("/api/worlds", session)
      .then((ws) => {
        setWorlds(ws);
        let saved: string | null = null;
        try { saved = localStorage.getItem("satcollision.world"); } catch { /* ignore */ }
        setWorld(ws.find((w) => w.name === saved)?.name ?? ws[0]?.name ?? null);
      })
      .catch((e) => { if (e instanceof ApiError && e.status === 401) logout(); });
  }, [session?.token]);

  const login = (s: Session) => { saveSession(s); setSession(s); };
  const logout = () => { saveSession(null); setSession(null); setWorlds([]); setWorld(null); };
  const pick = (name: string) => { setWorld(name); try { localStorage.setItem("satcollision.world", name); } catch { /* ignore */ } };

  if (!session) return <Login onLogin={login} />;

  const v = live.data as any;
  const who = session.role === "operator" ? `${session.operator} operator` : session.role === "federation" ? "Federation server" : "Observer";
  return (
    <div className="wrap">
      <header>
        <div>
          <div className="eyebrow">Federated digital twins · always running</div>
          <h1>Conjunction Watch</h1>
        </div>
        <div className="header-right">
          <nav className="tabs" role="tablist">
            {worlds.map((w) => (
              <button key={w.name} role="tab" aria-selected={w.name === world} className="tab" onClick={() => pick(w.name)}>
                <span className={`dot ${w.error ? "off" : "on"}`} />{w.title}
              </button>
            ))}
          </nav>
          <div className="who">
            <span className={`badge ${session.role}`}>{who}</span>
            <button className="btn ghost small" onClick={logout}>Sign out</button>
          </div>
        </div>
      </header>

      {!worlds.length && <div className="panel muted">No world is open to this login. {session.role === "operator" && `${session.operator} doesn't operate in any running world.`}</div>}
      {live.error && <div className="banner">{live.error}</div>}

      {v && (
        <>
          <section className="intro">
            <p>{v.world.description}</p>
            <div className="clock">
              <div className="big">{fmtTime(v.world.sim_time)}</div>
              <div className="small">{v.world.speed}× · tick {v.world.tick} · {v.world.source}</div>
              <div className="small"><span className={`dot inline ${live.connected ? "on" : "off"}`} />{live.connected ? "live" : "reconnecting (polling)"}</div>
            </div>
          </section>
          {v.role === "operator" && <OperatorView view={v} session={session} />}
          {v.role === "federation" && <FederationView view={v} />}
          {v.role === "observer" && <ObserverView view={v} session={session} />}
        </>
      )}
      {!v && world && !live.error && <div className="panel muted">Connecting to {world}…</div>}
    </div>
  );
}
