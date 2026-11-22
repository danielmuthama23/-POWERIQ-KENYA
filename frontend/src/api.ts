import { useCallback, useEffect, useRef, useState } from "react";

const BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? "";
const KEY_STORE = "poweriq_api_key";

export const getKey = (): string => { try { return sessionStorage.getItem(KEY_STORE) ?? ""; } catch { return ""; } };
export const setKey = (k: string) => { try { k ? sessionStorage.setItem(KEY_STORE, k) : sessionStorage.removeItem(KEY_STORE); } catch { /* storage unavailable */ } };

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string, public requestId?: string) { super(message); }
}

export async function api<T = any>(path: string, opts: { method?: string; body?: unknown; signal?: AbortSignal } = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/api/v1${path}`, {
      method: opts.method ?? "GET", signal: opts.signal,
      headers: { "Content-Type": "application/json", "X-API-Key": getKey() },
      body: opts.body ? JSON.stringify(opts.body) : undefined,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError(0, "network_error", "Cannot reach the POWERIQ API. Is the backend running?");
  }
  if (!res.ok) {
    let code = "http_error", msg = res.statusText, rid: string | undefined;
    try { const j = await res.json(); code = j.error?.code ?? code; msg = j.error?.message ?? msg; rid = j.error?.request_id; } catch { /* non-JSON */ }
    throw new ApiError(res.status, code, msg, rid);
  }
  return res.json();
}

export function useApi<T = any>(path: string | null, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(!!path);
  const ctl = useRef<AbortController | null>(null);
  const load = useCallback(() => {
    if (!path) return;
    ctl.current?.abort();
    const c = new AbortController(); ctl.current = c;
    setLoading(true); setError(null);
    api<T>(path, { signal: c.signal }).then(d => { setData(d); setLoading(false); })
      .catch(e => { if (e.name !== "AbortError") { setError(e); setLoading(false); } });
  }, [path]);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { load(); return () => ctl.current?.abort(); }, [load, ...deps]);
  return { data, error, loading, reload: load };
}
