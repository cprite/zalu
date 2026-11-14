"use client";
// API adapter: components call useApi() and never know whether data is SAMPLE (mock) or live.
import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { createLiveApi } from "./live";
import { createMockApi } from "./mock";
import type { ZaluApi } from "./types";

export { ApiError } from "./types";
export type { ZaluApi } from "./types";

export interface ApiConfig {
  baseUrl: string; // "" = mock
  token: string;
}

const KEY = "zalu.api";
const envDefault: ApiConfig = {
  baseUrl: process.env.NEXT_PUBLIC_ZALU_API ?? "",
  token: process.env.NEXT_PUBLIC_ZALU_TOKEN ?? "",
};

const Ctx = createContext<{ api: ZaluApi; config: ApiConfig; setConfig: (c: ApiConfig) => void } | null>(null);
const mockSingleton = createMockApi();

export function ApiProvider({ children }: { children: ReactNode }) {
  const [config, setConfigState] = useState<ApiConfig>(envDefault);
  useEffect(() => {
    try {
      const saved = localStorage.getItem(KEY);
      if (saved) setConfigState(JSON.parse(saved));
    } catch { /* ignore */ }
  }, []);
  const setConfig = (c: ApiConfig) => {
    setConfigState(c);
    localStorage.setItem(KEY, JSON.stringify(c));
  };
  const api = useMemo(() => (config.baseUrl ? createLiveApi(config.baseUrl, config.token) : mockSingleton), [config]);
  return <Ctx.Provider value={{ api, config, setConfig }}>{children}</Ctx.Provider>;
}

export function useApi() {
  const c = useContext(Ctx);
  if (!c) throw new Error("useApi outside ApiProvider");
  return c;
}
