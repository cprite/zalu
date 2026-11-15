"use client";

import { useState, type FormEvent } from "react";
import { DashboardRenderer } from "../dashboard/DashboardRenderer";
import type { DashboardSpec } from "../dashboard/dashboard-spec";
// These JSON files are written when the user downloads the project.
// @ts-ignore -- generated in the ZIP
import dashboard from "../dashboard/dashboard.json";
// @ts-ignore -- generated in the ZIP
import fixture from "../dashboard/sample-data.json";
import { useInferenceSession } from "../dashboard/useInferenceSession";
import type { FrameResult } from "../dashboard/session";

type Connection = { url: string; model: string; token: string };

export default function Page() {
  const [connection, setConnection] = useState<Connection | null>(null);
  const [configurationError, setConfigurationError] = useState<string | null>(null);

  function connect(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    try {
      const url = new URL(String(data.get("url") ?? ""));
      if (!["https:", "http:"].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
        throw new Error("Enter an HTTP(S) backend URL without credentials, query or fragment.");
      }
      if (url.protocol === "http:" && !["127.0.0.1", "localhost", "[::1]"].includes(url.hostname)) {
        throw new Error("Use HTTPS for a remote backend.");
      }
      const model = String(data.get("model") ?? "").trim();
      if (!model) throw new Error("Enter a model ID.");
      setConnection({ url: url.href.replace(/\/+$/, ""), model, token: String(data.get("token") ?? "") });
      event.currentTarget.reset();
      setConfigurationError(null);
    } catch (error) {
      setConfigurationError((error as Error).message);
    }
  }

  return <main>
    <Session key={connection ? `${connection.url}:${connection.model}` : "sample"} connection={connection} />
    <details>
      <summary>Data connection</summary>
      <form onSubmit={connect}>
        <label>Backend URL<input name="url" type="url" placeholder="http://localhost:9100" required /></label>
        <label>Model ID<input name="model" required /></label>
        <label>Access token<input name="token" type="password" autoComplete="off" /></label>
        <div className="controls">
          <button>Connect live</button>
          <button type="button" onClick={() => { setConnection(null); setConfigurationError(null); }}>Use sample data</button>
        </div>
        {configurationError && <p role="alert">{configurationError}</p>}
      </form>
    </details>
  </main>;
}

function Session({ connection }: { connection: Connection | null }) {
  const { classes, target_event, confidence } = fixture as { classes: string[]; target_event: string; confidence: number };

  const session = useInferenceSession(async (frame): Promise<FrameResult> => {
    if (!connection) throw new Error("Connect a backend first.");
    const headers = new Headers();
    if (connection.token) headers.set("Authorization", `Bearer ${connection.token}`);
    const body = new FormData();
    body.set("model_id", connection.model);
    body.set("image", frame, "frame.jpg");
    body.set("conf", String(confidence));
    if (target_event) body.set("target_event", target_event);
    const response = await fetch(`${connection.url}/api/inference`, { method: "POST", headers, body, redirect: "error" });
    if (!response.ok) throw new Error(`Inference returned HTTP ${response.status}.`);
    return response.json();
  }, !connection, classes, target_event);

  return <>
    <DashboardRenderer spec={dashboard as DashboardSpec} classes={classes} samples={session.samples}
      preview={session.preview} loading={session.loading} error={session.error} sampleMode={!connection} />
    <div className="controls">
      <button onClick={session.running ? session.stop : session.start}>
        {session.running ? "Stop inference" : connection ? "Start camera" : "Run sample inference"}
      </button>
      {connection && <label>Choose image<input type="file" accept="image/*" onChange={(event) => {
        const file = event.target.files?.[0];
        if (file) void session.useImage(file);
        event.target.value = "";
      }} /></label>}
    </div>
  </>;
}
