import { mkdir, readFile, writeFile } from "node:fs/promises";

const root = new URL("../../", import.meta.url);
const sourceFiles = {
  "src/app/page.tsx": "frontend/export-template/page.tsx",
  "src/app/layout.tsx": "frontend/export-template/layout.tsx",
  "src/app/styles.css": "frontend/export-template/styles.css",
  "src/dashboard/DashboardRenderer.tsx": "frontend/src/dashboard/DashboardRenderer.tsx",
  "src/dashboard/dashboard.css": "frontend/src/dashboard/dashboard.css",
  "src/dashboard/session.ts": "frontend/src/dashboard/session.ts",
  "src/dashboard/useInferenceSession.ts": "frontend/src/dashboard/useInferenceSession.ts",
  "src/dashboard/dashboard-spec.ts": "shared/dashboard_spec/dashboard-spec.ts",
  "dashboard-spec.v1.schema.json": "shared/dashboard_spec/dashboard-spec.v1.schema.json",
  "README.md": "frontend/export-template/README.md",
};
const files = {};
for (const [destination, source] of Object.entries(sourceFiles)) {
  const content = await readFile(new URL(source, root), "utf8");
  files[destination] = content.replaceAll("../../../shared/dashboard_spec/dashboard-spec", "./dashboard-spec");
}
const pkg = JSON.parse(await readFile(new URL("frontend/package-lock.json", root), "utf8"));
const versions = (names) => Object.fromEntries(names.map((name) => [name, pkg.packages[`node_modules/${name}`].version]));
files["package.json"] = JSON.stringify({
  name: "zalu-dashboard",
  version: "1.0.0",
  private: true,
  scripts: { dev: "next dev", build: "next build", start: "next start" },
  dependencies: versions(["next", "react", "react-dom"]),
  devDependencies: versions(["typescript", "@types/node", "@types/react", "@types/react-dom"]),
}, null, 2) + "\n";

const destination = new URL("frontend/src/dashboard/generated/", root);
await mkdir(destination, { recursive: true });
await writeFile(new URL("export-files.json", destination), JSON.stringify(files), "utf8");
