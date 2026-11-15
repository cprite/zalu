"use client";

import { createLiveApi } from "./live";
import { createMockApi } from "./mock";
import type { ZaluApi } from "./types";

export { ApiError } from "./types";
export type { ZaluApi } from "./types";

// Demo mode is an explicit build choice. The default always uses the deployed API.
const api: ZaluApi = process.env.NEXT_PUBLIC_ZALU_MODE === "demo"
  ? createMockApi() : createLiveApi();

export function useApi() {
  return { api };
}
