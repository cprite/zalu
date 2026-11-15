// Server-side only: adds the API access token to proxied /api requests, so the hosted demo needs no setup
// and the token never reaches the browser. ZALU_API_TOKEN is read from the server's environment at runtime.
import { NextResponse, type NextRequest } from "next/server";

export function proxy(req: NextRequest) {
  const token = process.env.ZALU_API_TOKEN;
  if (!token) return NextResponse.next();
  const headers = new Headers(req.headers);
  headers.set("authorization", `Bearer ${token}`);
  return NextResponse.next({ request: { headers } });
}

export const config = { matcher: "/api/:path*" };
