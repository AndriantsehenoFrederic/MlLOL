const apiUrl = process.env.MLLOL_API_URL ?? "http://127.0.0.1:8000";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const response = await fetch(`${apiUrl}/champions`, {
      cache: "no-store",
      signal: AbortSignal.timeout(10_000),
    });
    const payload = await response.json();
    return Response.json(payload, { status: response.status });
  } catch {
    return Response.json(
      { detail: "L'API Python ne répond pas. Lance uvicorn sur le port 8000." },
      { status: 503 },
    );
  }
}
