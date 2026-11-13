const apiUrl = process.env.MLLOL_API_URL ?? "http://127.0.0.1:8000";

export async function POST(request: Request) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return Response.json({ detail: "Requête JSON invalide." }, { status: 400 });
  }

  try {
    const response = await fetch(`${apiUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      cache: "no-store",
      signal: AbortSignal.timeout(30_000),
    });
    const payload = await response.json();
    return Response.json(payload, { status: response.status });
  } catch {
    return Response.json(
      { detail: "La prédiction est indisponible. Vérifie que l'API Python fonctionne." },
      { status: 503 },
    );
  }
}
