function json(payload, init = {}) {
  return Response.json(payload, {
    ...init,
    headers: {
      "cache-control": "no-store",
      ...(init.headers ?? {}),
    },
  });
}

function validateLatentId(value) {
  const latentId = String(value ?? "").trim();
  if (!/^\d{1,10}$/.test(latentId)) {
    throw new Error("latentId must be a numeric string");
  }
  return latentId;
}

function normalizeLabel(value) {
  return String(value ?? "").trim();
}

export async function onRequestGet(context) {
  const db = context.env.DB;
  if (!db) return json({ error: "D1 binding DB is not configured" }, { status: 503 });

  const { results } = await db
    .prepare("SELECT latent_id, label FROM latent_labels ORDER BY CAST(latent_id AS INTEGER)")
    .all();
  const labels = Object.fromEntries((results ?? []).map((row) => [String(row.latent_id), String(row.label)]));
  return json({ labels });
}

export async function onRequestPut(context) {
  const db = context.env.DB;
  if (!db) return json({ error: "D1 binding DB is not configured" }, { status: 503 });

  let payload;
  try {
    payload = await context.request.json();
  } catch (_error) {
    return json({ error: "Expected JSON body" }, { status: 400 });
  }

  let latentId;
  try {
    latentId = validateLatentId(payload.latentId);
  } catch (error) {
    return json({ error: error.message }, { status: 400 });
  }

  const label = normalizeLabel(payload.label);
  if (!label) {
    await db.prepare("DELETE FROM latent_labels WHERE latent_id = ?").bind(latentId).run();
    return json({ latentId, label: "" });
  }

  if (label.length > 500) {
    return json({ error: "label must be 500 characters or fewer" }, { status: 400 });
  }

  await db
    .prepare(
      `INSERT INTO latent_labels (latent_id, label, updated_at)
       VALUES (?, ?, CURRENT_TIMESTAMP)
       ON CONFLICT(latent_id) DO UPDATE SET
         label = excluded.label,
         updated_at = excluded.updated_at`,
    )
    .bind(latentId, label)
    .run();
  return json({ latentId, label });
}
