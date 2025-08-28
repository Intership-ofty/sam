export async function apiGet(path: string) {
  const res = await fetch(`/api${path}`, { credentials: 'include' })
  if (!res.ok) throw new Error(await res.text())
  return res.json().catch(()=> ({}))
}
export async function apiPost(path: string, body?: any) {
  const res = await fetch(`/api${path}`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {})
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json().catch(()=> ({}))
}
