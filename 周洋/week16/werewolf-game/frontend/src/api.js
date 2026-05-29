const BASE = ''

export async function fetchConfigs() {
  const res = await fetch(`${BASE}/configs`)
  if (!res.ok) throw new Error('获取配置失败')
  return res.json()
}

export async function createGame(config) {
  const res = await fetch(`${BASE}/games`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || '创建游戏失败')
  }
  return res.json()
}

export async function stepGame(gameId) {
  const res = await fetch(`${BASE}/games/${gameId}/step`, { method: 'POST' })
  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || '推进阶段失败')
  }
  return res.json()
}

export async function getGameState(gameId) {
  const res = await fetch(`${BASE}/games/${gameId}`)
  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || '获取游戏状态失败')
  }
  return res.json()
}

export async function listGames() {
  const res = await fetch(`${BASE}/games`)
  return res.json()
}

export async function deleteGame(gameId) {
  const res = await fetch(`${BASE}/games/${gameId}`, { method: 'DELETE' })
  return res.json()
}
