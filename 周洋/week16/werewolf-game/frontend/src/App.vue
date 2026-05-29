<template>
  <div class="app">
    <header class="app-header">
      <h1>🐺 狼人杀 · 多智能体观战台</h1>
      <div class="header-info">
        <span v-if="gameId" class="game-id">游戏ID: {{ gameId }}</span>
        <span v-if="gameId && !gameOver" class="phase-tag">{{ phaseLabel }}</span>
      </div>
    </header>

    <main class="app-main">
      <!-- Left: Controls -->
      <aside class="panel-left">
        <GameCreator
          v-if="!gameId"
          :configs="configs"
          @create="onCreate"
        />
        <GameControl
          v-else
          :phase="currentPhase"
          :game-over="gameOver"
          :auto-running="autoRunning"
          @step="onStep"
          @auto="onAutoToggle"
          @reset="onReset"
        />
        <div v-if="gameId" class="phase-indicator">
          <div
            v-for="(ph, i) in allPhases"
            :key="i"
            class="phase-dot"
            :class="{ active: currentPhase === ph.id, done: phaseIndex > i }"
          >
            {{ ph.label }}
          </div>
        </div>
      </aside>

      <!-- Center: Player panel -->
      <section class="panel-center">
        <PlayerPanel :players="players" />
      </section>

      <!-- Right: Game log -->
      <aside class="panel-right">
        <GameLog :log="log" />
      </aside>
    </main>

    <!-- Win overlay -->
    <WinOverlay v-if="gameOver && winner" :winner="winner" @close="onReset" />
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { fetchConfigs, createGame, stepGame } from './api.js'
import GameCreator from './components/GameCreator.vue'
import GameControl from './components/GameControl.vue'
import PlayerPanel from './components/PlayerPanel.vue'
import GameLog from './components/GameLog.vue'
import WinOverlay from './components/WinOverlay.vue'

const allPhases = [
  { id: 'night_wolf',    label: '狼人刀人' },
  { id: 'night_seer',    label: '预言家验人' },
  { id: 'night_witch',   label: '女巫用药' },
  { id: 'night_result',  label: '死亡结算' },
  { id: 'day_start',     label: '天亮' },
  { id: 'speech',        label: '发言' },
  { id: 'vote',          label: '投票' },
  { id: 'day_end',       label: '日终' },
]

const configs = ref([])
const gameId = ref(null)
const players = ref([])
const currentPhase = ref('not_started')
const phaseIndex = ref(-1)
const phaseLabel = ref('')
const gameOver = ref(false)
const winner = ref(null)
const log = ref([])
const autoRunning = ref(false)
let autoTimer = null

const WINNER_LABELS = { good: '👼 好人阵营', evil: '🐺 狼人阵营' }

onMounted(async () => {
  configs.value = await fetchConfigs()
})

function pushLog(type, text) {
  log.value.push({ time: new Date().toLocaleTimeString(), type, text })
}

async function onCreate(config) {
  try {
    const data = await createGame(config)
    gameId.value = data.game_id
    currentPhase.value = data.phase
    gameOver.value = data.is_game_over
    players.value = await fetchPlayers(data.game_id)
    pushLog('system', `游戏 ${data.game_id} 已创建（${config.config_name}）`)
  } catch (e) {
    pushLog('error', e.message)
  }
}

async function fetchPlayers(gid) {
  const state = await stepGame(gid)
  // Store initial state players without advancing
  return state.players
}

async function onStep() {
  if (!gameId.value || gameOver.value) return
  try {
    const result = await stepGame(gameId.value)
    currentPhase.value = result.phase
    gameOver.value = result.is_game_over
    winner.value = result.winner
    players.value = result.players

    if (result.phase === 'game_over') {
      pushLog('system', `🏁 游戏结束！${WINNER_LABELS[result.winner] || result.winner} 获胜！`)
      return
    }

    const phaseInfo = allPhases.find(p => p.id === result.phase)
    phaseLabel.value = phaseInfo ? phaseInfo.label : result.phase
    phaseIndex.value = allPhases.findIndex(p => p.id === result.phase)

    // Log phase data
    const sd = result.step_data
    switch (result.phase) {
      case 'night_wolf':
        if (sd.final_target !== null)
          pushLog('action', `狼人决定击杀 玩家${sd.final_target + 1}`)
        break
      case 'night_seer':
        if (sd.seer_target !== null)
          pushLog('action', `预言家查验了 玩家${sd.seer_target + 1} → ${sd.result === 'wolf' ? '🐺 狼人' : '😇 好人'}`)
        break
      case 'night_witch':
        if (sd.saved) pushLog('action', '女巫使用解药救活了目标')
        if (sd.poison_target !== null) pushLog('action', `女巫使用毒药毒死了 玩家${sd.poison_target + 1}`)
        break
      case 'night_result':
        if (sd.deaths?.length) {
          for (const pid of sd.deaths) {
            const cause = sd.death_causes?.[pid] === 'poison' ? '毒药' : '狼刀'
            pushLog('death', `💀 玩家${pid + 1} 在夜晚死亡（${cause}）`)
          }
        } else {
          pushLog('action', '昨晚是平安夜')
        }
        if (sd.hunter_shot !== null) pushLog('death', `💀 猎人开枪带走了 玩家${sd.hunter_shot + 1}`)
        break
      case 'day_start':
        pushLog('info', sd.announcement)
        break
      case 'speech':
        pushLog('info', `--- 发言阶段（${result.day_number} 天）---`)
        break
      case 'vote':
        if (sd.eliminated !== null) {
          pushLog('action', `📢 玩家${sd.eliminated + 1} 被投票放逐`)
          if (sd.hunter_shot !== null) pushLog('death', `💀 猎人开枪带走了 玩家${sd.hunter_shot + 1}`)
        } else {
          pushLog('info', '投票平局，无人被放逐')
        }
        break
      case 'day_end':
        pushLog('info', `第 ${result.day_number} 天结束`)
        break
    }

    if (result.is_game_over) {
      pushLog('system', `🏁 游戏结束！${WINNER_LABELS[result.winner] || result.winner} 获胜！`)
    }
  } catch (e) {
    pushLog('error', e.message)
  }
}

function onAutoToggle() {
  autoRunning.value = !autoRunning.value
  if (autoRunning.value) {
    autoTimer = setInterval(onStep, 2000)
  } else {
    clearInterval(autoTimer)
  }
}

function onReset() {
  clearInterval(autoTimer)
  autoRunning.value = false
  gameId.value = null
  players.value = []
  currentPhase.value = 'not_started'
  phaseIndex.value = -1
  phaseLabel.value = ''
  gameOver.value = false
  winner.value = null
  log.value = []
}
</script>

<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1923; color: #e0e0e0; }
</style>

<style scoped>
.app { min-height: 100vh; display: flex; flex-direction: column; }
.app-header {
  background: linear-gradient(135deg, #1a2a3a, #0d1b2a);
  padding: 16px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid #2a3a4a;
}
.app-header h1 { font-size: 1.4rem; font-weight: 700; color: #f0c040; }
.header-info { display: flex; gap: 12px; align-items: center; }
.game-id { font-size: 0.8rem; color: #8899aa; background: #1a2a3a; padding: 4px 10px; border-radius: 4px; }
.phase-tag {
  background: #2a5a3a;
  color: #80e0a0;
  padding: 4px 12px;
  border-radius: 12px;
  font-size: 0.85rem;
  font-weight: 600;
}
.app-main {
  flex: 1;
  display: grid;
  grid-template-columns: 240px 1fr 280px;
  gap: 16px;
  padding: 16px;
  max-height: calc(100vh - 64px);
  overflow: hidden;
}
.panel-left { display: flex; flex-direction: column; gap: 16px; }
.panel-center { overflow-y: auto; }
.panel-right { overflow-y: auto; }
.phase-indicator { display: flex; flex-direction: column; gap: 6px; }
.phase-dot {
  padding: 6px 12px;
  border-radius: 6px;
  font-size: 0.78rem;
  background: #1a2a3a;
  color: #668;
  border-left: 3px solid #334;
}
.phase-dot.active { background: #1a3a2a; color: #80e0a0; border-left-color: #40c080; }
.phase-dot.done { background: #2a2a3a; color: #8899aa; border-left-color: #668; }
</style>
