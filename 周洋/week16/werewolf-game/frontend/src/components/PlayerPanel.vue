<template>
  <div class="panel">
    <h3>👥 玩家列表（{{ aliveCount }}/{{ players.length }} 存活）</h3>
    <div class="grid">
      <div
        v-for="p in players"
        :key="p.player_id"
        class="card"
        :class="{ dead: !p.is_alive, wolf: p.camp === 'evil' && p.is_alive }"
      >
        <div class="card-id">#{{ p.player_id + 1 }}</div>
        <div class="card-name">{{ p.name }}</div>
        <div class="card-role" :class="p.camp">{{ p.role_name }}</div>
        <div class="card-camp">
          {{ p.camp === 'good' ? '😇 好人' : '🐺 狼人' }}
        </div>
        <div v-if="!p.is_alive" class="dead-tag">💀 死亡</div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({ players: Array })
const aliveCount = computed(() => props.players?.filter(p => p.is_alive).length ?? 0)
</script>

<style scoped>
.panel h3 { margin-bottom: 12px; font-size: 1rem; color: #f0c040; }
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 12px;
}
.card {
  background: #1a2a3a;
  border-radius: 10px;
  padding: 16px;
  text-align: center;
  border: 2px solid #2a3a4a;
  transition: all 0.2s;
  position: relative;
}
.card.wolf { border-color: #8a3a3a; background: #2a1a1a; }
.card.dead {
  opacity: 0.5;
  border-color: #333;
  background: #111;
}
.card-id { font-size: 0.75rem; color: #668; }
.card-name { font-size: 1rem; font-weight: 700; margin: 4px 0; }
.card-role {
  font-size: 0.85rem;
  padding: 2px 8px;
  border-radius: 10px;
  display: inline-block;
  margin: 4px 0;
}
.card-role.good { background: #1a3a5a; color: #80b0e0; }
.card-role.evil { background: #3a1a1a; color: #e08080; }
.card-camp { font-size: 0.8rem; color: #8899aa; margin-top: 4px; }
.dead-tag {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 1.5rem;
  font-weight: 700;
  color: #ff4444;
  text-shadow: 0 0 10px rgba(255,0,0,0.5);
}
</style>
