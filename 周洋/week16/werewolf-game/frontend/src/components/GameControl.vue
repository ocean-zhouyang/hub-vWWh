<template>
  <div class="control">
    <h3>🎮 游戏控制</h3>
    <div class="btn-group">
      <button class="btn btn-step" :disabled="gameOver" @click="$emit('step')">
        ⏭ 单步执行
      </button>
      <button
        class="btn"
        :class="autoRunning ? 'btn-pause' : 'btn-auto'"
        :disabled="gameOver"
        @click="$emit('auto')"
      >
        {{ autoRunning ? '⏸ 暂停' : '▶ 自动继续' }}
      </button>
      <button class="btn btn-reset" @click="$emit('reset')">
        🔄 重新开始
      </button>
    </div>
    <div class="status">
      <span>当前阶段: <strong>{{ currentPhase }}</strong></span>
      <span v-if="gameOver" class="over">游戏已结束</span>
    </div>
  </div>
</template>

<script setup>
defineProps({
  phase: String,
  gameOver: Boolean,
  autoRunning: Boolean,
})
defineEmits(['step', 'auto', 'reset'])
</script>

<style scoped>
.control {
  background: #1a2a3a;
  border-radius: 10px;
  padding: 20px;
  border: 1px solid #2a3a4a;
}
.control h3 { margin-bottom: 12px; font-size: 1rem; color: #f0c040; }
.btn-group { display: flex; flex-direction: column; gap: 8px; }
.btn {
  padding: 10px;
  border: none;
  border-radius: 8px;
  font-size: 0.9rem;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.1s;
}
.btn:active { transform: scale(0.97); }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-step { background: #2a5a8a; color: #fff; }
.btn-auto { background: #2a7a3a; color: #fff; }
.btn-pause { background: #8a5a2a; color: #fff; }
.btn-reset { background: #5a2a2a; color: #ff8888; }
.status { margin-top: 12px; font-size: 0.85rem; color: #8899aa; }
.status .over { color: #ff6666; font-weight: 700; }
</style>
