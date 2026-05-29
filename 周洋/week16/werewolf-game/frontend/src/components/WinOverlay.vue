<template>
  <div class="overlay">
    <div class="modal">
      <h2>🏁 游戏结束</h2>
      <div class="winner" :class="winner">
        {{ winner === 'good' ? '😇' : '🐺' }}
        {{ winner === 'good' ? '好人阵营获胜！' : '狼人阵营获胜！' }}
      </div>
      <p class="sub">感谢观看这场 AI 狼人杀对局</p>
      <button class="btn-close" @click="$emit('close')">返回大厅</button>
    </div>
  </div>
</template>

<script setup>
defineProps({ winner: String })
defineEmits(['close'])
</script>

<style scoped>
.overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
  animation: fadeIn 0.5s;
}
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}
.modal {
  background: linear-gradient(135deg, #1a2a3a, #0d1b2a);
  border-radius: 20px;
  padding: 40px 60px;
  text-align: center;
  border: 2px solid #f0c040;
  animation: scaleIn 0.3s;
  box-shadow: 0 0 60px rgba(240, 192, 64, 0.2);
}
@keyframes scaleIn {
  from { transform: scale(0.8); }
  to { transform: scale(1); }
}
.modal h2 { font-size: 1.8rem; color: #f0c040; margin-bottom: 20px; }
.winner { font-size: 2.4rem; font-weight: 700; margin: 20px 0; }
.winner.good { color: #80e0a0; }
.winner.evil { color: #ff6666; }
.sub { color: #8899aa; margin-bottom: 24px; }
.btn-close {
  padding: 12px 40px;
  border: none;
  border-radius: 10px;
  background: linear-gradient(135deg, #f0c040, #d4a030);
  color: #0d1b2a;
  font-weight: 700;
  font-size: 1rem;
  cursor: pointer;
}
.btn-close:hover { transform: scale(1.05); }
</style>
