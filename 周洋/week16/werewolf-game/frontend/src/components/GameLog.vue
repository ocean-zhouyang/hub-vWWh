<template>
  <div class="log-container">
    <h3>📜 游戏日志</h3>
    <div class="log-list" ref="logList">
      <div v-if="log.length === 0" class="empty">暂无日志，请创建游戏</div>
      <div
        v-for="(entry, i) in log"
        :key="i"
        class="log-entry"
        :class="entry.type"
      >
        <span class="log-time">{{ entry.time }}</span>
        <span class="log-text">{{ entry.text }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, nextTick } from 'vue'

const props = defineProps({ log: Array })
const logList = ref(null)

watch(
  () => props.log?.length,
  async () => {
    await nextTick()
    if (logList.value) {
      logList.value.scrollTop = logList.value.scrollHeight
    }
  }
)
</script>

<style scoped>
.log-container {
  background: #1a2a3a;
  border-radius: 10px;
  padding: 16px;
  border: 1px solid #2a3a4a;
  height: 100%;
  display: flex;
  flex-direction: column;
}
.log-container h3 { margin-bottom: 12px; font-size: 1rem; color: #f0c040; }
.log-list {
  flex: 1;
  overflow-y: auto;
  font-size: 0.82rem;
  line-height: 1.6;
}
.log-list::-webkit-scrollbar { width: 4px; }
.log-list::-webkit-scrollbar-thumb { background: #3a4a5a; border-radius: 2px; }
.empty { color: #556; text-align: center; padding: 40px 0; }
.log-entry { padding: 2px 0; border-bottom: 1px solid #1e2e3e; }
.log-time { color: #556; margin-right: 8px; }
.log-entry.system .log-text { color: #f0c040; font-weight: 600; }
.log-entry.action .log-text { color: #80e0a0; }
.log-entry.death .log-text { color: #ff6666; }
.log-entry.info .log-text { color: #80b0e0; }
.log-entry.error .log-text { color: #ff4444; }
</style>
