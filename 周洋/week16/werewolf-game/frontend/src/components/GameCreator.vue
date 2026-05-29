<template>
  <div class="creator">
    <h3>🎮 创建新游戏</h3>
    <div class="form-group">
      <label>角色配置</label>
      <select v-model="selectedConfig">
        <option v-for="cfg in configs" :key="cfg.name" :value="cfg.name">
          {{ cfg.description }}
        </option>
      </select>
    </div>
    <div class="form-group">
      <label class="checkbox-label">
        <input type="checkbox" v-model="shuffle" />
        随机分配角色
      </label>
    </div>
    <div class="form-group">
      <label class="checkbox-label">
        <input type="checkbox" v-model="useLlm" />
        🧠 大模型驱动AI玩家
      </label>
    </div>
    <button class="btn-create" @click="doCreate">创建游戏</button>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({ configs: Array })
const emit = defineEmits(['create'])

const selectedConfig = ref('standard_6')
const shuffle = ref(true)
const useLlm = ref(true)

function doCreate() {
  emit('create', { config_name: selectedConfig.value, shuffle: shuffle.value, use_llm: useLlm.value })
}
</script>

<style scoped>
.creator {
  background: #1a2a3a;
  border-radius: 10px;
  padding: 20px;
  border: 1px solid #2a3a4a;
}
.creator h3 { margin-bottom: 16px; font-size: 1rem; color: #f0c040; }
.form-group { margin-bottom: 12px; }
.form-group label { display: block; font-size: 0.85rem; color: #8899aa; margin-bottom: 4px; }
.form-group select {
  width: 100%;
  padding: 8px;
  border-radius: 6px;
  border: 1px solid #3a4a5a;
  background: #0d1b2a;
  color: #e0e0e0;
  font-size: 0.85rem;
}
.checkbox-label {
  display: flex !important;
  align-items: center;
  gap: 8px;
  cursor: pointer;
}
.checkbox-label input { width: 16px; height: 16px; }
.btn-create {
  width: 100%;
  padding: 10px;
  border: none;
  border-radius: 8px;
  background: linear-gradient(135deg, #f0c040, #d4a030);
  color: #0d1b2a;
  font-weight: 700;
  font-size: 1rem;
  cursor: pointer;
  transition: transform 0.1s;
}
.btn-create:hover { transform: scale(1.02); }
.btn-create:active { transform: scale(0.98); }
</style>
