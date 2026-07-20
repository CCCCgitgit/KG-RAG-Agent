<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { checkHealth, sendMessage } from './api'
import type { Message } from './types'

const storageKey = 'kg-rag-agent-ui-settings'
const messagesKey = 'kg-rag-agent-ui-messages'

const makeId = (prefix: string): string => {
  const random = globalThis.crypto?.randomUUID?.() || Math.random().toString(36).slice(2)
  return `${prefix}_${random.replace(/-/g, '').slice(0, 16)}`
}

const userId = ref('demo_user')
const projectId = ref('demo_project')
const sessionId = ref(makeId('session'))
const query = ref('')
const sending = ref(false)
const online = ref(false)
const settingsOpen = ref(false)
const messages = ref<Message[]>([])
const chatBody = ref<HTMLElement | null>(null)

const statusText = computed(() => (online.value ? '服务正常' : '服务离线'))
const canSend = computed(() => query.value.trim().length > 0 && !sending.value)

const loadLocalState = (): void => {
  try {
    const settings = JSON.parse(localStorage.getItem(storageKey) || '{}') as Record<string, string>
    userId.value = settings.userId || userId.value
    projectId.value = settings.projectId || projectId.value
    sessionId.value = settings.sessionId || sessionId.value

    const savedMessages = JSON.parse(localStorage.getItem(messagesKey) || '[]') as Message[]
    if (Array.isArray(savedMessages)) messages.value = savedMessages.slice(-40)
  } catch {
    localStorage.removeItem(storageKey)
    localStorage.removeItem(messagesKey)
  }
}

watch([userId, projectId, sessionId], () => {
  localStorage.setItem(
    storageKey,
    JSON.stringify({ userId: userId.value, projectId: projectId.value, sessionId: sessionId.value }),
  )
})

watch(
  messages,
  (value) => localStorage.setItem(messagesKey, JSON.stringify(value.slice(-40))),
  { deep: true },
)

const scrollToBottom = async (): Promise<void> => {
  await nextTick()
  if (chatBody.value) chatBody.value.scrollTop = chatBody.value.scrollHeight
}

const refreshHealth = async (): Promise<void> => {
  online.value = await checkHealth()
}

const send = async (): Promise<void> => {
  const text = query.value.trim()
  if (!text || sending.value) return

  messages.value.push({ id: makeId('msg'), role: 'user', content: text })
  query.value = ''
  sending.value = true
  await scrollToBottom()

  try {
    const response = await sendMessage({
      query: text,
      userId: userId.value.trim(),
      projectId: projectId.value.trim(),
      sessionId: sessionId.value.trim(),
    })
    messages.value.push({
      id: makeId('msg'),
      role: response.has_error ? 'error' : 'assistant',
      content: response.answer || response.error_message || '未返回有效回答。',
      response,
    })
    online.value = true
  } catch (error) {
    messages.value.push({
      id: makeId('msg'),
      role: 'error',
      content: error instanceof Error ? error.message : '请求失败。',
    })
    online.value = false
  } finally {
    sending.value = false
    await scrollToBottom()
  }
}

const handleKeydown = (event: KeyboardEvent): void => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    void send()
  }
}

const newSession = (): void => {
  sessionId.value = makeId('session')
  messages.value = []
}

const clearMessages = (): void => {
  messages.value = []
}

onMounted(() => {
  loadLocalState()
  void refreshHealth()
  window.setInterval(() => void refreshHealth(), 30_000)
})
</script>

<template>
  <div class="app-shell">
    <header class="topbar">
      <div class="brand">
        <div class="brand-mark">KG</div>
        <div>
          <h1>KG-RAG Agent</h1>
          <p>知识图谱增强问答</p>
        </div>
      </div>

      <div class="topbar-actions">
        <button class="status-pill" type="button" @click="refreshHealth">
          <span class="status-dot" :class="{ online }"></span>
          {{ statusText }}
        </button>
        <button class="ghost-button" type="button" @click="settingsOpen = !settingsOpen">
          设置
        </button>
      </div>
    </header>

    <main class="workspace">
      <aside class="sidebar" :class="{ open: settingsOpen }">
        <div class="sidebar-heading">
          <div>
            <strong>会话设置</strong>
            <span>用于 Memory 隔离</span>
          </div>
          <button class="close-button" type="button" @click="settingsOpen = false">×</button>
        </div>

        <label>
          <span>User ID</span>
          <input v-model="userId" autocomplete="off" />
        </label>
        <label>
          <span>Project ID</span>
          <input v-model="projectId" autocomplete="off" />
        </label>
        <label>
          <span>Session ID</span>
          <input v-model="sessionId" autocomplete="off" />
        </label>

        <button class="primary-button full" type="button" @click="newSession">新建会话</button>
        <button class="secondary-button full" type="button" @click="clearMessages">清空消息</button>

        <div class="sidebar-note">
          前端只保存最近 40 条展示消息。长期 Memory 由后端控制。
        </div>
      </aside>

      <section class="chat-panel">
        <div ref="chatBody" class="chat-body">
          <div v-if="messages.length === 0" class="empty-state">
            <div class="empty-icon">◎</div>
            <h2>开始一次 KG-RAG 问答</h2>
            <p>输入实体关系、路径、邻居或一般问题。回答会展示路由、引用和 Memory 状态。</p>
            <div class="examples">
              <button type="button" @click="query = 'UNITED STATES 与 CHINA 之间有什么关系？'">
                查询实体关系
              </button>
              <button type="button" @click="query = '请查找两个实体之间的多跳路径'">
                查询多跳路径
              </button>
            </div>
          </div>

          <article
            v-for="message in messages"
            :key="message.id"
            class="message-row"
            :class="message.role"
          >
            <div class="avatar">{{ message.role === 'user' ? '你' : message.role === 'error' ? '!' : 'AI' }}</div>
            <div class="message-content">
              <div class="bubble">{{ message.content }}</div>

              <details v-if="message.response" class="details-card">
                <summary>查看执行信息</summary>
                <div class="detail-grid">
                  <div><span>Route</span><strong>{{ message.response.route || '-' }}</strong></div>
                  <div><span>Answerability</span><strong>{{ message.response.answerability || '-' }}</strong></div>
                  <div><span>Score</span><strong>{{ message.response.semantic_score?.toFixed?.(3) ?? '-' }}</strong></div>
                  <div><span>Request</span><strong>{{ message.response.request_id || '-' }}</strong></div>
                </div>

                <div v-if="message.response.memory_status" class="memory-line">
                  <strong>Memory</strong>
                  <span>读取 {{ message.response.memory_status.retrieved_memory_count }}</span>
                  <span>近期消息 {{ message.response.memory_status.recent_message_count }}</span>
                  <span>写入 {{ message.response.memory_status.written_count }}</span>
                </div>

                <div v-if="message.response.citations?.length" class="citation-list">
                  <strong>引用</strong>
                  <div
                    v-for="(citation, index) in message.response.citations"
                    :key="String(citation.evidence_id || index)"
                    class="citation-item"
                  >
                    <span>#{{ index + 1 }}</span>
                    <p>{{ citation.text || citation.source || JSON.stringify(citation) }}</p>
                  </div>
                </div>

                <ul v-if="message.response.warnings?.length" class="warning-list">
                  <li v-for="warning in message.response.warnings" :key="warning">{{ warning }}</li>
                </ul>
              </details>
            </div>
          </article>

          <article v-if="sending" class="message-row assistant">
            <div class="avatar">AI</div>
            <div class="message-content"><div class="bubble typing">正在思考<span></span></div></div>
          </article>
        </div>

        <div class="composer-wrap">
          <div class="composer">
            <textarea
              v-model="query"
              rows="1"
              placeholder="输入问题，Enter 发送，Shift + Enter 换行"
              @keydown="handleKeydown"
            ></textarea>
            <button class="send-button" type="button" :disabled="!canSend" @click="send">
              {{ sending ? '发送中' : '发送' }}
            </button>
          </div>
          <p class="composer-hint">回答由后端 KG-RAG Agent 生成，请根据 Citation 核验关键信息。</p>
        </div>
      </section>
    </main>

    <div v-if="settingsOpen" class="overlay" @click="settingsOpen = false"></div>
  </div>
</template>
