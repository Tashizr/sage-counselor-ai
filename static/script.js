(() => {
  const chatArea = document.getElementById('chat-area');
  const messagesEl = document.getElementById('messages');
  const input = document.getElementById('text-input');
  const sendBtn = document.getElementById('send-btn');
  const emptyState = document.getElementById('empty-state');
  let nameSet = false, loading = false, lastReq = 0;

  function now() {
    const d = new Date(), h = d.getHours(), m = d.getMinutes();
    return `${h % 12 || 12}:${String(m).padStart(2, '0')} ${h >= 12 ? 'PM' : 'AM'}`;
  }

  function scroll() {
    requestAnimationFrame(() => chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: 'smooth' }));
  }

  document.getElementById('safety-dismiss').addEventListener('click', () => {
    document.getElementById('safety-bar').classList.add('hidden');
  });

  function addMessage(text, role) {
    emptyState.classList.add('hidden');
    const msg = document.createElement('div');
    msg.className = `message ${role}`;
    const row = document.createElement('div');
    row.className = 'message-row';
    const av = document.createElement('div');
    av.className = `avatar ${role === 'bot' ? 'bot-avatar' : 'user-avatar'}`;
    if (role === 'bot') {
      av.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/></svg>';
    } else {
      av.textContent = nameSet ? 'U' : '?';
    }
    row.appendChild(av);
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;
    row.appendChild(bubble);
    msg.appendChild(row);
    const time = document.createElement('div');
    time.className = 'message-time';
    time.textContent = now();
    msg.appendChild(time);
    messagesEl.appendChild(msg);
    scroll();
  }

  function showTyping() {
    const div = document.createElement('div');
    div.className = 'message bot typing';
    div.id = 'typing-indicator';
    const row = document.createElement('div');
    row.className = 'message-row';
    const av = document.createElement('div');
    av.className = 'avatar bot-avatar';
    av.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5"><path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/></svg>';
    row.appendChild(av);
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    for (let i = 0; i < 3; i++) { const d = document.createElement('span'); d.className = 'dot'; bubble.appendChild(d); }
    row.appendChild(bubble);
    div.appendChild(row);
    messagesEl.appendChild(div);
    scroll();
  }

  function removeTyping() { document.getElementById('typing-indicator')?.remove(); }

  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 120) + 'px';
  });

  async function sendMessage(text) {
    if (!text.trim() || loading) return;
    loading = true;
    const reqId = ++lastReq;
    addMessage(text, 'user');
    input.value = '';
    input.style.height = 'auto';
    showTyping();
    try {
      const resp = await fetch('/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: text }) });
      const data = await resp.json();
      if (reqId !== lastReq) return;
      removeTyping();
      addMessage(data.reply, 'bot');
      if (data.name_set && !nameSet) { nameSet = true; input.placeholder = "Share what's on your mind..."; }
    } catch {
      if (reqId === lastReq) { removeTyping(); addMessage("Sorry, couldn't reach the server. Try again.", 'bot'); }
    }
    loading = false;
  }

  input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(input.value); } });
  sendBtn.addEventListener('click', () => sendMessage(input.value));

  document.querySelectorAll('.suggestion-card').forEach(c => c.addEventListener('click', () => sendMessage(c.dataset.message)));

  document.querySelectorAll('.input-chip').forEach(c => c.addEventListener('click', () => {
    const a = c.dataset.action;
    if (a === 'breathe') openModal('breathing-modal');
    else if (a === 'mood') openModal('mood-modal');
    else if (a === 'crisis') openModal('crisis-modal');
    else if (a === 'ground') openModal('grounding-modal');
  }));

  document.getElementById('export-btn').addEventListener('click', () => {
    let text = 'SAGE Conversation Export\n' + '='.repeat(40) + '\n\n';
    messagesEl.querySelectorAll('.message').forEach(m => {
      const role = m.classList.contains('user') ? 'You' : 'SAGE';
      const time = m.querySelector('.message-time')?.textContent || '';
      text += `${role} (${time}):\n${m.querySelector('.bubble')?.textContent || ''}\n\n`;
    });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([text], { type: 'text/plain' }));
    a.download = `sage-${new Date().toISOString().split('T')[0]}.txt`;
    a.click();
  });

  function openModal(id) { document.getElementById(id)?.classList.add('active'); }
  function closeModal(id) { document.getElementById(id)?.classList.remove('active'); }
  document.querySelectorAll('.modal-close').forEach(b => b.addEventListener('click', () => b.closest('.modal-overlay')?.classList.remove('active')));
  document.querySelectorAll('.modal-overlay').forEach(o => o.addEventListener('click', e => { if (e.target === o) o.classList.remove('active'); }));

  // Breathing
  let breathInterval = null;
  document.getElementById('breathing-start')?.addEventListener('click', function () {
    const circle = document.getElementById('breathing-circle');
    const text = document.getElementById('breathing-text');
    const instr = document.getElementById('breathing-instruction');
    if (breathInterval) { clearInterval(breathInterval); breathInterval = null; circle.classList.remove('inhale', 'exhale'); text.textContent = 'Breathe In'; instr.textContent = 'Follow the circle.'; this.textContent = 'Start'; return; }
    this.textContent = 'Stop';
    let phase = 'inhale', count = 4;
    function tick() {
      if (phase === 'inhale') { circle.classList.remove('exhale'); circle.classList.add('inhale'); text.textContent = 'Breathe In'; instr.textContent = `Hold... ${count}`; if (--count < 0) { phase = 'hold'; count = 7; } }
      else if (phase === 'hold') { text.textContent = 'Hold'; instr.textContent = `Hold for ${count}s`; if (--count < 0) { phase = 'exhale'; count = 8; } }
      else { circle.classList.remove('inhale'); circle.classList.add('exhale'); text.textContent = 'Breathe Out'; instr.textContent = `Breathe out... ${count}`; if (--count < 0) { phase = 'inhale'; count = 4; } }
    }
    tick();
    breathInterval = setInterval(tick, 1000);
  });

  // Mood
  document.querySelectorAll('.mood-option').forEach(o => o.addEventListener('click', () => {
    o.classList.add('selected');
    const msgs = { great: "I'm feeling great today", good: "I'm feeling good", okay: "I'm feeling okay", low: "I'm feeling a bit low", bad: "I'm feeling bad", awful: "I'm feeling awful" };
    setTimeout(() => { closeModal('mood-modal'); sendMessage(msgs[o.dataset.mood] || "I wanted to check in"); }, 300);
  }));

  // Grounding
  const gSteps = [
    { n: 5, p: 'Name <strong>5 things</strong> you can <strong>see</strong>.' },
    { n: 4, p: 'Name <strong>4 things</strong> you can <strong>touch</strong>.' },
    { n: 3, p: 'Name <strong>3 things</strong> you can <strong>hear</strong>.' },
    { n: 2, p: 'Name <strong>2 things</strong> you can <strong>smell</strong>.' },
    { n: 1, p: 'Name <strong>1 thing</strong> you can <strong>taste</strong>.' },
  ];
  let gStep = 0;
  document.getElementById('grounding-next')?.addEventListener('click', () => {
    gStep++;
    if (gStep >= gSteps.length) { closeModal('grounding-modal'); gStep = 0; document.getElementById('grounding-progress-bar').style.width = '0%'; sendMessage("I completed the grounding exercise."); return; }
    const s = gSteps[gStep];
    document.getElementById('grounding-number').textContent = s.n;
    document.getElementById('grounding-prompt').innerHTML = s.p;
    document.getElementById('grounding-progress-bar').style.width = `${(gStep + 1) / gSteps.length * 100}%`;
  });

  document.addEventListener('keydown', e => { if (e.key === 'Escape') document.querySelectorAll('.modal-overlay.active').forEach(m => m.classList.remove('active')); });
  if ('visualViewport' in window) window.visualViewport.addEventListener('resize', () => setTimeout(() => scroll(), 150));

  // Init
  (async () => {
    showTyping();
    try {
      const resp = await fetch('/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: '' }) });
      const data = await resp.json();
      removeTyping();
      addMessage(data.reply, 'bot');
    } catch { removeTyping(); }
  })();
  input.focus();
})();
