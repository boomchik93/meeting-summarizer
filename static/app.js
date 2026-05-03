'use strict';

// ссылки на DOM элементы
const dropZone    = document.getElementById('drop-zone');
const fileInput   = document.getElementById('file-input');
const fileChip    = document.getElementById('file-chip');
const chipName    = document.getElementById('chip-name');
const chipRemove  = document.getElementById('chip-remove');
const runBtn      = document.getElementById('run-btn');
const progress    = document.getElementById('progress');
const progressTxt = document.getElementById('progress-text');
const errorBox    = document.getElementById('error-box');
const results     = document.getElementById('results');
const metaRow     = document.getElementById('meta-row');
const speakerList = document.getElementById('speaker-list');
const speakersCard= document.getElementById('speakers-card');
const segBody     = document.getElementById('seg-body');
const segCount    = document.getElementById('seg-count');
const jsonBlock   = document.getElementById('json-block');
const dlBtn       = document.getElementById('dl-btn');

const badgeWhisper = document.getElementById('badge-whisper');
const badgeDiar    = document.getElementById('badge-diar');
const badgeModel   = document.getElementById('badge-model');
const badgeLlm     = document.getElementById('badge-llm');

const summaryCard     = document.getElementById('summary-card');
const summaryBtn      = document.getElementById('summary-btn');
const summaryProgress = document.getElementById('summary-progress');
const summaryError    = document.getElementById('summary-error');
const summaryContent  = document.getElementById('summary-content');

let selectedFile = null;
let lastResult   = null;

// индекс цветов спикеров
const spColors = ['sp-0','sp-1','sp-2','sp-3','sp-4','sp-5'];
const spMap    = {};
let   spIdx    = 0;

function spClass(label) {
  if (!(label in spMap)) spMap[label] = spColors[spIdx++ % spColors.length];
  return spMap[label];
}

// проверка здоровья
async function checkHealth() {
  try {
    const r = await fetch('/api/health');
    const d = await r.json();

    setBadge(badgeWhisper, d.whisper,     'Whisper');
    setBadge(badgeDiar,    d.diarization, 'Диаризация');
    badgeModel.querySelector('.dot').style.background = 'var(--accent2)';
    badgeModel.querySelector('.dot').style.opacity = '1';
    badgeModel.lastChild.textContent = ` Модель: ${d.model}`;
    badgeModel.classList.add('ok');
    setBadge(badgeLlm, d.llm, `LLM: ${d.llm_model ?? '—'}`);
  } catch {
    setBadge(badgeWhisper, false, 'Whisper');
    setBadge(badgeDiar,    false, 'Диаризация');
    setBadge(badgeLlm,     false, 'LLM: —');
  }
}

function setBadge(el, ok, label) {
  el.querySelector('.dot').style.background = ok ? 'var(--green)' : 'var(--red)';
  el.lastChild.textContent = ` ${label}`;
  el.classList.toggle('ok',  ok);
  el.classList.toggle('err', !ok);
}

checkHealth();

// обработка файлов
const VALID_EXT = ['.mp3','.wav','.m4a','.ogg','.flac','.webm'];

function setFile(file) {
  if (!file) return;
  const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
  if (!VALID_EXT.includes(ext)) {
    showError(`Неподдерживаемый формат: ${ext}`);
    return;
  }
  selectedFile = file;
  chipName.textContent = file.name;
  fileChip.classList.add('show');
  runBtn.disabled = false;
  hideError();
  results.classList.remove('show');
}

function clearFile() {
  selectedFile = null;
  fileInput.value = '';
  fileChip.classList.remove('show');
  runBtn.disabled = true;
  results.classList.remove('show');
}

fileInput.addEventListener('change', e => setFile(e.target.files[0]));
chipRemove.addEventListener('click', clearFile);

dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('over'); });
dropZone.addEventListener('dragleave', ()  => dropZone.classList.remove('over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('over');
  setFile(e.dataTransfer.files[0]);
});

// транскрибация
runBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  runBtn.disabled = true;
  showProgress('Загрузка файла…');
  hideError();
  results.classList.remove('show');

  const fd = new FormData();
  fd.append('file', selectedFile);

  try {
    showProgress('Транскрибация и диаризация…');
    const resp = await fetch('/api/transcribe', { method: 'POST', body: fd });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || 'Ошибка сервера');
    }

    lastResult = await resp.json();
    renderResults(lastResult);
    
    // Автоматически показываем саммари если оно есть
    summaryCard.style.display = '';
    if (lastResult.summary && !lastResult.summary.error) {
      renderSummary(lastResult.summary);
      summaryBtn.textContent = '🔄 Обновить пересказ';
      summaryBtn.disabled = false;
    } else {
      summaryContent.innerHTML = '';
      summaryError.style.display = 'none';
      summaryBtn.disabled = false;
      summaryBtn.textContent = '✨ Составить пересказ';
    }

  } catch (e) {
    showError(e.message);
  } finally {
    hideProgress();
    runBtn.disabled = false;
  }
});

// рендер результатов
function renderResults(data) {
  // сбрасываем карту цветов спикеров
  Object.keys(spMap).forEach(k => delete spMap[k]);
  spIdx = 0;

  // мета
  metaRow.innerHTML = meta('Файл', data.filename)
    + meta('Язык', data.language?.toUpperCase() ?? '—')
    + meta('Модель', data.model)
    + meta('Спикеров', Object.keys(data.speakers ?? {}).length)
    + meta('Сегментов', data.segments?.length ?? 0)
    + meta('Диаризация', data.diarization_enabled ? '✅ вкл' : '❌ выкл');

  // спикеры
  const spEntries = Object.entries(data.speakers ?? {});
  if (spEntries.length > 0) {
    speakersCard.style.display = '';
    speakerList.innerHTML = spEntries.map(([sp, txt]) =>
      `<div class="speaker-item ${spClass(sp)}">
         <div class="sp-label">${sp}</div>
         <div>${escHtml(txt)}</div>
       </div>`
    ).join('');
  } else {
    speakersCard.style.display = 'none';
  }

  // сегменты
  segCount.textContent = `(${data.segments?.length ?? 0})`;
  segBody.innerHTML = (data.segments ?? []).map(s =>
    `<tr>
       <td class="time">${fmt(s.start)}</td>
       <td class="time">${fmt(s.end)}</td>
       <td class="sp">${escHtml(s.speaker ?? '—')}</td>
       <td>${escHtml(s.text)}</td>
     </tr>`
  ).join('');

  // json
  jsonBlock.textContent = JSON.stringify(data, null, 2);

  results.classList.add('show');
}

function meta(label, value) {
  return `<div class="meta-item">
    <span class="label">${label}</span>
    <span class="value">${escHtml(String(value))}</span>
  </div>`;
}

function fmt(sec) {
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(1).padStart(4, '0');
  return `${m}:${s}`;
}

function escHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// пересказ
summaryBtn.addEventListener('click', async () => {
  if (!lastResult) return;

  summaryBtn.disabled = true;
  summaryBtn.textContent = '⏳ Генерация…';
  summaryProgress.style.display = 'block';
  summaryError.style.display = 'none';
  summaryContent.innerHTML = '';

  try {
    const resp = await fetch('/api/summarize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        segments: lastResult.segments ?? [],
        speakers: lastResult.speakers ?? {},
      }),
    });

    const data = await resp.json();

    if (!resp.ok) {
      throw new Error(data.detail || 'Ошибка сервера');
    }
    if (data.error) {
      throw new Error(data.error);
    }

    renderSummary(data.topics ?? []);
    summaryBtn.textContent = '🔄 Обновить пересказ';
  } catch (e) {
    summaryError.textContent = '⚠ ' + e.message;
    summaryError.style.display = 'block';
    summaryBtn.textContent = '✨ Составить пересказ';
  } finally {
    summaryProgress.style.display = 'none';
    summaryBtn.disabled = false;
  }
});

function renderSummary(summaryData) {
  // Если передан старый формат (массив topics)
  if (Array.isArray(summaryData)) {
    if (!summaryData.length) {
      summaryContent.innerHTML = '<p style="color:var(--muted);font-size:14px">Не удалось выделить темы.</p>';
      return;
    }
    summaryContent.innerHTML = `<div class="summary-topics">${
      summaryData.map((t, i) => `
        <div class="summary-topic-block">
          <div class="summary-topic-title">
            <span class="topic-num">Тема ${i + 1}</span>${escHtml(t.title)}
          </div>
          <ul class="summary-points">
            ${(t.points ?? []).map(p => `<li>${escHtml(p)}</li>`).join('')}
          </ul>
        </div>`
      ).join('')
    }</div>`;
    return;
  }

  // Новый формат с полной структурой
  let html = '';

  // Краткое резюме
  if (summaryData.summary) {
    html += `<div class="summary-section">
      <h3 class="summary-section-title">📋 Краткое резюме</h3>
      <p class="summary-text">${escHtml(summaryData.summary)}</p>
    </div>`;
  }

  // Темы
  if (summaryData.topics && summaryData.topics.length > 0) {
    html += `<div class="summary-section">
      <h3 class="summary-section-title">🎯 Ключевые темы</h3>
      <div class="summary-topics">${
        summaryData.topics.map((t, i) => {
          const categoryEmoji = {
            'technical': '⚙️',
            'business': '💼',
            'organizational': '🏢',
            'decision': '✅',
            'problem': '⚠️'
          }[t.category] || '📌';
          
          return `
            <div class="summary-topic-block">
              <div class="summary-topic-title">
                <span class="topic-num">${categoryEmoji} Тема ${i + 1}</span>${escHtml(t.title)}
              </div>
              <ul class="summary-points">
                ${(t.points ?? []).map(p => `<li>${escHtml(p)}</li>`).join('')}
              </ul>
            </div>`;
        }).join('')
      }</div>
    </div>`;
  }

  // Решения
  if (summaryData.decisions && summaryData.decisions.length > 0) {
    html += `<div class="summary-section">
      <h3 class="summary-section-title">✅ Принятые решения</h3>
      <div class="summary-list">${
        summaryData.decisions.map(d => `
          <div class="summary-item">
            <div class="summary-item-title">${escHtml(d.decision)}</div>
            ${d.context ? `<div class="summary-item-text">${escHtml(d.context)}</div>` : ''}
            ${d.responsible ? `<div class="summary-item-meta">Ответственный: ${escHtml(d.responsible)}</div>` : ''}
          </div>`
        ).join('')
      }</div>
    </div>`;
  }

  // Задачи
  if (summaryData.action_items && summaryData.action_items.length > 0) {
    html += `<div class="summary-section">
      <h3 class="summary-section-title">📝 Задачи (Action Items)</h3>
      <div class="summary-list">${
        summaryData.action_items.map(a => `
          <div class="summary-item">
            <div class="summary-item-title">${escHtml(a.action)}</div>
            ${a.responsible ? `<div class="summary-item-meta">Ответственный: ${escHtml(a.responsible)}</div>` : ''}
            ${a.deadline ? `<div class="summary-item-meta">Срок: ${escHtml(a.deadline)}</div>` : ''}
          </div>`
        ).join('')
      }</div>
    </div>`;
  }

  // Риски
  if (summaryData.risks && summaryData.risks.length > 0) {
    html += `<div class="summary-section">
      <h3 class="summary-section-title">⚠️ Риски и проблемы</h3>
      <div class="summary-list">${
        summaryData.risks.map(r => `
          <div class="summary-item risk-item">
            <div class="summary-item-title">${escHtml(r.risk)}</div>
            ${r.impact ? `<div class="summary-item-text">${escHtml(r.impact)}</div>` : ''}
          </div>`
        ).join('')
      }</div>
    </div>`;
  }

  // Ключевые моменты
  if (summaryData.key_points && summaryData.key_points.length > 0) {
    html += `<div class="summary-section">
      <h3 class="summary-section-title">💡 Ключевые моменты</h3>
      <ul class="summary-key-points">
        ${summaryData.key_points.map(p => `<li>${escHtml(p)}</li>`).join('')}
      </ul>
    </div>`;
  }

  if (!html) {
    summaryContent.innerHTML = '<p style="color:var(--muted);font-size:14px">Не удалось сгенерировать саммари.</p>';
  } else {
    summaryContent.innerHTML = html;
  }
}

// скачивание
dlBtn.addEventListener('click', () => {
  if (!lastResult) return;
  const blob = new Blob([JSON.stringify(lastResult, null, 2)], { type: 'application/json' });
  const url  = URL.createObjectURL(blob);
  const a    = Object.assign(document.createElement('a'), {
    href: url,
    download: `transcription_${Date.now()}.json`,
  });
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
});

// хелперы
function showProgress(msg) {
  progressTxt.textContent = msg;
  progress.classList.add('show');
}
function hideProgress() { progress.classList.remove('show'); }
function showError(msg) {
  errorBox.textContent = '⚠ ' + msg;
  errorBox.classList.add('show');
}
function hideError() { errorBox.classList.remove('show'); }
