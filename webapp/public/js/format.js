/* Presentation helpers. ASCII/typographic formatting only — no colour or style
   decisions live here; risk tones come from CSS classes in base.css. */

export function percent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

export function decimal(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return Number(value).toFixed(digits);
}

export function compact(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  const number = Number(value);
  if (Math.abs(number) >= 1000) return `${(number / 1000).toFixed(1)}k`;
  return String(number);
}

export function seconds(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return `${Number(value).toFixed(digits)}s`;
}

export function integer(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return Math.round(Number(value)).toLocaleString();
}

export function riskTone(risk) {
  if (risk === 'High Risk') return 'tone-high';
  if (risk === 'Moderate Risk') return 'tone-moderate';
  return 'tone-low';
}

export function riskKeyword(risk) {
  if (risk === 'High Risk') return 'High';
  if (risk === 'Moderate Risk') return 'Moderate';
  return 'Low';
}

export function bandLabel(band) {
  return ['—', 'Low', 'Moderate', 'High'][band] || '—';
}

export function shortLabel(feature) {
  return String(feature).replace(/_/g, ' ');
}

/* "2026-08-31 00:08:34" -> "31 Aug 2026" */
export function trainingDate(value) {
  if (!value) return '—';
  const parsed = new Date(String(value).replace(' ', 'T'));
  if (Number.isNaN(parsed.getTime())) return String(value);
  return parsed.toLocaleDateString(undefined, { day: '2-digit', month: 'short', year: 'numeric' });
}

export function relativeTime(iso) {
  if (!iso) return '—';
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return '—';
  const diff = Math.max(0, Date.now() - then);
  const minutes = Math.floor(diff / 60000);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  return `${Math.floor(hours / 24)} d ago`;
}

/* Scale a numeric range to 0-100 for bars and rings. */
export function toPercent(value, min, max) {
  if (value === null || value === undefined) return 0;
  const span = Number(max) - Number(min);
  if (!span) return 0;
  const ratio = (Number(value) - Number(min)) / span;
  return Math.max(0, Math.min(100, ratio * 100));
}

export function parseHyperparameters(raw) {
  if (!raw) return [];
  const text = String(raw);
  const pairs = text.replace(/^\{|\}$/g, '').split(',');
  return pairs
    .map((pair) => {
      const [key, ...rest] = pair.split(':');
      if (!key || rest.length === 0) return null;
      const label = key.trim().replace(/^classifier__/, '').replace(/_/g, ' ');
      const value = rest.join(':').trim().replace(/^['"]|['"]$/g, '');
      return { label, value };
    })
    .filter(Boolean);
}
