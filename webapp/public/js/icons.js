/* Inline SVG icon set — 24x24, currentColor, no emoji anywhere in the UI. */

const PATHS = {
  arrowRight: '<path d="M4 12h15M13 6l6 6-6 6"/>',
  arrowUpRight: '<path d="M7 17 17 7M9 7h8v8"/>',
  chevronDown: '<path d="M6 9l6 6 6-6"/>',
  check: '<path d="M4 12.5 9.5 18 20 6.5"/>',
  star: '<path d="M12 3.5l2.7 5.6 6.1.9-4.4 4.3 1 6.1-5.4-2.9-5.4 2.9 1-6.1L3.2 10l6.1-.9z"/>',
  pulse: '<path d="M3 12.5h4l2.5-6 4 12 2.5-6h5"/>',
  sliders: '<path d="M5 20V13M5 9V4M12 20v-9M12 7V4M19 20v-5M19 11V4"/><path d="M3 13h4M10 11h4M17 15h4"/>',
  chart: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
  layers: '<path d="M12 3 3 8l9 5 9-5-9-5Z"/><path d="m3 13 9 5 9-5"/>',
  cpu: '<rect x="7" y="7" width="10" height="10" rx="2"/><path d="M4 10V8a4 4 0 0 1 4-4h2M20 14v2a4 4 0 0 1-4 4h-2M2 12h3M19 12h3M12 2v3M12 19v3"/>',
  settings: '<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M2 12h3M19 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1"/>',
  user: '<circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 3.6-6 8-6s8 2 8 6"/>',
  bell: '<path d="M6 9a6 6 0 0 1 12 0c0 5 2 6 2 6H4s2-1 2-6Z"/><path d="M10 20a2 2 0 0 0 4 0"/>',
  lock: '<rect x="4" y="10" width="16" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/>',
  shield: '<path d="M12 3l8 3v6c0 5-3.4 8.4-8 9.9C7.4 20.4 4 17 4 12V6l8-3Z"/>',
  shieldCheck: '<path d="M12 3l8 3v6c0 5-3.4 8.4-8 9.9C7.4 20.4 4 17 4 12V6l8-3Z"/><path d="M8.5 12.2 11 14.7l4.6-4.7"/>',
  alert: '<path d="M12 4 2.5 20h19L12 4Z"/><path d="M12 10v4.5M12 17.5h.01"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.6h.01"/>',
  target: '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3.2"/>',
  database: '<ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
  file: '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5Z"/><path d="M14 3v5h5M9 13h6M9 17h6"/>',
  spark: '<path d="M12 3v5M12 16v5M3 12h5M16 12h5M6.5 6.5l3 3M14.5 14.5l3 3M17.5 6.5l-3 3M9.5 14.5l-3 3"/>',
  refresh: '<path d="M20 11a8 8 0 1 0-2.3 6.3"/><path d="M20 5v6h-6"/>',
  play: '<path d="M7 4.5 19 12 7 19.5v-15Z"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  x: '<path d="M6 6l12 12M18 6 6 18"/>',
  globe: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c2.5 3 2.5 15 0 18M12 3c-2.5 3-2.5 15 0 18"/>',
  link: '<path d="M10 13a5 5 0 0 0 7 0l2-2a5 5 0 0 0-7-7l-1 1"/><path d="M14 11a5 5 0 0 0-7 0l-2 2a5 5 0 0 0 7 7l1-1"/>',
};

const FILLED = new Set(['star', 'play', 'shield', 'shieldCheck']);

export function icon(name, className = 'icon') {
  const body = PATHS[name] || PATHS.info;
  const filled = name === 'star' || name === 'play';
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('class', className);
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('fill', filled ? 'currentColor' : 'none');
  svg.setAttribute('stroke', filled ? 'none' : 'currentColor');
  svg.setAttribute('stroke-width', '1.6');
  svg.setAttribute('stroke-linecap', 'round');
  svg.setAttribute('stroke-linejoin', 'round');
  svg.innerHTML = body;
  return svg;
}

export function hasIcon(name) {
  return Object.prototype.hasOwnProperty.call(PATHS, name);
}

export const ICON_NAMES = Object.keys(PATHS);
export const FILLED_ICONS = Array.from(FILLED);

/* The Apple and Google marks are brand assets: their shapes and colours are
   fixed by the vendors and deliberately ignore the app theme. */
export function appleMark(className = 'icon') {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('class', className);
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('fill', 'currentColor');
  svg.innerHTML =
    '<path d="M17.05 12.54c.02-2.2 1.79-3.26 1.87-3.31-1.02-1.49-2.6-1.7-3.16-1.72-1.34-.14-2.62.79-3.3.79-.68 0-1.73-.77-2.85-.75-1.47.02-2.82.85-3.58 2.17-1.53 2.66-.39 6.6 1.1 8.76.73 1.06 1.6 2.25 2.74 2.2 1.1-.04 1.52-.71 2.85-.71 1.33 0 1.7.71 2.86.69 1.18-.02 1.93-1.08 2.65-2.14.83-1.22 1.18-2.4 1.2-2.46-.03-.01-2.3-.88-2.33-3.52ZM15.03 5.7c.6-.73 1-1.74.89-2.75-.86.04-1.9.57-2.52 1.29-.55.64-1.04 1.67-.91 2.66.96.07 1.94-.49 2.54-1.2Z"/>';
  return svg;
}

export function googleMark(className = 'icon') {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('class', className);
  svg.setAttribute('aria-hidden', 'true');
  svg.innerHTML =
    '<path fill="#4285F4" d="M23.5 12.3c0-.9-.1-1.5-.2-2.2H12v4.1h6.5c-.1 1.1-.8 2.7-2.3 3.8l3.5 2.7c2.1-1.9 3.8-4.8 3.8-8.4Z"/>' +
    '<path fill="#34A853" d="M12 24c3.2 0 5.9-1.1 7.7-2.9l-3.6-2.8c-1 .7-2.3 1.2-4.1 1.2-3.1 0-5.8-2.1-6.7-4.9l-3.7 2.9C3.4 21.3 7.4 24 12 24Z"/>' +
    '<path fill="#FBBC05" d="M5.3 14.6c-.2-.7-.4-1.5-.4-2.6 0-.9.2-1.8.4-2.6L1.6 6.5C.6 8.2 0 10 0 12s.6 3.8 1.6 5.5l3.7-2.9Z"/>' +
    '<path fill="#EA4335" d="M12 4.7c2.2 0 3.7.9 4.5 1.7l3.3-3.2C17.9 1.2 15.2 0 12 0 7.4 0 3.4 2.7 1.6 6.5l3.7 2.9C6.2 6.7 8.9 4.7 12 4.7Z"/>';
  return svg;
}
