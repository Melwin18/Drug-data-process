/* Image treatment. Every photo goes through .img-plate so the theme's
   --tp-img-* treatment and gradient overlay are never skipped. */

import { el } from './dom.js';

export function imgPlate({ src, alt, wide = false, band = false, mirror = false }) {
  const classes = ['img-plate'];
  if (wide) classes.push('img-wide');
  if (band) classes.push('img-band');
  if (mirror) classes.push('img-plate-mirror');
  return el('div', { class: classes.join(' ') }, [
    el('img', { src, alt, loading: 'lazy', decoding: 'async' }),
    el('span', { class: 'img-overlay', 'aria-hidden': 'true' }),
  ]);
}

/* Fallback when no photo exists: --tp-grad block, never a grey box. */
export function gradientPlate({ ratio = 'wide', label }) {
  const block = el('div', {
    class: `grad-block ${ratio === 'wide' ? 'img-wide' : ''}`.trim(),
    role: 'img',
    'aria-label': label || 'Chrome gradient panel',
  });
  if (ratio === 'band') block.classList.add('img-band');
  return block;
}

export function bandBackground(src, alt = '') {
  return el('div', { class: 'band-img' }, [el('img', { src, alt, loading: 'lazy' })]);
}
