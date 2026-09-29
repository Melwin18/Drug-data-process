/* Shared UI primitives. All styling comes from classes in base.css, which in
   turn resolves to var(--tp-*) tokens. */

import { el } from './dom.js';
import { icon } from './icons.js';

export function metricTile({ kicker, value, caption, tone }) {
  return el('article', { class: 'plate tile' }, [
    el('p', { class: 'kicker', text: kicker }),
    el('p', { class: `numeric ${tone || ''}`.trim(), text: value }),
    caption ? el('p', { class: 'small mute', text: caption }) : null,
  ]);
}

export function chip(text, variant = '') {
  return el('span', { class: `chip ${variant}`.trim() }, [text]);
}

export function statusChip(text, tone) {
  return el('span', { class: `chip ${tone}`.trim() }, [
    el('span', { class: 'dot' }),
    text,
  ]);
}

export function barRow({ label, value, display, quiet = false }) {
  const fill = el('div', { class: `fill ${quiet ? 'fill-quiet' : ''}`.trim() });
  fill.style.setProperty('--v', String(Math.max(0, Math.min(100, value))));
  return el('div', { class: 'barrow' }, [
    el('div', { class: 'barrow-head' }, [
      el('span', { text: label }),
      el('span', { class: 'barrow-val', text: display }),
    ]),
    el('div', { class: 'track' }, [fill]),
  ]);
}

export function columnChart({ columns, caption }) {
  return el('div', {}, [
    el(
      'div',
      { class: 'columns', role: 'img', 'aria-label': caption || 'Column chart' },
      columns.map((column) => {
        const bar = el('div', { class: 'column-bar', title: `${column.label}: ${column.display}` });
        bar.style.setProperty('--v', String(Math.max(0, Math.min(100, column.value))));
        return el('div', { class: 'column' }, [
          bar,
          el('span', { class: 'column-label', text: column.label }),
        ]);
      })
    ),
    caption ? el('p', { class: 'tiny mute', text: caption }) : null,
  ]);
}

export function scoreRing({ value, figure, caption, size = '', toneClass = '' }) {
  const ring = el('div', { class: `ring ${size} ${toneClass}`.trim() }, [
    el('div', { class: 'ring-figure' }, [
      figure ? el('span', { class: 'numeric', text: figure }) : null,
      caption ? el('span', { class: 'kicker', text: caption }) : null,
    ]),
  ]);
  ring.style.setProperty('--v', String(Math.max(0, Math.min(100, value))));
  ring.setAttribute('role', 'img');
  ring.setAttribute('aria-label', `${figure || value}${caption ? `, ${caption}` : ''}`);
  return ring;
}

export function definitionList(entries) {
  return el(
    'dl',
    {},
    entries.map(([term, definition]) =>
      el('div', { class: 'kv' }, [
        el('dt', { text: term }),
        el('dd', { text: definition }),
      ])
    )
  );
}

export function checkList(items) {
  return el(
    'ul',
    { class: 'check-list' },
    items.map((item) =>
      el('li', {}, [
        el('span', { class: 'tick' }, [icon('check', 'icon-sm')]),
        el('span', { text: item }),
      ])
    )
  );
}

export function skeleton(lines = 3) {
  return el(
    'div',
    { class: 'sk-stack', 'aria-hidden': 'true' },
    Array.from({ length: lines }, (_, index) =>
      el('div', {
        class: `skeleton sk-line ${
          index === lines - 1 ? 'sk-line-short' : index % 2 ? 'sk-line-mid' : ''
        }`.trim(),
      })
    )
  );
}

export function skeletonTile() {
  return el('div', { class: 'plate tile', 'aria-hidden': 'true' }, [
    el('div', { class: 'skeleton sk-line sk-line-short' }),
    el('div', { class: 'skeleton sk-line' }),
    el('div', { class: 'skeleton sk-block' }),
  ]);
}

export function emptyState({ title, body, actionLabel, onAction, iconName = 'layers' }) {
  return el('div', { class: 'empty-state' }, [
    el('div', { class: 'empty-glyph' }, [icon(iconName, 'icon-lg')]),
    el('h3', { text: title }),
    body ? el('p', { class: 'lead', text: body }) : null,
    actionLabel
      ? el('button', { class: 'btn btn-primary', type: 'button', onClick: onAction }, [actionLabel])
      : null,
  ]);
}

export function alertRow({ tone = 'info', title, body }) {
  const glyph = { info: 'info', warn: 'alert', ok: 'shieldCheck' }[tone] || 'info';
  return el('div', { class: 'alert alert-line' }, [
    icon(glyph, 'icon'),
    el('div', { class: 'stack gap-xs' }, [
      title ? el('strong', { text: title }) : null,
      body ? el('p', { class: 'small mute', text: body }) : null,
    ]),
  ]);
}

export function sectionHeading({ kicker, title, lead, actions }) {
  return el('div', { class: 'between wrapf row gap-lg' }, [
    el('div', { class: 'stack gap-sm' }, [
      kicker ? el('p', { class: 'kicker', text: kicker }) : null,
      el('h2', { text: title }),
      lead ? el('p', { class: 'lead', text: lead }) : null,
    ]),
    actions || null,
  ]);
}

export function segmented({ options, active, onSelect }) {
  return el(
    'div',
    { class: 'segmented', role: 'tablist' },
    options.map((option) =>
      el(
        'button',
        {
          type: 'button',
          role: 'tab',
          'aria-selected': option.value === active ? 'true' : 'false',
          'data-value': option.value,
          onClick: () => onSelect(option.value),
        },
        [option.label]
      )
    )
  );
}

export function table({ columns, rows, caption }) {
  return el('table', { class: 'table' }, [
    caption ? el('caption', { text: caption }) : null,
    el('thead', {}, [
      el('tr', {}, columns.map((column) => el('th', { text: column.label, scope: 'col' }))),
    ]),
    el(
      'tbody',
      {},
      rows.map((row) =>
        el(
          'tr',
          {},
          columns.map((column) =>
            el('td', {
              class: column.numeric ? 'num' : '',
              text:
                row[column.key] === null || row[column.key] === undefined
                  ? '—'
                  : String(row[column.key]),
            })
          )
        )
      )
    ),
  ]);
}

export function starRow(count = 5) {
  return el(
    'span',
    { class: 'starlist', role: 'img', 'aria-label': `${count} out of 5` },
    Array.from({ length: count }, () => icon('star', 'icon-sm'))
  );
}

export function metaLine({ value, label }) {
  return el('div', { class: 'stack gap-xs' }, [
    el('p', { class: 'num-site', text: value }),
    el('p', { class: 'kicker', text: label }),
  ]);
}

export function toast(message, glyph = 'check') {
  const node = el('div', { class: 'toast', role: 'status' }, [
    icon(glyph, 'icon'),
    el('span', { class: 'small', text: message }),
  ]);
  document.body.append(node);
  window.setTimeout(() => node.remove(), 4200);
  return node;
}
