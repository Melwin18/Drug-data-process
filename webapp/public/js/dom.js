/* Tiny DOM helpers. No framework, no build step. */

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key === 'html') node.innerHTML = value;
    else if (key === 'style' && typeof value === 'object') Object.assign(node.style, value);
    else if (key.startsWith('on') && typeof value === 'function') {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else node.setAttribute(key, value === true ? '' : String(value));
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export const qs = (selector, scope = document) => scope.querySelector(selector);
export const qsa = (selector, scope = document) => Array.from(scope.querySelectorAll(selector));

export function mount(target, nodes) {
  const host = typeof target === 'string' ? qs(target) : target;
  if (!host) return null;
  host.replaceChildren(...[].concat(nodes).filter(Boolean));
  return host;
}

export function setText(target, value) {
  const host = typeof target === 'string' ? qs(target) : target;
  if (host) host.textContent = value;
}

export function show(target, visible = true) {
  const host = typeof target === 'string' ? qs(target) : target;
  if (host) host.classList.toggle('hidden', !visible);
}

export function clear(target) {
  const host = typeof target === 'string' ? qs(target) : target;
  if (host) host.replaceChildren();
}

/* Sets a data-driven CSS custom property, e.g. a bar fill percentage. */
export function setValueVar(node, name, value) {
  if (node) node.style.setProperty(name, String(value));
}

export function debounce(fn, wait = 120) {
  let handle = null;
  return (...args) => {
    window.clearTimeout(handle);
    handle = window.setTimeout(() => fn(...args), wait);
  };
}
