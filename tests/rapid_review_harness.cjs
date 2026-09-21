// Execute the rendered page's real script with a small DOM fixture and real localhost fetch.
// No npm packages are required; browser rendering is checked separately in the UI smoke test.
const fs = require('node:fs');
const vm = require('node:vm');
const config = JSON.parse(fs.readFileSync(0, 'utf8'));
const nodes = [];
const ids = new Map();
const navigations = [];
const requests = [];
const pending = [];
const listeners = {};
function attributes(text) {
  return Object.fromEntries(Array.from(text.matchAll(/([\w-]+)="([^"]*)"/g), m => [m[1], m[2]]));
}
function element(tag, attrs = {}) {
  const node = {
    tagName: tag.toUpperCase(), ...attrs, value: attrs.value || '',
    defaultValue: attrs.value || '', defaultChecked: !!attrs.checked,
    disabled: false, textContent: '', style: {}, isContentEditable: false,
    closest() { return ['INPUT', 'TEXTAREA', 'SELECT'].includes(this.tagName) ||
      this.isContentEditable ? this : null; },
    addEventListener(type, callback) { this[type] = callback; },
    scrollIntoView() {},
    getAttribute(name) { return this[name]; }
  };
  let checked = !!attrs.checked;
  Object.defineProperty(node, 'checked', {
    get() { return checked; },
    set(value) {
      checked = value;
      if (value && node.type === 'radio') {
        nodes.filter(other => other !== node && other.name === node.name)
          .forEach(other => { other.checked = false; });
      }
    }
  });
  nodes.push(node);
  if (node.id) { ids.set(node.id, node); }
  return node;
}
for (const match of config.html.matchAll(/<input\b([^>]*)>/g)) {
  const attrs = attributes(match[1]);
  attrs.checked = /\bchecked\b/.test(match[1]);
  element('input', attrs);
}
element('textarea', {id: 'note', name: 'note'});
element('button', {type: 'submit'});
for (const id of ['override-state', 'save-status', 'save-error']) { element('p', {id}); }
const previousMatch = config.html.match(/<a id="previous-case" href="([^"]*)"/);
if (previousMatch) { element('a', {id: 'previous-case', href: previousMatch[1]}); }
const form = element('form', {id: 'form', action: config.base + '/decision', method: 'post'});
form.dataset = {nextUndecided: config.html.match(/data-next-undecided="([^"]*)"/)[1]};
form.elements = nodes.filter(node => ['INPUT', 'TEXTAREA', 'BUTTON'].includes(node.tagName));
form.elements.forEach(node => { if (node.name) { form.elements[node.name] = node; } });
form.reset = () => form.elements.forEach(node => {
  node.value = node.defaultValue; node.checked = node.defaultChecked;
});
form.requestSubmit = () => pending.push(form.submit({currentTarget: form, preventDefault() {}}));
function query(selector) {
  if (!selector.startsWith('input')) { return null; }
  const name = selector.match(/\[name="([^"]+)"\]/);
  const value = selector.match(/\[value="([^"]+)"\]/);
  return nodes.find(node => node.tagName === 'INPUT' &&
    (!name || node.name === name[1]) && (!value || node.value === value[1]) &&
    (!selector.includes(':checked') || node.checked)) || null;
}
const body = element('body');
const document = {
  activeElement: body, getElementById(id) { return ids.get(id) || null; },
  querySelector: query, querySelectorAll() { return []; },
  addEventListener(type, callback) { listeners[type] = callback; }
};
const context = vm.createContext({
  document, URL, URLSearchParams,
  FormData: class {
    constructor(source) { this.form = source; }
    *[Symbol.iterator]() {
      for (const node of this.form.elements) {
        if (node.name && !node.disabled && (node.type !== 'radio' || node.checked)) {
          yield [node.name, node.value];
        }
      }
    }
  },
  DOMParser: class {
    parseFromString(html) {
      const match = html.match(/<p\b[^>]*>([\s\S]*?)<\/p>/);
      return {querySelector() { return match ? {
        textContent: match[1].replace(/<[^>]*>/g, '').replaceAll('&amp;', '&')
      } : null; }};
    }
  },
  window: {
    location: {origin: config.base, assign(url) { navigations.push(url); }},
    addEventListener(type, callback) { if (type === 'load') { listeners.load = callback; } }
  },
  async fetch(url, options) {
    requests.push({url, body: String(options.body),
      saving: ids.get('save-status').textContent,
      allDisabled: form.elements.every(node => node.disabled)});
    if (config.delay) { await new Promise(resolve => setTimeout(resolve, config.delay)); }
    return fetch(url, options);
  }
});
const script = config.html.match(/<script>([\s\S]*?)<\/script>/)[1];
vm.runInContext(script, context);
listeners.load();
(async function () {
  for (const action of config.actions) {
    if (action.select) {
      query(`input[name="${action.select}_decision"][value="${action.value}"]`).checked = true;
    } else if (action.center) {
      vm.runInContext(`overrides.${action.center} = {row: 610, column: 430}; render();`, context);
      query(`input[name="${action.center}_decision"][value="NEEDS_CENTER_OVERRIDE"]`).checked = true;
    } else if (action.laterality) {
      query(`input[name="laterality_decision"][value="${action.laterality}"]`).checked = true;
    } else if (action.note !== undefined) {
      ids.get('note').value = action.note;
    } else if (action.submit) {
      form.requestSubmit();
    } else if (action.key) {
      let target = body;
      if (action.focus === 'textarea') { target = ids.get('note'); }
      else if (action.focus === 'input') { target = element('input'); }
      else if (action.focus === 'select') { target = element('select'); }
      else if (['editable', 'editable-child'].includes(action.focus)) {
        target = element('span'); target.isContentEditable = true;
      }
      document.activeElement = target;
      listeners.keydown({key: action.key, target, repeat: !!action.repeat,
        ctrlKey: !!action.ctrlKey, isComposing: !!action.isComposing,
        preventDefault() {}});
    }
  }
  await Promise.all(pending);
  console.log(JSON.stringify({requests, navigations,
    selected: ['screen_left', 'screen_right'].map(panel => {
      const chosen = query(`input[name="${panel}_decision"]:checked`);
      return chosen ? chosen.value : null;
    }), overrides: JSON.parse(ids.get('overrides').value), note: ids.get('note').value,
    status: ids.get('save-status').textContent, error: ids.get('save-error').textContent,
    overrideState: ids.get('override-state').textContent,
    disabled: form.elements.some(node => node.disabled)}));
})().catch(error => { console.error(error); process.exitCode = 1; });
