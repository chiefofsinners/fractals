// Small UI bits independent of the renderer in main.js: touch-aware hint
// copy, custom <select> dropdowns wired to hidden native ones, and the
// mobile controls-panel toggle. Three IIFEs because they don't share state.

(function () {
  var touch = window.matchMedia('(pointer: coarse)').matches;
  document.getElementById('hint').textContent = touch
    ? 'Pinch to zoom · drag to pan'
    : 'Scroll to zoom · drag to pan · click to zoom in · shift-click to zoom out';
})();

// Custom select dropdowns — wired to the hidden native <select> elements
// so main.js can keep using .value and the "change" event unchanged.
(function () {
  function initCsel(csel) {
    const selectId = csel.dataset.for;
    const nativeSelect = document.getElementById(selectId);
    const trigger = csel.querySelector('.csel__trigger');
    const label = csel.querySelector('.csel__label');
    const options = csel.querySelectorAll('.csel__option');

    function syncFromNative() {
      const val = nativeSelect.value;
      options.forEach(o => {
        const nativeOpt = nativeSelect.querySelector(`option[value="${o.dataset.value}"]`);
        const disabled = !!(nativeOpt && nativeOpt.disabled);
        const active = o.dataset.value === val;
        o.disabled = disabled;
        o.setAttribute('aria-disabled', String(disabled));
        o.setAttribute('aria-selected', String(active));
      });
      if (nativeSelect.selectedOptions && nativeSelect.selectedOptions.length > 0) {
        label.textContent = nativeSelect.selectedOptions[0].textContent;
      }
    }

    function open() {
      csel.dataset.open = '';
      trigger.setAttribute('aria-expanded', 'true');
    }
    function close() {
      delete csel.dataset.open;
      trigger.setAttribute('aria-expanded', 'false');
    }
    function toggle() { csel.hasAttribute('data-open') ? close() : open(); }

    function select(value) {
      const prev = nativeSelect.value;
      const nativeOpt = nativeSelect.querySelector(`option[value="${value}"]`);
      if (!nativeOpt || nativeOpt.disabled) return;
      options.forEach(o => {
        const active = o.dataset.value === value;
        o.setAttribute('aria-selected', String(active));
      });
      const chosen = csel.querySelector(`.csel__option[data-value="${value}"]`);
      if (chosen) label.textContent = chosen.textContent;
      nativeSelect.value = value;
      if (prev !== value) {
        nativeSelect.dispatchEvent(new Event('change', { bubbles: true }));
      }
      close();
    }

    // Also sync back if main.js changes nativeSelect.value directly
    // (e.g. boot() disabling WebGPU and falling back to webgl).
    nativeSelect.addEventListener('change', syncFromNative);

    // Initial sync from the native select so HTML option order/selected
    // stays the source of truth for both desktop and mobile controls.
    syncFromNative();

    trigger.addEventListener('click', e => { e.stopPropagation(); toggle(); });
    options.forEach(o => o.addEventListener('click', () => select(o.dataset.value)));

    // Keyboard: Escape closes, arrow keys move, Enter/Space selects
    csel.addEventListener('keydown', e => {
      const open = csel.hasAttribute('data-open');
      if (e.key === 'Escape') { close(); trigger.focus(); return; }
      if ((e.key === 'Enter' || e.key === ' ') && !open) { e.preventDefault(); open(); return; }
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        const opts = [...options];
        const cur = opts.findIndex(o => o === document.activeElement);
        const next = e.key === 'ArrowDown' ? Math.min(cur + 1, opts.length - 1) : Math.max(cur - 1, 0);
        opts[next].focus();
      }
    });
  }

  // Close any open dropdown when clicking elsewhere
  document.addEventListener('click', () => {
    document.querySelectorAll('.csel[data-open]').forEach(c => {
      delete c.dataset.open;
      c.querySelector('.csel__trigger').setAttribute('aria-expanded', 'false');
    });
    // Also close the mobile controls panel on outside tap
    const controls = document.getElementById('controls');
    const toggle = document.getElementById('controlsToggle');
    if (controls && controls.classList.contains('open')) {
      controls.classList.remove('open');
      toggle.setAttribute('aria-expanded', 'false');
    }
  });

  document.querySelectorAll('.csel').forEach(initCsel);
})();

// Controls panel toggle (mobile)
(function () {
  const toggle = document.getElementById('controlsToggle');
  const controls = document.getElementById('controls');
  const reset = document.getElementById('reset');
  if (!toggle) return;

  function closePanel() {
    controls.classList.remove('open');
    toggle.setAttribute('aria-expanded', 'false');
    controls.querySelectorAll('.csel[data-open]').forEach(function (c) {
      delete c.dataset.open;
      c.querySelector('.csel__trigger').setAttribute('aria-expanded', 'false');
    });
  }

  toggle.addEventListener('click', function (e) {
    e.stopPropagation(); // prevent the outside-click handler from immediately closing it
    const open = controls.classList.toggle('open');
    toggle.setAttribute('aria-expanded', String(open));
    if (!open) {
      closePanel();
    }
  });
  if (reset) reset.addEventListener('click', closePanel);
  // Also stop clicks inside the panel from bubbling to the outside-click handler
  controls.addEventListener('click', function (e) { e.stopPropagation(); });
})();
