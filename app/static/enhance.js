/* Visual-only helper: turns "#11 • To do • No due date" inside .task-meta
   into separate styled chips. Does not touch app.js or any backend call. */
(function () {
  var list = document.getElementById('task-list');
  if (!list) return;

  function chip(text, cls) {
    var el = document.createElement('span');
    el.className = cls;
    el.textContent = text;
    return el;
  }

  function statusClass(t) {
    var v = t.toLowerCase();
    if (v.indexOf('done') > -1 || v.indexOf('complete') > -1) return 'status-done';
    if (v.indexOf('progress') > -1) return 'status-in-progress';
    return 'status-todo';
  }

  function enhance() {
    list.querySelectorAll('.task-meta').forEach(function (meta) {
      if (meta.children.length || meta.textContent.indexOf('\u2022') === -1) return;
      var parts = meta.textContent.split('\u2022').map(function (p) { return p.trim(); }).filter(Boolean);
      meta.textContent = '';
      parts.forEach(function (p) {
        if (/^#\d+/.test(p)) meta.appendChild(chip(p, 'task-id'));
        else if (/due/i.test(p)) {
          var none = /no due/i.test(p);
          var over = /overdue/i.test(p);
          meta.appendChild(chip(p, 'due' + (none ? ' none' : '') + (over ? ' overdue' : '')));
        } else meta.appendChild(chip(p, statusClass(p)));
      });
    });
  }

  new MutationObserver(enhance).observe(list, { childList: true, subtree: true });
  enhance();
})();