// Append to a generated leaderboard and open with a headless browser.
// Uses the page's actual DOM and event handlers; no GPU or benchmark recomputation.
(() => {
  try {
    const assert = (condition, message) => { if (!condition) throw Error(message); };
    const rows = () => Array.from(document.querySelector('tbody').rows);
    const models = () => rows().map(row => data.models[row.dataset.row]);
    assert(document.querySelectorAll('table').length === 1, 'Expected one table');
    assert(rows().length === data.models.length, 'One row per model');
    const headers = Array.from(document.querySelectorAll('thead th'));
    assert(headers.length === data.datasets.length + 4, 'Metadata plus dataset columns');
    assert(headers[3].getAttribute('aria-sort') === 'descending', 'Default: descending average');
    let means = models().map(mean).filter(v => v !== null);
    for (let i = 1; i < means.length; i++) {
      assert(means[i - 1] >= means[i], 'Default average ordering');
    }
    headers[3].querySelector('button').click();
    means = models().map(mean).filter(v => v !== null);
    for (let i = 1; i < means.length; i++) {
      assert(means[i - 1] <= means[i], 'Ascending average ordering');
    }
    for (let column = 0; column < headers.length; column++) {
      headers[column].querySelector('button').click();
      const direction = headers[column].getAttribute('aria-sort');
      const values = models().map(model => value(model, column)).filter(v => v !== null);
      for (let i = 1; i < values.length; i++) {
        const delta = typeof values[i] === 'string' ? values[i - 1].localeCompare(values[i]) : values[i - 1] - values[i];
        assert(direction === 'descending' ? delta >= 0 : delta <= 0, `Column ${column} ordering`);
      }
    }
    metric.value = 'ndcg@10';
    metric.dispatchEvent(new Event('change'));
    for (const row of rows()) {
      const model = data.models[row.dataset.row];
      assert(row.querySelector('th').textContent === model.model, 'Full model ID includes publisher');
      const scores = data.datasets.map(d => model.cells[d]?.metrics['ndcg@10'] ?? null);
      const expected = scores.some(v => v === null) ? '—' : (scores.reduce((a, b) => a + b, 0) / scores.length * 100).toFixed(2);
      assert(row.querySelector('[data-score="mean"]').textContent === expected, 'NDCG mean updates');
    }
    const filter = document.getElementById('filter');
    filter.value = models()[0].model;
    filter.dispatchEvent(new Event('input'));
    assert(rows().filter(row => !row.hidden).length === 1, 'Model filter');
    document.body.dataset.uiTest = 'passed';
  } catch (error) {
    document.body.dataset.uiTest = 'failed';
    document.body.dataset.uiError = error.message;
  }
})();
