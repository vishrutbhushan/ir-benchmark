const byId = id => document.getElementById(id);
let page = 1;
let active = null;
let controller = null;

function element(tag, text, className) {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
}

function link(text, url) {
  const node = element('a', text);
  const target = new URL(url, location.origin);
  if (['http:', 'https:'].includes(target.protocol)) node.href = target.href;
  node.target = '_blank';
  node.rel = 'noopener noreferrer';
  return node;
}

function renderResult(item) {
  const article = element('article', '', 'result');
  article.append(element('div', `#${item.rank} | ${item.doc_id} | Score: ${item.score.toFixed(4)}`, 'meta'));
  article.append(element('h2', item.title));
  article.append(element('div', `Technology: ${item.technology}`, 'metadata'));
  article.append(element('div', `Category: ${item.content_categories.join(', ')}`, 'metadata'));
  const links = element('div', '', 'links');
  links.append(link(item.url, item.url), link('Saved HTML', item.document_url));
  article.append(links);
  const details = element('details', '', 'metadata');
  details.append(element('summary', 'Scores by zone'));
  const table = element('table', '');
  const header = element('tr', '');
  for (const text of ['Zone', 'Cosine', 'Weighted contribution']) header.append(element('th', text));
  table.append(header);
  for (const [zone, score] of Object.entries(item.zone_scores)) {
    const row = element('tr', '');
    const contribution = item.zone_contributions?.[zone];
    for (const text of [zone, score === null ? 'Absent' : score.toFixed(4), contribution == null ? '-' : contribution.toFixed(4)]) {
      row.append(element('td', text));
    }
    table.append(row);
  }
  details.append(table);
  article.append(details);
  return article;
}

async function search(newSearch) {
  if (newSearch) {
    page = 1;
    active = {q: byId('query').value, mode: byId('mode').value};
  }
  if (!active?.q.trim()) return;
  if (controller) controller.abort();
  const request = new AbortController();
  controller = request;
  byId('search').disabled = true;
  byId('prev').disabled = true;
  byId('next').disabled = true;
  byId('status').textContent = 'Searching...';
  try {
    const response = await fetch('/api/search?' + new URLSearchParams({...active, page}), {signal: request.signal});
    const data = await response.json();
    if (!response.ok) throw Error(data.error || 'Search failed');
    byId('results').replaceChildren(...data.results.map(renderResult));
    if (!data.results.length) byId('results').append(element('p', 'No results.'));
    byId('status').textContent = `${data.total_results} results | ${data.elapsed_ms.toFixed(0)} ms | ` +
      `${data.scanned_vectors.toLocaleString()} vectors searched${data.cache_hit ? ' | cached' : ''}`;
    const pages = Math.ceil(data.total_results / data.page_size);
    byId('pagination').hidden = pages <= 1;
    byId('page').textContent = `Page ${page} of ${pages}`;
    byId('prev').disabled = page === 1;
    byId('next').disabled = page >= pages;
  } catch (error) {
    if (error.name !== 'AbortError') {
      byId('status').textContent = error.message;
      byId('results').replaceChildren();
      byId('pagination').hidden = true;
    }
  } finally {
    if (controller === request) byId('search').disabled = false;
  }
}

async function loadOptions() {
  const response = await fetch('/api/options');
  if (!response.ok) throw Error('Search service unavailable');
  const options = await response.json();
  byId('corpus').textContent = `${options.documents.toLocaleString()} documents`;
  const labels = {dense: 'Dense cosine', zones: 'Weighted zones', 'routed-centroid': 'Category routing'};
  for (const mode of options.modes) {
    const option = element('option', labels[mode]);
    option.value = mode;
    byId('mode').append(option);
  }
  byId('mode').value = options.default_mode;
}

byId('query').value = '';
byId('form').onsubmit = event => {event.preventDefault(); search(true);};
byId('prev').onclick = () => {page--; search(false);};
byId('next').onclick = () => {page++; search(false);};
loadOptions().catch(() => {byId('status').textContent = 'Could not connect to the search service.';});
