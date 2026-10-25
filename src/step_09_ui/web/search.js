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

function renderResult(item) {
  const article = element('article', '', 'result');
  article.dataset.docId = item.doc_id;
  article.append(element('div',
    `${item.technology} / ${(item.content_categories || []).join(", ")} · #${item.rank} · ${item.score.toFixed(4)}`, 'meta'));
  const heading = element('h2', '');
  const link = element('a', item.title);
  link.href = item.document_url;
  link.target = '_blank';
  link.rel = 'noopener noreferrer';
  heading.append(link);
  const original = element('a', item.url, 'url');
  const target = new URL(item.url);
  if (['http:', 'https:'].includes(target.protocol)) original.href = target.href;
  original.target = '_blank';
  original.rel = 'noopener noreferrer';
  article.append(heading, original);
  if (item.passage.heading) article.append(element('div', item.passage.heading, 'heading'));
  article.append(element('p', item.passage.text, 'snippet'));
  if (item.zone_contributions) {
    const details = element('details', '', 'meta');
    details.append(element('summary', 'Zone contributions'));
    details.append(element('div', Object.entries(item.zone_contributions).map(([zone, value]) => `${zone}: ${value.toFixed(3)}`).join(' · ')));
    article.append(details);
  }
  return article;
}

async function search(newSearch) {
  if (newSearch) {
    page = 1;
    active = {q: byId('query').value, technology: byId('technology').value,
              category: byId('category').value, mode: byId('mode').value};
  }
  if (!active?.q.trim()) return;
  if (controller) controller.abort();
  const request = new AbortController();
  controller = request;
  byId('search').disabled = true;
  byId('status').textContent = 'Searching the corpus…';
  try {
    const response = await fetch('/api/search?' + new URLSearchParams({...active, page}),
                                 {signal: request.signal});
    const data = await response.json();
    if (!response.ok) throw Error(data.error || 'Search failed');
    byId('results').replaceChildren(...data.results.map(renderResult));
    if (!data.results.length) {
      byId('results').append(element('p', 'No documents match these filters.', 'empty'));
    }
    byId('status').textContent = `${data.total_documents.toLocaleString()} documents · ` +
      `showing up to ${data.retrieved_depth.toLocaleString()} ranked results · ` +
      `${data.elapsed_ms.toFixed(0)} ms${data.cache_hit ? ' · cached' : ''}`;
    byId('pagination').hidden = !data.results.length;
    byId('page').textContent = `Page ${page} of ${Math.ceil(data.retrieved_depth / 10)}`;
    byId('prev').disabled = page === 1;
    byId('next').disabled = page * 10 >= data.retrieved_depth;
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
  byId('corpus').textContent = `${options.documents.toLocaleString()} DOCUMENTS · ` +
                              `${options.passages.toLocaleString()} PASSAGES`;
  for (const name of ['technology', 'category']) {
    const values = options[name === 'technology' ? 'technologies' : 'categories'];
    for (const value of values) {
      const option = element('option', value);
      option.value = value;
      byId(name).append(option);
    }
  }
  const labels = {'dense':'Dense cosine baseline','zones':'Learned zone scoring','routed-centroid':'Centroid category routing'};
  for (const mode of options.modes) {
    const option = element('option', labels[mode]);
    option.value = mode;
    byId('mode').append(option);
  }
  byId('mode').value = options.default_mode;
  const query = new URLSearchParams(location.search).get('q');
  if (query) {
    byId('query').value = query;
    search(true);
  }
}

byId('form').onsubmit = event => {event.preventDefault(); search(true);};
byId('prev').onclick = () => {page--; search(false);};
byId('next').onclick = () => {page++; search(false);};
for (const button of document.querySelectorAll('.examples button')) {
  button.onclick = () => {byId('query').value = button.textContent; search(true);};
}
loadOptions().catch(() => {
  byId('status').textContent = 'Could not connect to the local search service.';
});
