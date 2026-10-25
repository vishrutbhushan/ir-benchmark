const byId = id => document.getElementById(id);
let options;
let position = 0;
let requestNumber = 0;

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

async function loadQuery() {
  const number = ++requestNumber;
  const query = options.queries[position];
  byId('position').textContent = `Query ${position + 1} of ${options.queries.length}`;
  byId('query').textContent = query.query_text;
  byId('query_meta').textContent = `${query.query_id} | ${query.query_type} | ${query.target_tech}`;
  byId('prev').disabled = position === 0;
  byId('next').disabled = position === options.queries.length - 1;
  byId('error').textContent = '';
  byId('results').replaceChildren();
  byId('progress').textContent = 'Loading results...';
  const values = {engine: byId('engine').value, variant: options.variants[byId('engine').value][0], query_id: query.query_id};
  const response = await fetch('/api/query?' + new URLSearchParams(values));
  const data = await response.json();
  if (!response.ok) throw Error(data.error);
  if (number !== requestNumber) return;
  byId('progress').textContent = `${data.judged}/10 results labeled in this variant | ${data.pool_judged}/${data.pool_size} labeled in this query's combined pool. Labels carry across variants.`;
  const results = data.results.map(row => {
    const article = element('article', '', 'result');
    article.append(element('h3', `${row.rank}. ${row.title}`));
    article.append(element('div', `${row.doc_id} | ${row.technology} | ${JSON.parse(row.categories).join(', ')} | Score ${Number(row.score).toFixed(4)}`, 'meta'));
    const links = element('div', '', 'links');
    links.append(link(row.url, row.url));
    if (row.saved_url) links.append(link('Saved HTML', row.saved_url));
    else links.append(element('span', 'Saved HTML unavailable in supplied corpus', 'meta'));
    article.append(links);
    const buttons = element('div', '', 'judgments');
    for (const [text, value, className] of [['Relevant', 1, 'relevant'], ['Non-relevant', 0, 'irrelevant'], ['Clear', null, 'clear']]) {
      const button = element('button', text, className + (row.relevant === value ? ' selected' : ''));
      button.onclick = async () => {
        for (const child of buttons.children) child.disabled = true;
        try {
          const response = await fetch('/api/judgment', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({query_id: query.query_id, doc_id: row.doc_id, relevant: value})});
          const result = await response.json();
          if (!response.ok) throw Error(result.error);
          await loadQuery();
        } catch (error) {
          byId('error').textContent = error.message;
          for (const child of buttons.children) child.disabled = false;
        }
      };
      buttons.append(button);
    }
    article.append(buttons);
    return article;
  });
  byId('results').replaceChildren(...results);
}

function showError(error) { byId('error').textContent = error.message; }
byId('engine').onchange = () => loadQuery().catch(showError);
byId('prev').onclick = () => {position--; loadQuery().catch(showError);};
byId('next').onclick = () => {position++; loadQuery().catch(showError);};
fetch('/api/options').then(response => response.json()).then(data => {
  options = data;
  byId('comparison').textContent = 'Labels save automatically and carry across matching query/document pairs. Recall and MAP use the combined judged pool.';
  loadQuery().catch(showError);
}).catch(showError);
