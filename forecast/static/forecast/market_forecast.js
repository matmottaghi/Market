/* Hour-index slicer. Pure selection helpers are also used by the regression checks. */
function forecastWindow(rows, start, end, point) {
  const last = Math.max(0, rows.length - 1);
  const clamp = v => Math.max(0, Math.min(last, Number.isFinite(Number(v)) ? Math.trunc(Number(v)) : 0));
  start = clamp(start); end = Math.max(start, clamp(end));
  point = Math.max(start, Math.min(end, clamp(point)));
  return {start, end, point, rows: rows.slice(start, end + 1)};
}
function forecastLabel(row) {
  return row.target_day !== undefined ? `${row.target_day === 0 ? 'باقی‌ماندهٔ جلسه جاری' : 'جلسه ' + row.target_day} · ساعت ${String(row.target_hour).padStart(2, '0')}:00` : `افق ${row.horizon} روز معاملاتی`;
}
if (typeof module !== 'undefined') module.exports = {forecastWindow, forecastLabel};
if (typeof document !== 'undefined') (() => {
  const root = document.getElementById('market-forecast');
  if (!root) return;
  const payload = JSON.parse(document.getElementById('market-forecast-data').textContent);
  const rows = payload.rows || [], $ = id => document.getElementById('forecast-' + id);
  if (!rows.length) {
    $('empty').hidden = false;
    for (const id of ['controls', 'summary', 'price', 'return', 'distribution']) $(id).hidden = true;
    return;
  }
  if (typeof Plotly === 'undefined') {
    $('empty').hidden = false; $('empty').textContent = 'بارگذاری نمودارها انجام نشد؛ صفحه را دوباره بارگذاری کنید.'; return;
  }
  const hourlyModel = ['hourly_returns_v2', 'hourly_returns_v3'].includes(rows[0].target_schema);
  const hourlyReturn = i => {
    const row = rows[i], previous = rows[i - 1];
    if (!hourlyModel || !Number.isFinite(row.mu_log)) return null;
    if (Number(row.horizon) === 1) return Math.expm1(row.mu_log);
    return previous && Number(previous.horizon) + 1 === Number(row.horizon) && Number.isFinite(previous.mu_log) ? Math.expm1(row.mu_log - previous.mu_log) : null;
  };
  if (!hourlyModel) { $('return-mode').value = 'cumulative'; $('return-mode').options[0].disabled = true; }
  const empirical = rows.some(r => Array.isArray(r.scenario_prices) && r.scenario_prices.length);
  $('model').textContent = ['hourly_returns_v2', 'hourly_returns_v3'].includes(rows[0].target_schema) ? 'مدل بازده ساعتی' : 'خروجی مدل قدیمی';
  $('scenarios').disabled = !empirical;
  $('scenarios').checked = empirical;
  const number = v => v === null || v === undefined || !Number.isFinite(Number(v)) ? '—' : Number(v).toLocaleString('fa-IR', {maximumFractionDigits: 2});
  const percent = v => v === null || v === undefined ? '—' : number(v * 100) + '٪';
  function options(el, indexes) {
    el.replaceChildren(...indexes.map(i => { const option = document.createElement('option'); option.value = i; option.textContent = forecastLabel(rows[i]); return option; }));
  }
  const indexes = rows.map((_, i) => i);
  for (const id of ['start-select', 'end-select']) options($(id), indexes);
  for (const id of ['start', 'end']) $(id).max = rows.length - 1;
  const url = new URL(location.href), integer = (name, fallback) => url.searchParams.has(name) && Number.isFinite(Number(url.searchParams.get(name))) ? Number(url.searchParams.get(name)) : fallback;
  const initialPoint = Math.max(0, rows.findIndex(r => Number(r.horizon) === Number(payload.horizon)));
  let state = forecastWindow(rows, integer('forecast_from', 0), integer('forecast_to', rows.length - 1), integer('forecast_point', initialPoint));
  const config = {responsive: true, displaylogo: false};
  function layout(title, ylabel, visible) {
    const step = Math.max(1, Math.ceil(visible.length / 8));
    const ticks = visible.filter((_, i) => i % step === 0 || i === visible.length - 1);
    return {title: {text: title}, height: 370, margin: {t: 55, b: 85, l: 70, r: 25}, paper_bgcolor: '#152237', plot_bgcolor: '#152237', font: {color: '#dce6f4', family: 'Tahoma'}, xaxis: {title: {text: 'ساعت‌های معاملاتی آینده'}, tickmode: 'array', tickvals: ticks.map(r => r.horizon), ticktext: ticks.map(forecastLabel)}, yaxis: {title: {text: ylabel}}, legend: {orientation: 'h', y: 1.12}, hovermode: 'x unified'};
  }
  function render() {
    const visible = state.rows, chosen = rows[state.point];
    const chosenEmpirical = Array.isArray(chosen.scenario_prices) && chosen.scenario_prices.length > 0;
    for (const id of ['start', 'start-select']) $(id).value = state.start;
    for (const id of ['end', 'end-select']) $(id).value = state.end;
    options($('point'), indexes.slice(state.start, state.end + 1)); $('point').value = state.point;
    $('window').textContent = `${forecastLabel(visible[0])} تا ${forecastLabel(visible[visible.length - 1])} — ${number(visible.length)} ساعت از ${number(rows.length)} ساعت دارای خروجی`;
    const summary = [['ساعت بررسی', forecastLabel(chosen)], [chosenEmpirical ? 'قیمت مرکزی (میانه)' : 'قیمت مورد انتظار', number(chosen.predicted_price)], ['بازده از قیمت مبنا', percent(chosen.predicted_return)], ['حد پایین', number(chosen.lower_price)], ['حد بالا', number(chosen.upper_price)], ['احتمال صعود از مبنا', percent(chosen.probability_up)]];
    $('summary').replaceChildren(...summary.map(([label, value]) => { const card = document.createElement('div'); card.className = 'card'; const strong = document.createElement('strong'); strong.textContent = value; card.append(document.createTextNode(label), strong); return card; }));
    const x = visible.map(r => r.horizon), labels = visible.map(forecastLabel), traces = [];
    const trace = (key, name, extra = {}) => ({x, y: visible.map(r => r[key]), text: labels, name, mode: 'lines', hovertemplate: '%{text}<br>%{y:,.2f}<extra>%{fullData.name}</extra>', ...extra});
    if ($('band').checked) {
      traces.push(trace('upper_price', 'حد بالا', {line: {width: 0}}));
      traces.push(trace('lower_price', 'محدودهٔ قیمت ' + percent(payload.confidence), {line: {width: 0}, fill: 'tonexty', fillcolor: 'rgba(105,224,197,.15)'}));
    }
    if ($('scenarios').checked) {
      const count = Math.min(24, ...visible.map(r => (r.scenario_prices || []).length));
      for (let j = 0; j < count; j++) traces.push({x, y: visible.map(r => r.scenario_prices[j]), mode: 'lines', line: {width: 1, color: 'rgba(150,170,220,.3)'}, showlegend: false, hoverinfo: 'skip', name: `سناریو ${j + 1}`});
    }
    traces.push(trace('predicted_price', empirical ? 'میانهٔ قیمت' : 'قیمت مورد انتظار', {line: {color: '#69e0c5', width: 3}}));
    traces.push({x: [chosen.horizon], y: [chosen.predicted_price], mode: 'markers', marker: {color: '#f7c979', size: 11}, name: 'ساعت بررسی'});
    const priceLayout = layout(payload.symbol + ' — مسیر قیمت و عدم‌قطعیت', 'قیمت', visible);
    priceLayout.shapes = [{type: 'line', xref: 'paper', x0: 0, x1: 1, y0: chosen.current_price, y1: chosen.current_price, line: {dash: 'dot', color: '#93afcc'}}];
    Plotly.react($('price'), traces, {...priceLayout, height: 460}, config);
    const hourly = $('return-mode').value === 'hourly';
    const returns = visible.map((r, i) => hourly ? hourlyReturn(state.start + i) : r.predicted_return);
    Plotly.react($('return'), [{x, y: returns.map(r => r === null ? null : 100 * r), text: labels, type: 'bar', marker: {color: returns.map(r => r >= 0 ? '#69e0c5' : '#ff657c')}, name: hourly ? 'بازده ساعتی شبکه' : 'بازده از مبنا', hovertemplate: '%{text}<br>%{y:.2f}٪<extra></extra>'}], layout(hourly ? 'بازده هر ساعت نسبت به ساعت معاملاتی قبل' : 'بازده تا هر ساعت نسبت به قیمت مبنا', 'بازده ٪', visible), config);
    let distribution;
    if (Array.isArray(chosen.scenario_prices) && chosen.scenario_prices.length) {
      distribution = [{x: chosen.scenario_prices, type: 'histogram', histnorm: 'probability density', marker: {color: '#69e0c5'}, name: 'سناریوهای قیمت'}];
    } else {
      // Legacy models retain their saved lognormal interpretation.
      const sigma = chosen.sigma_log, mu = chosen.mu_log, base = chosen.current_price;
      const prices = Array.from({length: 200}, (_, i) => base * Math.exp(mu - 3 * sigma + 6 * sigma * i / 199));
      distribution = [{x: prices, y: prices.map(p => Math.exp(-.5 * ((Math.log(p / base) - mu) / sigma) ** 2) / (p * sigma * Math.sqrt(2 * Math.PI))), fill: 'tozeroy', name: 'چگالی قیمت مدل قدیمی'}];
    }
    Plotly.react($('distribution'), distribution, {title: {text: 'توزیع قیمت — ' + forecastLabel(chosen)}, height: 370, paper_bgcolor: '#152237', plot_bgcolor: '#152237', font: {color: '#dce6f4', family: 'Tahoma'}, margin: {t: 60, b: 55, l: 65, r: 25}, xaxis: {title: {text: 'قیمت'}}, yaxis: {title: {text: 'چگالی'}}, shapes: [chosen.lower_price, chosen.upper_price].map(p => ({type: 'line', x0: p, x1: p, yref: 'paper', y0: 0, y1: 1, line: {dash: 'dot', color: '#f7c979'}}))}, config);
    $('method').textContent = chosenEmpirical ? `محدودهٔ ${percent(payload.confidence)} و توزیع از همان ${number(chosen.scenario_prices.length)} مسیر نمونه‌گیری‌شده ساخته شده‌اند. خط‌چین قیمت مبناست. با افزایش افق، خطاها در مسیر قیمت انباشته می‌شوند؛ پهنای بازه الزاماً در تک‌تک ساعت‌ها افزایش یکنواخت ندارد.` : 'این خروجی متعلق به مدل قدیمی است و توزیع قیمت از تقریب لگ‌نرمال محاسبه می‌شود. برای مشاهدهٔ سناریوهای تجربی، مدل بازده ساعتی را آموزش و انتخاب کنید.';
    $('table').replaceChildren(...visible.map((r, i) => { const tr = document.createElement('tr'); for (const value of [forecastLabel(r), number(r.predicted_price), percent(r.predicted_return), percent(hourlyReturn(state.start + i)), number(r.lower_price), number(r.upper_price), percent(r.probability_up)]) { const td = document.createElement('td'); td.textContent = value; tr.append(td); } return tr; }));
    const next = new URL(location.href);
    next.searchParams.set('forecast_from', state.start); next.searchParams.set('forecast_to', state.end); next.searchParams.set('forecast_point', state.point);
    history.replaceState(null, '', next);
  }
  function change(which, value) {
    let {start, end, point} = state;
    if (which === 'start') {start = Number(value); end = Math.max(start, end);}
    if (which === 'end') {end = Number(value); start = Math.min(start, end);}
    if (which === 'point') point = Number(value);
    state = forecastWindow(rows, start, end, point); render();
  }
  for (const which of ['start', 'end']) for (const suffix of ['', '-select']) $(which + suffix).addEventListener('input', e => change(which, e.target.value));
  $('point').addEventListener('change', e => change('point', e.target.value));
  $('reset').addEventListener('click', () => {state = forecastWindow(rows, 0, rows.length - 1, state.point); render();});
  for (const id of ['band', 'scenarios', 'return-mode']) $(id).addEventListener('change', render);
  render();
})();
