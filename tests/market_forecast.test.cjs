const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {forecastWindow} = require('../forecast/static/forecast/market_forecast.js');
const rows = Array.from({length: 84}, (_, i) => ({horizon: i + 1, target_day: 1 + Math.floor(i/4), target_hour: 9 + i%4, current_price: 100, mu_log: .001*(i+1), sigma_log: .02, predicted_price: 100+i, predicted_return: i/100, lower_price: 95+i, upper_price: 105+i, probability_up: .6, target_schema: 'hourly_returns_v2', scenario_prices: [95+i,100+i,105+i]}));
assert.equal(forecastWindow(rows, 4, 60, 80).point, 60);
assert.equal(forecastWindow(rows, -1, 999, 0).rows.length, 84);
assert.equal(forecastWindow(rows, 7, 7, 1).rows.length, 1);
class Element {
  constructor() {this.listeners={};this.children=[];this.checked=true;this.value=0;this.options=[{}];}
  replaceChildren(...children) {this.children=children;}
  append(...children) {this.children.push(...children);}
  addEventListener(name, fn) {this.listeners[name]=fn;}
  fire(name, value) {this.value=value;this.listeners[name]({target:this});}
}
const elements=new Map(), calls=new Map();
const get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
get('market-forecast-data').textContent=JSON.stringify({symbol:'fixture',confidence:.9,horizon:1,rows});
get('forecast-return-mode').value='hourly';
let current='http://localhost/market/?symbol=fixture';
const context={document:{getElementById:get,createElement:()=>new Element(),createTextNode:text=>text},location:{href:current},history:{replaceState(a,b,url){current=String(url);}},URL,Plotly:{react(el,traces,layout){calls.set(el,{traces,layout});}}};
vm.runInNewContext(fs.readFileSync('forecast/static/forecast/market_forecast.js','utf8'),context);
assert.equal(calls.get(get('forecast-price')).traces.at(-2).x.length,84);
get('forecast-start').fire('input',10);
get('forecast-end-select').fire('input',20);
assert.equal(calls.get(get('forecast-price')).traces.at(-2).x.length,11);
assert.equal(get('forecast-table').children.length,11);
assert.equal(get('forecast-point').children.length,11);
get('forecast-point').fire('change',15);
assert.equal(calls.get(get('forecast-distribution')).traces[0].x[0],110);
get('forecast-band').checked=false;get('forecast-band').fire('change',0);
assert(!calls.get(get('forecast-price')).traces.some(t=>t.fill==='tonexty'));
get('forecast-reset').fire('click',0);
assert.equal(calls.get(get('forecast-price')).traces.at(-2).x.length,84);
assert(current.includes('forecast_to=83'));
console.log('Market slicer: chart, table, distribution, band toggle and reset passed');
