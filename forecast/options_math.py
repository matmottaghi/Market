# توضیح فایل: سازندهٔ آزاد legs و کاتالوگ استراتژی؛ محاسبهٔ هزینه، سود سررسید و امید سود تحت توزیع پیش‌بینی فیزیکی.
# evaluate_strategy قیمت هر leg را با bid/ask یا override صریح می‌گیرد؛ چندسررسید فقط با IV و تاریخ ارزیابی معتبر قیمت‌گذاری می‌شود.
from datetime import date
from math import lcm
import numpy as np
from scipy.stats import norm

STRATEGIES={
 'long_call':('خرید کال',[('call',1,0)]), 'short_call':('فروش کال',[('call',-1,0)]),
 'long_put':('خرید پوت',[('put',1,0)]),'short_put':('فروش پوت',[('put',-1,0)]),
 'covered_call':('کال پوشش‌دار',[('stock',1,0),('call',-1,0)]),
 'protective_put':('پوت حمایتی',[('stock',1,0),('put',1,0)]),
 'bull_call':('اسپرد صعودی کال',[('call',1,-1),('call',-1,1)]),
 'bear_call':('اسپرد نزولی کال',[('call',-1,-1),('call',1,1)]),
 'bull_put':('اسپرد صعودی پوت',[('put',1,-1),('put',-1,1)]),
 'bear_put':('اسپرد نزولی پوت',[('put',-1,-1),('put',1,1)]),
 'straddle':('استرادل خرید',[('call',1,0),('put',1,0)]),
 'short_straddle':('استرادل فروش',[('call',-1,0),('put',-1,0)]),
 'strangle':('استرانگل خرید',[('put',1,-1),('call',1,1)]),
 'short_strangle':('استرانگل فروش',[('put',-1,-1),('call',-1,1)]),
 'butterfly_call':('پروانه کال',[('call',1,-1),('call',-2,0),('call',1,1)]),
 'butterfly_put':('پروانه پوت',[('put',1,-1),('put',-2,0),('put',1,1)]),
 'iron_butterfly':('پروانه آهنی',[('put',1,-1),('put',-1,0),('call',-1,0),('call',1,1)]),
 'iron_condor':('کندور آهنی',[('put',1,-2),('put',-1,-1),('call',-1,1),('call',1,2)]),
 'condor_call':('کندور کال',[('call',1,-2),('call',-1,-1),('call',-1,1),('call',1,2)]),
 'collar':('کالر',[('stock',1,0),('put',1,-1),('call',-1,1)]),
 'synthetic_long':('خرید مصنوعی',[('call',1,0),('put',-1,0)]),
 'synthetic_short':('فروش مصنوعی',[('call',-1,0),('put',1,0)]),
 'call_backspread':('بک‌اسپرد کال',[('call',-1,-1),('call',2,1)]),
 'put_backspread':('بک‌اسپرد پوت',[('put',2,-1),('put',-1,1)]),
 'call_ratio':('نسبتی کال',[('call',1,-1),('call',-2,1)]),
 'put_ratio':('نسبتی پوت',[('put',-2,-1),('put',1,1)]),
 'conversion':('تبدیل',[('stock',1,0),('put',1,0),('call',-1,0)]),
 'reversal':('تبدیل معکوس',[('stock',-1,0),('put',-1,0),('call',1,0)]),
 'calendar_call':('تقویمی کال؛ سررسیدهای جدا',[('call',-1,0),('call',1,0)]),
 'calendar_put':('تقویمی پوت؛ سررسیدهای جدا',[('put',-1,0),('put',1,0)]),
 'diagonal_call':('مورب کال؛ سررسیدهای جدا',[('call',-1,-1),('call',1,1)]),
 'custom':('ترکیب آزاد',[])
}

def template_legs(key,contracts,spot,expiry):
    specification=STRATEGIES[key][1]
    available=sorted({c['strike'] for c in contracts if c['expiry']==expiry})
    if not available and any(kind!='stock' for kind,_,_ in specification):raise ValueError('No contracts for selected expiry')
    atm=min(range(len(available)),key=lambda i:abs(available[i]-spot)) if available else 0
    balanced={}
    if key in ('butterfly_call','butterfly_put','iron_butterfly'):
        center=available[atm]
        pairs=[(low,high) for low in available if low<center for high in available if high>center and abs((center-low)-(high-center))<1e-8]
        if not pairs:raise ValueError('No symmetric strike wings; use custom legs for a broken-wing butterfly')
        low,high=min(pairs,key=lambda pair:pair[1]-pair[0]);balanced={-1:low,0:center,1:high}
    legs=[]; unit=next((c['multiplier'] for c in contracts if c['expiry']==expiry),1)
    expiries=sorted({c['expiry'] for c in contracts})
    for number,(kind,quantity,offset) in enumerate(specification):
        if kind=='stock':legs.append(dict(kind='stock',quantity=quantity*unit));continue
        index=available.index(balanced[offset]) if balanced and offset in balanced else atm+offset
        if not 0<=index<len(available):raise ValueError('Not enough strikes for this template; choose custom legs')
        target_expiry=expiry
        if key.startswith(('calendar','diagonal')) and number==1:
            later=[d for d in expiries if d>expiry]
            if not later:raise ValueError('A later expiry is required')
            target_expiry=later[0]
        matches=[c for c in contracts if c['expiry']==target_expiry and c['strike']==available[index] and c['kind']==kind]
        if not matches:raise ValueError('Required contract is absent; choose custom legs')
        legs.append(dict(contract=matches[0]['code'],quantity=quantity))
    sizes=[int(next(c['multiplier'] for c in contracts if c['code']==leg['contract'])) for leg in legs if 'contract' in leg]
    common=lcm(*sizes) if sizes else unit
    if common>1000000:raise ValueError('Contract sizes require an excessively large common lot; use custom legs')
    for leg in legs:
        if leg.get('kind')=='stock':leg['quantity']=int(leg['quantity']/unit*common)
        else:
            multiplier=next(c['multiplier'] for c in contracts if c['code']==leg['contract'])
            leg['quantity']=int(leg['quantity']*common/multiplier)
    return legs

def evaluate_strategy(legs,contracts,spot,evaluation_date,fees=0,distribution=None,capital=None,rate=0,dividend=0,underlying_book=None):
    if len(legs)>20 or not legs:raise ValueError('Use 1 to 20 legs')
    if not np.isfinite([spot,fees,rate,dividend]).all() or spot<=0 or fees<0:raise ValueError('Invalid prices or fees')
    if distribution:
        base,mu,sigma=map(float,distribution)
        if not np.isfinite([base,mu,sigma]).all() or base<=0 or sigma<=0 or abs(np.log(base))+abs(mu)+20*sigma>650:
            raise ValueError('Predictive distribution is outside the supported numerical range')
    evaluation=date.fromisoformat(evaluation_date)
    catalog={c['code']:c for c in contracts};normalized=[];entry=0
    for leg in legs:
        quantity=float(leg['quantity'])
        if not np.isfinite(quantity) or quantity==0 or quantity!=int(quantity) or abs(quantity)>1000000:raise ValueError('Quantity must be a nonzero integer')
        if leg.get('kind')=='stock':
            if 'price' in leg:
                price=float(leg['price'])
            else:
                book=underlying_book or {}
                price=float(book.get('ask' if quantity>0 else 'bid',0))
                available=float(book.get('ask_volume' if quantity>0 else 'bid_volume',0))
                if available<abs(quantity):raise ValueError('Underlying book volume insufficient; enter an explicit manual stock price')
                if book.get('bid',0)>0 and book.get('ask',0)>0 and book['bid']>book['ask']:raise ValueError('Crossed underlying quote')
            item=dict(kind='stock',quantity=quantity,multiplier=1,price=price)
        else:
            if leg.get('contract') not in catalog:raise ValueError('Unknown contract')
            contract=catalog[leg['contract']]
            if leg.get('price_mode')!='manual' and contract['bid']>0 and contract['ask']>0 and contract['bid']>contract['ask']:raise ValueError('Crossed bid/ask snapshot; refresh or enter an explicit manual scenario')
            if contract['kind'] not in ('call','put') or contract['strike']<=0 or not np.isfinite(contract['strike']):raise ValueError('Invalid contract specifications')
            if date.fromisoformat(contract['expiry'])<evaluation:raise ValueError('A leg expires before evaluation; path-dependent settlement is unsupported')
            if date.fromisoformat(contract['expiry'])>evaluation and not leg.get('iv'):raise ValueError('Calendar/early valuation requires explicit IV for unexpired legs')
            price=float(leg['price']) if leg.get('price_mode')=='manual' else float(contract['ask'] if quantity>0 else contract['bid'])
            available=float(contract.get('ask_volume' if quantity>0 else 'bid_volume',0))
            if leg.get('price_mode')!='manual' and available<abs(quantity):raise ValueError('Displayed quote volume is insufficient for requested contracts')
            item=dict(contract,quantity=quantity,price=price,iv=float(leg.get('iv',0)))
        if not np.isfinite(price) or price<=0:raise ValueError('Executable price missing; zero/last price is not a fill. Enter an explicit manual price if needed.')
        if item['multiplier']<=0:raise ValueError('Unknown contract multiplier')
        item['entry_value']=quantity*item['multiplier']*price
        entry+=item['entry_value'];normalized.append(item)
    cost=entry+fees

    def pnl(s):
        result=np.zeros_like(np.asarray(s,dtype=float))-cost
        for item in normalized:
            if item['kind']=='stock':value=s
            else:
                remaining=(date.fromisoformat(item['expiry'])-evaluation).days/365
                strike=item['strike']
                if remaining<=0:value=np.maximum(s-strike,0) if item['kind']=='call' else np.maximum(strike-s,0)
                else:
                    vol=item['iv']
                    if not 0<vol<=5:raise ValueError('IV must be an annual decimal between 0 and 5')
                    d1=(np.log(np.maximum(s,1e-12)/strike)+(rate-dividend+.5*vol*vol)*remaining)/(vol*np.sqrt(remaining));d2=d1-vol*np.sqrt(remaining)
                    value=s*np.exp(-dividend*remaining)*norm.cdf(d1)-strike*np.exp(-rate*remaining)*norm.cdf(d2) if item['kind']=='call' else strike*np.exp(-rate*remaining)*norm.cdf(-d2)-s*np.exp(-dividend*remaining)*norm.cdf(-d1)
            result+=item['quantity']*item['multiplier']*value
        return result
    low,high=spot*.25,spot*2
    if distribution:
        base,mu,sigma=distribution
        low=min(low,base*np.exp(mu-4*sigma));high=max(high,base*np.exp(mu+4*sigma))
    x=np.unique(np.r_[np.linspace(max(low,1e-6),high,600),[c['strike'] for c in normalized if c['kind']!='stock']]);values=pnl(x)
    result=dict(entry_debit=cost,entry_credit=max(-cost,0),legs=normalized,x=x.tolist(),pnl=values.tolist(),fees=fees)
    if distribution:
        base,mu,sigma=map(float,distribution)
        if base<=0 or sigma<=0 or not np.isfinite([base,mu,sigma]).all():raise ValueError('Invalid predictive distribution')
        # Gauss-Hermite quadrature under physical forecast probabilities; not risk-neutral option pricing.
        nodes,weights=np.polynomial.hermite.hermgauss(96)
        prices=base*np.exp(mu+np.sqrt(2)*sigma*nodes);profits=pnl(prices)
        expected=float(np.sum(weights*profits)/np.sqrt(np.pi))
        if all(c['kind']=='stock' or date.fromisoformat(c['expiry'])==evaluation for c in normalized):
            mean=base*np.exp(mu+.5*sigma*sigma)
            expected=-cost
            for item in normalized:
                if item['kind']=='stock':value=mean
                else:
                    d2=(np.log(base/item['strike'])+mu)/sigma;d1=d2+sigma
                    value=mean*norm.cdf(d1)-item['strike']*norm.cdf(d2) if item['kind']=='call' else item['strike']*norm.cdf(-d2)-mean*norm.cdf(-d1)
                expected+=item['quantity']*item['multiplier']*value
        result.update(expected_profit=expected,profit_at_expected_price=float(pnl(np.array([base*np.exp(mu+.5*sigma*sigma)]))[0]),predicted_price=base*np.exp(mu+.5*sigma*sigma))
        # Probability of profit via deterministic normal quantiles to avoid quadrature sign instability.
        quantiles=(np.arange(20000)+.5)/20000
        result['probability_profit']=float(np.mean(pnl(base*np.exp(mu+sigma*norm.ppf(quantiles)))>0))
        result['probability_profit_percent']=100*result['probability_profit']
        if capital is not None:
            capital=float(capital)
            if capital<=0 or not np.isfinite(capital):raise ValueError('Capital must be positive')
            result['expected_roi']=expected/capital
            result['expected_roi_percent']=100*expected/capital
    slope=sum(c['quantity']*c['multiplier']*(np.exp(-dividend*max(0,(date.fromisoformat(c['expiry'])-evaluation).days/365)) if c['kind']=='call' else 1) for c in normalized if c['kind'] in ('stock','call'))
    result['unbounded_upside_loss']=slope < -1e-10
    # Grid extrema are labelled scenario values; no claim that they are true global maximum/minimum.
    result['scenario_min']=float(values.min());result['scenario_max']=float(values.max())
    return result
