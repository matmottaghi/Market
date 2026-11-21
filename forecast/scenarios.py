"""Reproducible branching price scenarios from held-out hourly errors."""
import numpy as np

def scenario_paths(hourly_returns, residuals, price, seed=42, count=256):
    rng=np.random.default_rng(seed)
    residuals=np.asarray(residuals,float)
    errors=np.zeros((count,len(hourly_returns)))
    for j in range(errors.shape[1]):
        values=residuals[:,j];values=values[np.isfinite(values)]
        if not len(values):continue
        errors[:,j]=rng.choice(values-values.mean(),size=count)
    return price*np.exp(np.cumsum(np.asarray(hourly_returns)[None,:]+errors,axis=1))

def scenario_summary(prices,price,confidence=.9):
    values=np.asarray(prices,float);alpha=(1-confidence)/2
    up=float(np.mean(values>price))
    return dict(predicted_price=float(np.median(values)),predicted_return=float(np.median(values)/price-1),lower_price=float(np.quantile(values,alpha)),upper_price=float(np.quantile(values,1-alpha)),probability_up=up,probability_down=1-up,confidence=confidence,scenario_prices=values.tolist(),distribution='empirical_scenario_tree')
