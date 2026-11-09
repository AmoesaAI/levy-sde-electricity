"""Identifiability vs estimator variance (Appendix I, Table alpha_vs_n).
Same Levy-OU simulator, AR(1) drift removal and joint (alpha, sigma) MLE objective as
sensititvity.ipynb, repeated over sample sizes n. The stable log-density is evaluated from a
precomputed scipy table with bicubic interpolation (fastlik.py), which reproduces the exact
scipy MLE to 4 decimals but is ~3000x faster. Run: python run_ident.py  (~5 min)."""
import json, time, numpy as np, pandas as pd, sys
from scipy.stats import pearsonr
sys.path.insert(0, '.')
from fastlik import FastStableLogpdf, fit_alpha_sigma_fast
# Reuses the ORIGINAL simulator and AR(1)-residual step from sensititvity.ipynb (cell 1)
nb = json.load(open('sensititvity.ipynb'))
src = ''.join(nb['cells'][1]['source']).split('# ============================================================\n# Run the full grid')[0]
ns = {}; exec(src, ns)                      # original simulator, AR(1) residuals, constants
import os
from fastlik import build_table
if not os.path.exists("stable_logpdf_table.npy"):          # one-off, ~40 s
    np.save("stable_logpdf_table.npy", build_table())
fl = FastStableLogpdf(np.load("stable_logpdf_table.npy"))
THETA, SIGMA = ns['THETA_TRUE'], ns['SIGMA_TRUE']
ALPHA_TRUE = [1.2, 1.5, 1.8]
N_GRID = [300, 1200, 4800, 19200]
N_SEEDS = 40
rows = []
t0 = time.time()
for a in ALPHA_TRUE:
    for n in N_GRID:
        ae, se = [], []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed * 1000 + int(a * 100))    # same seeding rule as the original
            x = ns['simulate_ou_levy_fast'](THETA, SIGMA, a, 0.0, 1.0, n, rng)
            ah, sh = fit_alpha_sigma_fast(ns['ar1_residuals'](x), fl)
            ae.append(ah - a); se.append(sh - SIGMA)
        ae, se = np.array(ae), np.array(se)
        r, p = pearsonr(ae, se)
        rows.append({"alpha_true": a, "n": n, "bias_alpha": ae.mean(), "std_alpha": ae.std(ddof=1),
                     "std_alpha_x_sqrt_n": ae.std(ddof=1) * np.sqrt(n), "std_sigma": se.std(ddof=1),
                     "r_alpha_sigma": r, "p_r": p, "frac_at_2": np.mean(np.abs(ae + a - 2.0) < 0.005)})
        print(f"alpha={a} n={n}: std(alpha_hat)={ae.std(ddof=1):.4f}  sqrt(n)*std={ae.std(ddof=1)*np.sqrt(n):.3f}  r={r:+.3f}  [{time.time()-t0:.0f}s]", flush=True)
df = pd.DataFrame(rows); df.to_csv("identifiability_vs_n.csv", index=False)
for a in ALPHA_TRUE:   # log-log slope of std vs n: -0.5 means 1/sqrt(n)
    d = df[df.alpha_true == a]
    slope = np.polyfit(np.log(d.n), np.log(d.std_alpha), 1)[0]
    print(f"alpha={a}: log-log slope of std(alpha_hat) vs n = {slope:.3f}")
print(df.round(4).to_string(index=False))
