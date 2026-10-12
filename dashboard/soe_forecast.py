"""
STAGE 2 of 3 — FORECASTING
==========================
Takes the cleaned sales history from stage 1 and produces one forecast file per
target:

    forecast_revenue.json      baht booked per week
    forecast_units.json        pieces shipped per week

Five models, one protocol, identical for both targets:

    LSTM       recurrent neural network, written from scratch in NumPy
    LightGBM   histogram-based gradient boosted trees (scikit-learn)
    XGBoost    exact-greedy, second-order regularised boosting, from scratch
    SARIMA     seasonal ARIMA by conditional least squares, from scratch
    Ensemble   our own: the unweighted mean of the four above

    1. aggregate to weekly points (Apr 2021 - Mar 2025 -> 209 weeks)
    2. hold out the last 52 weeks, train on the rest, score on monthly totals
    3. retrain on everything and forecast the next 52 weeks (FY25)

Weekly rather than monthly because 48 monthly points is far too little to train on.

    python soe_forecast.py          # runs stage 1 first, then both targets
"""
import json, os
import numpy as np, pandas as pd


def _need(package, pip_name):
    """import a heavy dependency late, with an instruction instead of a stack trace"""
    try:
        return __import__(package, fromlist=['_'])
    except ImportError:
        raise SystemExit(f'\n  {package} is not installed in this environment.\n'
                         f'  run:  pip install {pip_name}\n')

H = 52                       # hold-out / forecast horizon, in weeks
BASE = ['LSTM', 'LightGBM', 'XGBoost', 'SARIMA']


# ============================================================ shared helpers
def _monthly(vals, idx):
    return pd.Series(vals, index=idx).groupby(pd.Series(idx).dt.to_period('M').values).sum()


def _score(actual, pred, idx):
    """MAPE and MAE, both measured on monthly totals rather than weekly points"""
    a, f = _monthly(actual, idx), _monthly(pred, idx)
    return float(np.mean(np.abs(a - f) / a) * 100), float(np.mean(np.abs(a - f)))


# ============================================================ LSTM (NumPy)
LOOK, NH = 26, 24            # 26 weeks of context, 24 hidden units


def _seasonal(idx):
    woy = idx.isocalendar().week.values.astype(float)
    return np.stack([np.sin(2 * np.pi * woy / 52.0), np.cos(2 * np.pi * woy / 52.0)], axis=1)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class LSTM:
    def __init__(self, nx, nh, seed=0):
        r = np.random.default_rng(seed); k = 1.0 / np.sqrt(nh); self.nh = nh
        self.Wx = r.uniform(-k, k, (nx, 4 * nh)); self.Wh = r.uniform(-k, k, (nh, 4 * nh))
        self.b = np.zeros(4 * nh); self.b[nh:2 * nh] = 1.0        # forget gate starts open
        self.Wy = r.uniform(-k, k, (nh, 1)); self.by = np.zeros(1)
        self.params = ['Wx', 'Wh', 'b', 'Wy', 'by']
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.v = {p: np.zeros_like(getattr(self, p)) for p in self.params}; self.t = 0

    def forward(self, X):
        B, T, _ = X.shape; nh = self.nh
        h = np.zeros((B, nh)); c = np.zeros((B, nh)); cache = []
        for t in range(T):
            x = X[:, t, :]; a = x @ self.Wx + h @ self.Wh + self.b
            i = _sigmoid(a[:, :nh]); f = _sigmoid(a[:, nh:2 * nh])
            o = _sigmoid(a[:, 2 * nh:3 * nh]); g = np.tanh(a[:, 3 * nh:])
            c_prev, h_prev = c, h; c = f * c_prev + i * g; tc = np.tanh(c); h = o * tc
            cache.append((x, h_prev, c_prev, i, f, o, g, c, tc))
        return h @ self.Wy + self.by, cache

    def backward(self, cache, dy):
        nh = self.nh; g_ = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        g_['Wy'] += (cache[-1][8] * cache[-1][5]).T @ dy; g_['by'] += dy.sum(0)
        dh = dy @ self.Wy.T; dc = np.zeros_like(dh)
        for t in range(len(cache) - 1, -1, -1):
            x, h_prev, c_prev, i, f, o, g, c, tc = cache[t]
            do = dh * tc; dc = dc + dh * o * (1 - tc ** 2)
            di, dg, df = dc * g, dc * i, dc * c_prev; dc_prev = dc * f
            da = np.concatenate([di * i * (1 - i), df * f * (1 - f),
                                 do * o * (1 - o), dg * (1 - g ** 2)], axis=1)
            g_['Wx'] += x.T @ da; g_['Wh'] += h_prev.T @ da; g_['b'] += da.sum(0)
            dh = da @ self.Wh.T; dc = dc_prev
        return g_

    def step(self, g_, lr=0.01, b1=.9, b2=.999, eps=1e-8, clip=5.0):
        self.t += 1
        for p in self.params:
            gp = g_[p]; n = np.linalg.norm(gp)
            if n > clip: gp = gp * (clip / n)
            self.m[p] = b1 * self.m[p] + (1 - b1) * gp
            self.v[p] = b2 * self.v[p] + (1 - b2) * gp ** 2
            mh = self.m[p] / (1 - b1 ** self.t); vh = self.v[p] / (1 - b2 ** self.t)
            setattr(self, p, getattr(self, p) - lr * mh / (np.sqrt(vh) + eps))


def _lstm_train(zs, ss, seed=0, epochs=300, val_frac=0.15):
    X = np.array([np.concatenate([zs[i:i + LOOK, None], ss[i:i + LOOK]], axis=1)
                  for i in range(len(zs) - LOOK)])
    Y = np.array([zs[i + LOOK] for i in range(len(zs) - LOOK)])[:, None]
    nval = max(8, int(len(X) * val_frac))
    Xtr, Ytr, Xva, Yva = X[:-nval], Y[:-nval], X[-nval:], Y[-nval:]
    net = LSTM(3, NH, seed=seed); r = np.random.default_rng(seed); idx = np.arange(len(Xtr))
    best, best_w, bad = np.inf, None, 0
    for _ in range(epochs):                                   # early stopping on a tail split
        r.shuffle(idx)
        for s in range(0, len(idx), 16):
            b = idx[s:s + 16]; yh, cache = net.forward(Xtr[b])
            net.step(net.backward(cache, 2 * (yh - Ytr[b]) / len(b)))
        vmse = float(np.mean((net.forward(Xva)[0] - Yva) ** 2))
        if vmse < best - 1e-5:
            best, bad = vmse, 0; best_w = {p: getattr(net, p).copy() for p in net.params}
        else:
            bad += 1
            if bad >= 40: break
    for p, w in (best_w or {}).items(): setattr(net, p, w)
    return net


def _lstm_forecast(nets, zs, ss, future_dates, steps, mu, sd):
    outs = []
    for net in nets:
        hist = list(zs[-LOOK:]); shist = list(ss[-LOOK:]); sfut = _seasonal(future_dates); o = []
        for k in range(steps):                                # recursive, one week at a time
            X = np.concatenate([np.array(hist)[:, None], np.array(shist)], axis=1)[None, :, :]
            v = float(net.forward(X)[0][0, 0]); o.append(v)
            hist = hist[1:] + [v]; shist = shist[1:] + [sfut[k]]
        outs.append(np.expm1(np.array(o) * sd + mu))
    return np.mean(outs, axis=0)                              # 5-seed ensemble


# ============================================================ tree models
LAGS = [1, 2, 3, 4, 6, 8, 12, 26, 52]


def _tree_features(series, i, when):
    f = [series[i - l] for l in LAGS]
    f += [np.mean(series[i - 4:i]), np.mean(series[i - 12:i]), np.std(series[i - 12:i])]
    woy = when.isocalendar()[1]
    return f + [np.sin(2 * np.pi * woy / 52.0), np.cos(2 * np.pi * woy / 52.0), float(i)]


def _tree_matrix(series, idx):
    start = max(LAGS)
    X = np.array([_tree_features(series, i, idx[i]) for i in range(start, len(series))])
    Y = np.array([series[i] for i in range(start, len(series))])
    return X, Y


def _tree_forecast(predict, series, future_dates, steps):
    s = list(series); out = []
    for k in range(steps):
        out.append(float(predict(np.array([_tree_features(s, len(s), future_dates[k])]))[0]))
        s.append(out[-1])
    return np.array(out)


def _lgb_fit(X, Y):
    HistGradientBoostingRegressor = _need('sklearn.ensemble', 'scikit-learn').HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=8, l2_regularization=1.0,
                                         random_state=0).fit(X, Y).predict


class _Node:
    __slots__ = ('feat', 'thr', 'left', 'right', 'value')


class XGBoost:
    """Exact-greedy regularised boosting. Every split maximises
           gain = 1/2 [ GL^2/(HL+L) + GR^2/(HR+L) - G^2/(H+L) ] - gamma
       and each leaf gets weight -G/(H+L). Squared-error loss, so gradient = pred - y
       and hessian = 1."""

    def __init__(self, n_estimators=300, learning_rate=0.05, max_depth=4, reg_lambda=1.0,
                 gamma=0.0, min_child_weight=3.0, subsample=0.9, colsample=0.9, seed=0):
        self.n, self.lr, self.md = n_estimators, learning_rate, max_depth
        self.lam, self.gamma, self.mcw = reg_lambda, gamma, min_child_weight
        self.sub, self.col, self.seed = subsample, colsample, seed
        self.trees, self.base = [], 0.0

    def _build(self, X, g, h, rows, feats, depth):
        node = _Node(); G, Hh = g[rows].sum(), h[rows].sum()
        node.value = -G / (Hh + self.lam); node.feat = None
        if depth >= self.md or len(rows) < 2 or Hh < 2 * self.mcw:
            return node
        best_gain, best, parent = 0.0, None, G * G / (Hh + self.lam)
        for f in feats:
            v = X[rows, f]; order = np.argsort(v, kind='stable')
            vs, gs, hs = v[order], g[rows][order], h[rows][order]
            GL = np.cumsum(gs)[:-1]; HL = np.cumsum(hs)[:-1]; GR, HR = G - GL, Hh - HL
            ok = (vs[:-1] < vs[1:]) & (HL >= self.mcw) & (HR >= self.mcw)
            if not ok.any(): continue
            gain = 0.5 * (GL * GL / (HL + self.lam) + GR * GR / (HR + self.lam) - parent) - self.gamma
            gain = np.where(ok, gain, -np.inf); k = int(np.argmax(gain))
            if gain[k] > best_gain:
                best_gain, best = float(gain[k]), (f, (vs[k] + vs[k + 1]) / 2.0)
        if best is None: return node
        f, thr = best; mask = X[rows, f] <= thr
        node.feat, node.thr = f, thr
        node.left = self._build(X, g, h, rows[mask], feats, depth + 1)
        node.right = self._build(X, g, h, rows[~mask], feats, depth + 1)
        return node

    @staticmethod
    def _apply(node, X):
        out = np.empty(len(X))
        for i, row in enumerate(X):
            n = node
            while n.feat is not None:
                n = n.left if row[n.feat] <= n.thr else n.right
            out[i] = n.value
        return out

    def fit(self, X, Y):
        rng = np.random.default_rng(self.seed)
        self.base = float(np.mean(Y)); pred = np.full(len(Y), self.base); n_feat = X.shape[1]
        for _ in range(self.n):
            g = pred - Y; h = np.ones_like(g); rows = np.arange(len(Y))
            if self.sub < 1:
                rows = rows[rng.random(len(Y)) < self.sub]
                if len(rows) < 5: rows = np.arange(len(Y))
            feats = np.arange(n_feat)
            if self.col < 1:
                feats = rng.choice(n_feat, max(2, int(n_feat * self.col)), replace=False)
            tree = self._build(X, g, h, rows, feats, 0)
            self.trees.append(tree); pred = pred + self.lr * self._apply(tree, X)
        return self

    def predict(self, X):
        out = np.full(len(X), self.base)
        for t in self.trees:
            out += self.lr * self._apply(t, X)
        return out


# ============================================================ SARIMA
def _diff(x, d, D, s):
    for _ in range(D): x = x[s:] - x[:-s]
    for _ in range(d): x = x[1:] - x[:-1]
    return x


def _undiff(fc, d, D, s, orig):
    out, series = [], list(orig)
    for v in fc:
        val = v
        for _ in range(d): val += series[-1]
        for _ in range(D): val += series[-s]
        out.append(val); series.append(val)
    return np.array(out)


class SARIMA:
    """SARIMA(p,d,q)(P,D,Q,s) by conditional least squares: difference the series, run the
       ARMA residual recursion, minimise the sum of squared residuals."""

    def __init__(self, order=(1, 1, 1), seasonal=(1, 0, 0, 52)):
        self.p, self.d, self.q = order
        self.P, self.D, self.Q, self.s = seasonal

    def _resid(self, params, w):
        p, q, P, Q, s = self.p, self.q, self.P, self.Q, self.s
        phi, th = params[:p], params[p:p + q]
        Phi, Th, c = params[p + q:p + q + P], params[p + q + P:p + q + P + Q], params[-1]
        n = len(w); e = np.zeros(n)
        start = max(p, q, P * s, Q * s, s if P or Q else 0)
        for t in range(start, n):
            pred = c
            for i in range(p): pred += phi[i] * w[t - 1 - i]
            for i in range(P): pred += Phi[i] * w[t - (i + 1) * s]
            for i in range(q): pred += th[i] * e[t - 1 - i]
            for i in range(Q): pred += Th[i] * e[t - (i + 1) * s]
            e[t] = w[t] - pred
        return e, start

    def fit(self, series):
        self.orig = np.asarray(series, float)
        self.w = w = _diff(self.orig.copy(), self.d, self.D, self.s)
        # a constant on a differenced series is a drift term; estimated from noisy weekly
        # data it extrapolates badly over 52 steps, so keep it only for the stationary form
        self.use_const = (self.d + self.D) == 0
        n_par = self.p + self.q + self.P + self.Q + 1
        x0 = np.zeros(n_par); x0[-1] = w.mean() if self.use_const else 0.0
        if self.p: x0[0] = 0.3
        if self.P: x0[self.p + self.q] = 0.3
        bounds = [(-0.99, 0.99)] * (n_par - 1) + [(None, None) if self.use_const else (0.0, 0.0)]

        minimize = _need('scipy.optimize', 'scipy').minimize

        def sse(params):
            e, start = self._resid(params, w)
            return float(np.sum(e[start:] ** 2))

        self.params = minimize(sse, x0, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 500}).x
        self.e, self.start = self._resid(self.params, w)
        return self

    def forecast(self, steps):
        p, q, P, Q, s = self.p, self.q, self.P, self.Q, self.s
        phi, th = self.params[:p], self.params[p:p + q]
        Phi, Th, c = self.params[p + q:p + q + P], self.params[p + q + P:p + q + P + Q], self.params[-1]
        w, e, out = list(self.w), list(self.e), []
        for _ in range(steps):
            t = len(w); pred = c
            for i in range(p): pred += phi[i] * w[t - 1 - i]
            for i in range(P): pred += Phi[i] * w[t - (i + 1) * s]
            for i in range(q): pred += th[i] * e[t - 1 - i]
            for i in range(Q): pred += Th[i] * e[t - (i + 1) * s]
            out.append(pred); w.append(pred); e.append(0.0)
        return _undiff(np.array(out), self.d, self.D, self.s, self.orig)


SPECS = [((2, 0, 2), (1, 0, 0, 52)), ((1, 0, 1), (1, 0, 0, 52)), ((2, 0, 1), (1, 0, 0, 52)),
         ((0, 1, 1), (0, 0, 0, 52)), ((1, 1, 1), (0, 0, 0, 52)), ((1, 1, 1), (1, 0, 0, 52))]


def _sarima_forecast(series, steps, order, seasonal):
    return SARIMA(order, seasonal).fit(series).forecast(steps)


# ============================================================ the pipeline
def weekly_series(sales, target):
    """cleaned order lines -> one number per week (baht in millions, or units)"""
    col = 'Revenue' if target == 'revenue' else 'Quantity'
    daily = sales.groupby('Sales Order Date')[col].sum()
    daily = daily.reindex(pd.date_range(sales['Sales Order Date'].min(),
                                        sales['Sales Order Date'].max(), freq='D'), fill_value=0.0)
    weekly = daily.resample('W-SUN').sum()
    weekly = weekly[weekly.index <= sales['Sales Order Date'].max()]
    y = weekly.values.astype(float)
    return (y / 1e6 if target == 'revenue' else y), weekly.index


def forecast(sales, target='revenue', verbose=True):
    """train, backtest and forecast every model for one target; returns the dashboard dict"""
    y, dates = weekly_series(sales, target)
    rnd = (lambda v: round(float(v), 2)) if target == 'revenue' else (lambda v: int(round(float(v))))
    test_dates = dates[-H:]
    fut_dates = pd.date_range(dates[-1] + pd.Timedelta(days=7), periods=H, freq='W-SUN')
    say = print if verbose else (lambda *a, **k: None)
    say(f'\n--- {target}: {len(y)} weeks  {dates[0].date()} -> {dates[-1].date()}')

    bt, fut = {}, {}

    # LSTM on the log-scaled series, 5 seeds averaged
    ylog = np.log1p(y); mu, sd = ylog.mean(), ylog.std(); z = (ylog - mu) / sd
    S = _seasonal(dates)
    nets = [_lstm_train(z[:-H], S[:-H], seed=s) for s in (1, 2, 3, 4, 5)]
    bt['LSTM'] = _lstm_forecast(nets, z[:-H], S[:-H], test_dates, H, mu, sd)
    nets = [_lstm_train(z, S, seed=s) for s in (1, 2, 3, 4, 5)]
    fut['LSTM'] = _lstm_forecast(nets, z, S, fut_dates, H, mu, sd)

    # the two tree models, trained on the log series
    zl = np.log1p(y)
    for name, fitter in [('LightGBM', _lgb_fit), ('XGBoost', lambda X, Y: XGBoost().fit(X, Y).predict)]:
        X, Y = _tree_matrix(zl[:-H], dates[:-H])
        bt[name] = np.expm1(_tree_forecast(fitter(X, Y), zl[:-H], test_dates, H))
        X, Y = _tree_matrix(zl, dates)
        fut[name] = np.expm1(_tree_forecast(fitter(X, Y), zl, fut_dates, H))

    # SARIMA — the specification is picked on an inner split, never on the test year
    V = 26
    inner_test, inner_dates = y[-H - V:-H], dates[-H - V:-H]
    best_spec, best_err = None, np.inf
    for order, seas in SPECS:
        try:
            f = _sarima_forecast(y[:-H - V], V, order, seas)
            err = _score(inner_test, np.maximum(f, 0), inner_dates)[0]
            if err < best_err: best_err, best_spec = err, (order, seas)
        except Exception:
            pass
    say(f'    SARIMA spec {best_spec[0]}{best_spec[1]}  (inner validation {best_err:.1f}%)')
    bt['SARIMA'] = np.maximum(_sarima_forecast(y[:-H], H, *best_spec), 0)
    fut['SARIMA'] = np.maximum(_sarima_forecast(y, H, *best_spec), 0)

    # model 5 — our own: no extra training, just the mean of the four above
    bt['Ensemble'] = np.mean([bt[k] for k in BASE], axis=0)
    fut['Ensemble'] = np.mean([fut[k] for k in BASE], axis=0)

    engines = {'LSTM': 'LSTM (NumPy implementation)',
               'LightGBM': 'LightGBM (sklearn HistGradientBoosting)',
               'XGBoost': 'XGBoost (from-scratch implementation)',
               'SARIMA': f'SARIMA (conditional least squares) {best_spec[0]}{best_spec[1]}',
               'Ensemble': 'Ensemble (unweighted mean of the four models)'}
    models, scores = {}, {}
    for k in bt:
        mape, mae = _score(y[-H:], bt[k], test_dates)
        scores[k] = (mape, mae)
        models[k] = {'engine': engines[k], 'mape': round(mape, 1), 'mae': rnd(mae),
                     'forecast_values_m': [rnd(v) for v in _monthly(fut[k], fut_dates).values],
                     'backtest_pred_monthly': [rnd(v) for v in _monthly(bt[k], test_dates).values]}
        unit = '฿M' if target == 'revenue' else 'units'
        say(f'    {k:10s} off {mape:5.1f}%   MAE {mae:8.1f} {unit}/month   '
            f'FY25 {sum(models[k]["forecast_values_m"]):,.0f}')

    out = {'weeks': len(y),
           'fy24_actual_m': rnd(y[-H:].sum()),
           'forecast_labels': [str(p) for p in _monthly(fut['LSTM'], fut_dates).index],
           'backtest_labels': [str(p) for p in _monthly(y[-H:], test_dates).index],
           'backtest_actual_monthly': [rnd(v) for v in _monthly(y[-H:], test_dates).values],
           'backtest': [{'name': k, 'mape': round(v[0], 1), 'mae': rnd(v[1])} for k, v in scores.items()],
           'models': models}
    # keys the dashboard's older code still reads
    out['forecast_values_m'] = models['LSTM']['forecast_values_m']
    out['backtest_pred_monthly'] = models['LSTM']['backtest_pred_monthly']
    out['gbdt'] = models['LightGBM']
    return out


def main(folder='.'):
    import soe_clean
    sales = soe_clean.clean_all(folder)['sales']
    paths = {}
    for target in ('revenue', 'units'):
        out = forecast(sales, target)
        p = os.path.join(folder, f'forecast_{target}.json')
        json.dump(out, open(p, 'w'), indent=1)
        paths[target] = p
        print(f'    -> {os.path.basename(p)}')
    return paths


if __name__ == '__main__':
    main(os.path.dirname(os.path.abspath(__file__)) or '.')
