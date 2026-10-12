"""
STAGE 3 of 3 — BUILD THE DASHBOARD
==================================
Turns the cleaned tables (stage 1) and the forecast files (stage 2) into the finished
single-file dashboard.

    dashboard_template.html   the UI — markup, CSS and all the JavaScript, no data
    soe_dashboard.py          the analytics — every number the UI draws
    dashboard.html            template + data, written fresh on every run

The template carries one /*{{NAME}}*/ placeholder per dataset, so the UI can be edited
by hand without touching this file, and this file can change a calculation without
touching the UI.

    python soe_dashboard.py          # runs stages 1 and 2 first if needed
"""
import json, os
import numpy as np, pandas as pd
import soe_export

# ---------------------------------------------------------------- settings
AS_OF = None              # None = the last date in the data
ABC_CUT = (0.80, 0.95)    # share of revenue: A up to 80%, B up to 95%, C the rest
XYZ_CUT = (0.50, 1.00)    # coefficient of variation of monthly demand
VOL_CUT = (0.80, 0.95)    # same idea as ABC but on units shipped
MIN_ORDERS_REORDER = 3    # orders needed before a reorder cycle can be measured
REORDER_FLOOR = 70        # % of the usual cycle before a pair is worth listing
MISMATCH_DAYS = 180       # "still selling" window for discontinued items
FBT_MIN_SHARED = 5        # orders two items must share before the pair is reported
FBT_TOP = 300
# Average realised discount, measured off the sales history. Kept as a setting because
# the dashboard prices and the reorder values are quoted net of it.
DISCOUNT_ENDUSER_PCT = 15.1

DATASETS = ['inventoryData', 'QTY', 'QTY_OUT', 'AGG', 'mismatchItems', 'itemClassification',
            'customerData', 'reorderData', 'coOrderingPairs', 'LSTM_OUT']


def _fy(dates):
    """fiscal years run Apr-Mar and are named by the year they start in"""
    return np.where(dates.dt.month >= 4, dates.dt.year, dates.dt.year - 1)


# ---------------------------------------------------------------- item views
def inventory(item):
    """every SKU that carries a list price — the catalogue the dashboard shows"""
    df = item[item['Has List Price']].sort_values('Item Code')
    return [{'id': str(int(r['Item Internal ID'])), 'code': r['Item Code'],
             'cat1': r['Product Category 1'], 'cat2': r['Product Category 2'],
             'status': r['Production Status'], 'price': int(round(r['List price']))}
            for _, r in df.iterrows()]


def classification(sales, item):
    """Value Tier (ABC, by revenue) and Demand Pattern (XYZ, by how steady demand is)"""
    rev = sales.groupby('Item Internal ID')['Revenue'].sum().sort_values(ascending=False)
    before = (rev.cumsum() - rev) / rev.sum()          # revenue ranked ahead of this item
    abc = {int(i): ('A' if v < ABC_CUT[0] else 'B' if v < ABC_CUT[1] else 'C')
           for i, v in before.items()}

    m = sales.assign(ym=sales['Sales Order Date'].dt.to_period('M')) \
             .groupby(['Item Internal ID', 'ym'])['Quantity'].sum()
    stats = m.groupby(level=0).agg(['count', 'mean', 'std'])          # months with sales only
    xyz = {}
    for i, r in stats.iterrows():
        if r['count'] < 2:                                            # one month of history
            xyz[int(i)] = 'Insufficient Data'; continue
        cv = r['std'] / r['mean']
        xyz[int(i)] = 'X' if cv <= XYZ_CUT[0] else 'Y' if cv <= XYZ_CUT[1] else 'Z'

    return {str(int(i)): {'abc': abc.get(int(i), 'C'), 'xyz': xyz.get(int(i), 'Insufficient Data')}
            for i in item['Item Internal ID']}


def mismatches(sales, item, as_of):
    """items the master says are dead but that are still taking orders"""
    recent = sales[sales['Sales Order Date'] >= as_of - pd.Timedelta(days=MISMATCH_DAYS)]
    dead = item[item['Production Status'].isin(['Discontinued', 'To Be Advised'])
                & item['Has List Price']]
    rev = recent[recent['Item Internal ID'].isin(dead['Item Internal ID'])] \
        .groupby('Item Internal ID')['Revenue'].sum()
    rev = rev[rev > 0].sort_values(ascending=False)
    by_id = dead.set_index('Item Internal ID')
    return [{'id': str(int(i)), 'code': by_id.at[i, 'Item Code'],
             'cat1': by_id.at[i, 'Product Category 1'],
             'status': by_id.at[i, 'Production Status'],
             'recent_revenue_6mo': int(round(v))} for i, v in rev.items()]


# ---------------------------------------------------------------- reorder cycles
def reorders(sales, item, as_of):
    """one row per customer-product pair that is due, or past due, for a repeat order.

    The cycle is measured per pair from that customer's own ordering rhythm: the gaps
    between their orders for that item. The median gap drives the status (it ignores the
    one-off long gap a mean would be dragged by); the mean is reported as 'usually every'.
    """
    price = dict(zip(item['Item Internal ID'], item['List price']))
    code = dict(zip(item['Item Internal ID'], item['Item Code']))
    cat = dict(zip(item['Item Internal ID'], item['Product Category 1']))
    name = sales.drop_duplicates('Customer Internal ID') \
                .set_index('Customer Internal ID')['Company Name'].to_dict()

    per_order = sales.groupby(['Customer Internal ID', 'Item Internal ID', 'Sales Order Number']) \
                     .agg(d=('Sales Order Date', 'min'), q=('Quantity', 'sum')).reset_index()

    rows = []
    for (cid, iid), g in per_order.groupby(['Customer Internal ID', 'Item Internal ID'], sort=False):
        if len(g) < MIN_ORDERS_REORDER:
            continue
        d = np.sort(g['d'].values)
        gaps = np.diff(d).astype('timedelta64[D]').astype(float)
        cycle = float(np.median(gaps))
        if cycle <= 0:
            continue
        days = int((as_of - pd.Timestamp(d[-1])).days)
        pct = float(round(days / cycle * 100))
        if pct < REORDER_FLOOR:
            continue
        rows.append({'custId': int(cid), 'custName': name.get(cid, ''), 'itemId': int(iid),
                     'itemCode': code.get(iid, ''), 'cat1': cat.get(iid, ''),
                     'daysSince': days, 'progressPct': pct,
                     'status': ('Approaching Reorder' if pct < 100 else
                                'Expected Now' if pct < 130 else 'Overdue'),
                     'estValue': int(round(float(g['q'].mean()) * price.get(iid, 0)))})
    rows.sort(key=lambda r: -r['estValue'])
    return rows


# ---------------------------------------------------------------- customers
SEGMENTS = {}          # (r_band, f_band, m_band) -> name, filled by _segment()


def _segment(r, f, m):
    """standard RFM grid: how recently / how often / how much, each scored 1-5"""
    if r <= 2:
        if f <= 2:  return 'Hibernating' if m <= 2 else 'Need Attention'
        if f == 3:  return 'At Risk'
        return 'At Risk' if m <= 3 else "Can't Lose Them"
    if r == 3:
        if f <= 2:  return 'Promising'
        return 'Need Attention' if m <= 2 else 'Loyal Customers'
    if f <= 2:      return 'New Customers'
    if f == 3:      return 'Need Attention' if m <= 2 else 'Loyal Customers'
    return 'Need Attention' if m <= 2 else 'Loyal Customers' if m == 3 else 'Champions'


def customers(sales, customer, reorder, as_of):
    g = sales.groupby('Customer Internal ID').agg(
        revenue=('Revenue', 'sum'), orders=('Sales Order Number', 'nunique'),
        last=('Sales Order Date', 'max')).reindex(customer['Customer Internal ID'])
    g['recency'] = (as_of - g['last']).dt.days
    g[['revenue', 'orders']] = g[['revenue', 'orders']].fillna(0)      # on the books, never ordered
    g['recency'] = g['recency'].fillna((as_of - sales['Sales Order Date'].min()).days)
    q = lambda v, lab: pd.qcut(v.rank(method='first'), 5, labels=lab).astype(int)
    g['r'] = q(-g['recency'], [1, 2, 3, 4, 5])
    g['f'] = q(g['orders'], [1, 2, 3, 4, 5])
    g['m'] = q(g['revenue'], [1, 2, 3, 4, 5])

    overdue = pd.Series(0, index=g.index, dtype=int)
    for r in reorder:
        if r['status'] == 'Overdue' and r['custId'] in overdue.index:
            overdue[r['custId']] += 1

    info = customer.set_index('Customer Internal ID')
    out = []
    for cid, r in g.sort_values('revenue', ascending=False).iterrows():
        out.append({'id': int(cid),
                    'name': info.at[cid, 'Company Name'] if cid in info.index else '',
                    'type': info.at[cid, 'Customer Type'] if cid in info.index else '',
                    'industry': info.at[cid, 'FA Sub Industries'] if cid in info.index else '',
                    'revenue': int(round(r['revenue'])), 'orders': int(r['orders']),
                    'productsOverdue': int(overdue[cid]),
                    'segment': _segment(r['r'], r['f'], r['m'])})
    return out


# ---------------------------------------------------------------- market basket
def bought_together(sales, item, top=None):
    """item pairs that keep turning up in the same sales order.

    confAB = of the orders containing A, the share that also contained B. It is not
    symmetric, so both directions are reported. Pairs are ranked by lift — how much more
    often they appear together than they would if the two were ordered independently.
    """
    from itertools import combinations
    from collections import Counter
    baskets = sales.groupby('Sales Order Number')['Item Internal ID'].apply(set)
    n_orders = len(baskets)
    seen = Counter()
    pairs = Counter()
    for b in baskets:
        seen.update(b)
        if len(b) > 1:
            pairs.update(combinations(sorted(b), 2))

    code = dict(zip(item['Item Internal ID'], item['Item Code']))
    cat = dict(zip(item['Item Internal ID'], item['Product Category 1']))
    rows = []
    for (a, b), n in pairs.items():
        if n < FBT_MIN_SHARED:
            continue
        rows.append({'aCode': code.get(a, ''), 'aCat': cat.get(a, ''),
                     'bCode': code.get(b, ''), 'bCat': cat.get(b, ''),
                     'coOcc': n, 'confAB': round(n / seen[a], 4), 'confBA': round(n / seen[b], 4),
                     '_lift': n * n_orders / (seen[a] * seen[b])})
    rows.sort(key=lambda r: -r['_lift'])
    for r in rows: r.pop('_lift')
    return rows[:top]


# ---------------------------------------------------------------- headline figures
def headline(sales, inv):
    d = sales['Sales Order Date']
    monthly = sales.groupby(d.dt.to_period('M'))['Revenue'].sum() / 1e6
    fy = sales.groupby(_fy(d))['Revenue'].sum() / 1e6
    total = float(monthly.sum())
    status = {s: 0 for s in ['Continuing', 'Discontinued', 'To Be Advised']}
    for i in inv: status[i['status']] += 1
    return {
        'monthly_labels': [p.strftime("%b '%y") for p in monthly.index],
        'monthly_values_m': [round(v, 3) for v in monthly.values],
        'fy_labels': [f'FY{y % 100}/{(y + 1) % 100}' for y in fy.index],
        'fy_values_m': [round(v, 2) for v in fy.values],
        'total_revenue_m': round(total, 2),
        'avg_monthly_revenue_m': round(total / len(monthly), 2),
        'yoy_growth_pct': round((fy.iloc[-1] / fy.iloc[-2] - 1) * 100, 1),
        'discount_enduser_pct': DISCOUNT_ENDUSER_PCT,
        'status_totals': status,
        'order_lines': int(len(sales)),
    }


def units(sales, item, inv_ids):
    """the same orders counted in pieces instead of baht"""
    d = sales['Sales Order Date']
    monthly = sales.groupby(d.dt.to_period('M'))['Quantity'].sum()
    fy = sales.groupby(_fy(d))['Quantity'].sum()
    per_item = sales.groupby('Item Internal ID')['Quantity'].sum().sort_values(ascending=False)
    per_item = per_item[per_item > 0]
    before = (per_item.cumsum() - per_item) / per_item.sum()
    total = int(monthly.sum())
    return {
        'monthly_units': [int(v) for v in monthly.values],
        'fy_units': [int(v) for v in fy.values],
        'total_units': total,
        'avg_monthly_units': round(total / len(monthly), 1),
        'yoy_units_pct': round((fy.iloc[-1] / fy.iloc[-2] - 1) * 100, 1),
        'units_by_item': {str(int(i)): int(v) for i, v in per_item.items()},
        'volume_tier': {str(int(i)): ('A' if v < VOL_CUT[0] else 'B' if v < VOL_CUT[1] else 'C')
                        for i, v in before.items()},
        'units_by_cust': {str(int(i)): int(v) for i, v in
                          sales.groupby('Customer Internal ID')['Quantity'].sum().items() if v > 0},
    }


# ---------------------------------------------------------------- assemble
def build(cleaned, forecasts, folder='.', template='dashboard_template.html',
          out='dashboard.html'):
    sales, item, customer = cleaned['sales'], cleaned['item'], cleaned['customer']
    as_of = AS_OF or sales['Sales Order Date'].max()
    print('=' * 62); print('STAGE 3 — DASHBOARD'); print('=' * 62)
    print(f'  as of {as_of.date()}  ({len(sales):,} order lines)')

    inv = inventory(item)
    cls = classification(sales, item)
    mis = mismatches(sales, item, as_of)
    reo = reorders(sales, item, as_of)
    cus = customers(sales, customer, reo, as_of)
    fbt_all = bought_together(sales, item, top=None)   # the full set, for the export
    fbt = fbt_all[:FBT_TOP]                           # the dashboard shows the strongest
    agg = headline(sales, inv)
    agg['mismatch_count'] = len(mis)
    agg['mismatch_by_status'] = {s: sum(1 for m in mis if m['status'] == s)
                                 for s in ['Discontinued', 'To Be Advised']}
    qty = units(sales, item, {i['id'] for i in inv})


    soe_export.export_all({'cleaned': cleaned, 'as_of': as_of, 'forecasts': forecasts,
                           'inventory': inv, 'classification': cls,
                           'mismatches': mis, 'reorders': reo, 'customers': cus,
                           'pairs': fbt_all, 'units': qty, 'agg': agg}, folder)

    for r in reo: r.pop('custId')          # the dashboard itself does not need it
    blocks = {'inventoryData': inv, 'itemClassification': cls,
              'mismatchItems': mis, 'reorderData': reo, 'customerData': cus,
              'coOrderingPairs': fbt, 'AGG': agg, 'QTY': qty,
              'LSTM_OUT': forecasts['revenue'], 'QTY_OUT': forecasts['units']}

    print(f'  {len(inv):,} SKUs · {len(cus):,} customers · {len(reo):,} reorder pairs · '
          f'{len(fbt)} product pairs · {len(mis)} mismatches')

    html = open(os.path.join(folder, template), encoding='utf-8').read()
    for name in DATASETS:
        token = '/*{{%s}}*/' % name
        if token not in html:
            raise SystemExit(f'template is missing the {token} placeholder')
        html = html.replace(token, json.dumps(blocks[name], separators=(',', ':')))
    path = os.path.join(folder, out)
    open(path, 'w', encoding='utf-8').write(html)
    print(f'  -> {out}  ({len(html) / 1e6:.2f} MB)')
    print('=' * 62)
    return {'path': path, 'order_lines': len(sales), 'skus': len(inv), 'customers': len(cus),
            'reorders': len(reo), 'pairs': len(fbt), 'mismatches': len(mis)}


def main(folder='.'):
    import soe_clean, soe_forecast
    cleaned = soe_clean.clean_all(folder)
    forecasts = {}
    for target in ('revenue', 'units'):
        p = os.path.join(folder, f'forecast_{target}.json')
        if os.path.exists(p):
            forecasts[target] = json.load(open(p))
        else:
            forecasts[target] = soe_forecast.forecast(cleaned['sales'], target)
            json.dump(forecasts[target], open(p, 'w'), indent=1)
    return build(cleaned, forecasts, folder)['path']


if __name__ == '__main__':
    main(os.path.dirname(os.path.abspath(__file__)) or '.')
