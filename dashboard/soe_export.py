"""
EXPORTS — the workbooks people take away from the dashboard
===========================================================
Stage 3 calls export_all() once the analytics are in memory, so nothing here recomputes
anything; it only arranges what the dashboard already knows into files that can be
emailed to someone who has never opened the dashboard.

    cleaned/Forecast_FY25.xlsx      the FY25 forecast, beside the cleaned data
    exports/Call_List.xlsx          sales — who to call, what to offer
    exports/Data_Issues.xlsx        data owner — what to fix
    exports/Inventory_Plan.xlsx     purchasing — what to stock
    exports/Customers.xlsx          sales manager — who matters, who is slipping

Every workbook opens on a "How to read this" sheet: what it is, as of when, where the
numbers came from and what to watch out for. These files travel on their own.
"""
import os
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

EXPORT_DIR = 'exports'

# every file a finished run leaves behind: (folder, name, what it is).
# run_all checks this list before deciding a rebuild can be skipped, and app.py
# offers the same list for download — so adding an export here is all it takes.
OUTPUTS = [('exports', 'Call_List.xlsx',      'Who to call next, with what to offer'),
           ('exports', 'Inventory_Plan.xlsx', 'Every SKU, with value / volume / demand tier'),
           ('exports', 'Customers.xlsx',      'The customer base by segment'),
           ('exports', 'Data_Issues.xlsx',    'The worklist for whoever owns the master data'),
           ('cleaned', 'Forecast_FY25.xlsx',  'FY25 by month and model')]
FONT = 'Arial'
THB, NUM, PCT = '#,##0', '#,##0', '0.0%'

# same cut-offs the dashboard itself applies before showing a reorder pair
MAX_DAYS, MAX_PROGRESS = 365, 500
UPSELL_MIN_CONF = 0.50        # only suggest a pairing that holds in half the orders
DEAD_STOCK_DAYS = 365

# the dashboard rolls the eight RFM boxes into five groups people can act on
SEGMENT_GROUP = {'Champions': 'Best Customers', 'Loyal Customers': 'Regular Customers',
                 "Can't Lose Them": 'At Risk', 'At Risk': 'At Risk', 'Need Attention': 'At Risk',
                 'New Customers': 'New & Growing', 'Promising': 'New & Growing',
                 'Hibernating': 'Inactive'}
_group = lambda s: SEGMENT_GROUP.get(s, s)

_head = Font(name=FONT, bold=True, color='FFFFFF', size=10)
_fill = PatternFill('solid', fgColor='22262B')
_bold = Font(name=FONT, bold=True, size=10)
_body = Font(name=FONT, size=10)
_note = Font(name=FONT, size=9, italic=True, color='5A6068')
_title = Font(name=FONT, bold=True, size=12)
_rule = Border(top=Side(style='thin', color='BFC5CC'))


# ---------------------------------------------------------------- small builders
def sheet(wb, name, headers, rows, fmts=None, widths=None, title=None, note=None):
    ws = wb.create_sheet(name[:31])
    r = 1
    if title:
        ws.cell(1, 1, title).font = _title
        r = 2
    if note:
        ws.cell(r, 1, note).font = _note
        r += 1
    r += 1
    for c, h in enumerate(headers, 1):
        cell = ws.cell(r, c, h); cell.font = _head; cell.fill = _fill
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    for i, values in enumerate(rows, r + 1):
        for c, v in enumerate(values, 1):
            cell = ws.cell(i, c, v)
            cell.font = _body
            if fmts and fmts[c - 1]:
                cell.number_format = fmts[c - 1]
    for c, w in enumerate(widths or [], 1):
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.freeze_panes = ws.cell(r + 1, 1)
    ws.auto_filter.ref = f'A{r}:{get_column_letter(len(headers))}{r + len(rows)}'
    return ws


def cover(wb, title, as_of, what, columns, caveats=()):
    """the first sheet of every export: what this file is, for someone who was not in the room"""
    ws = wb.create_sheet('How to read this', 0)
    ws.column_dimensions['A'].width = 22
    ws.column_dimensions['B'].width = 95
    ws['A1'] = title; ws['A1'].font = Font(name=FONT, bold=True, size=14)
    r = 3
    for label, text in [('What this is', what),
                        ('Data as of', f'{as_of:%d %b %Y} — the last day in the sales history'),
                        ('Source', 'Sale Order History.xlsx · Item.xlsx · Customer.xlsx, '
                                   'cleaned by the SOE dashboard pipeline. Confirmed sales only '
                                   '(Billed / Closed), duplicate rows removed.')]:
        ws.cell(r, 1, label).font = _bold
        c = ws.cell(r, 2, text); c.font = _body; c.alignment = Alignment(wrap_text=True, vertical='top')
        ws.row_dimensions[r].height = 30
        r += 2
    ws.cell(r, 1, 'Columns').font = _bold
    for name, meaning in columns:
        ws.cell(r, 2, f'{name} — {meaning}').font = _body
        r += 1
    if caveats:
        r += 1
        ws.cell(r, 1, 'Watch out').font = _bold
        for text in caveats:
            ws.cell(r, 2, text).font = _note
            r += 1
    return ws


def _save(wb, folder, sub, name):
    path = os.path.join(folder, sub, name) if sub else os.path.join(folder, name)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    wb.save(path)
    print(f'  -> {sub + "/" if sub else ""}{name}')
    return path


# ---------------------------------------------------------------- 1. the call list
def call_list(d, folder, as_of):
    """who sales should ring, most valuable first, with the obvious add-on to mention"""
    price = {i['id']: i['price'] for i in d['inventory']}
    segment = {c['id']: _group(c['segment']) for c in d['customers']}

    # best partner product for each item, from the market-basket pairs, either direction
    partner = {}
    for p in d['pairs']:
        for a, b, conf in ((p['aCode'], p['bCode'], p['confAB']), (p['bCode'], p['aCode'], p['confBA'])):
            if conf >= UPSELL_MIN_CONF and conf > partner.get(a, (None, 0))[1]:
                partner[a] = (b, conf)

    rows = []
    for r in d['reorders']:
        if r['daysSince'] > MAX_DAYS or r['progressPct'] > MAX_PROGRESS:
            continue                                  # same filter the dashboard uses
        p = price.get(str(r['itemId']), 0)
        also = partner.get(r['itemCode'])
        rows.append([r['custName'], segment.get(r['custId'], ''), r['itemCode'], r['cat1'],
                     r['daysSince'], max(1, round(r['daysSince'] / (r['progressPct'] / 100))),
                     r['progressPct'] / 100, r['estValue'],
                     max(1, round(r['estValue'] / p)) if p else None,
                     f'{also[0]} ({also[1]:.0%} of orders)' if also else ''])
    rows.sort(key=lambda x: -x[7])

    headers = ['Customer', 'Segment', 'Item', 'Category', 'Days since last order',
               'Usual cycle (days)', '% of cycle', 'Est. value (THB)', 'Est. units', 'Suggest also']
    fmts = [None, None, None, None, NUM, NUM, PCT, THB, NUM, None]
    widths = [26, 15, 10, 16, 12, 12, 11, 16, 10, 24]

    wb = Workbook(); wb.remove(wb.active)
    cover(wb, 'Reorder call list', as_of,
          'Every customer-product pair whose usual reorder date has arrived or passed, judged '
          'against that customer\'s own ordering rhythm. One sheet per urgency — start with '
          '"Expected Now".',
          [('% of cycle', 'how far through the usual gap between that customer\'s orders we are. '
                          '100% = the reorder date has just arrived'),
           ('Usual cycle', 'the median gap between that customer\'s orders for that item'),
           ('Est. value', 'their average order quantity x list price, before discount'),
           ('Suggest also', 'the product most often bought in the same order as this one')],
          ['Only pairs ordered 3+ times are listed — fewer than that and there is no rhythm to measure.',
           'Pairs last ordered over a year ago, or more than 5 cycles overdue, are left out: '
           'those customers have stopped buying rather than being due.',
           'Est. value is an opportunity, not a booked sale.'])
    for name, status in [('Expected Now', 'Expected Now'), ('Overdue', 'Overdue'),
                         ('Approaching', 'Approaching Reorder')]:
        part = [r for r in rows if _status(r[6]) == status]
        sheet(wb, name, headers, part, fmts, widths,
              title=f'{name} — {len(part):,} to follow up',
              note=f'Sorted by estimated value. Total {sum(r[7] for r in part):,.0f} THB in play.')
    return _save(wb, folder, EXPORT_DIR, 'Call_List.xlsx')


def _status(pct):
    return 'Approaching Reorder' if pct < 1.0 else 'Expected Now' if pct < 1.3 else 'Overdue'


# ---------------------------------------------------------------- 2. data issues
def data_issues(d, folder, as_of):
    """the worklist for whoever owns the master data"""
    wb = Workbook(); wb.remove(wb.active)
    cover(wb, 'Data issues worklist', as_of,
          'Records where the master files and the sales history disagree, plus a log of what '
          'the cleaning step removed. Each row is something a person has to decide about — '
          'the system can spot them but cannot settle them.',
          [('Still selling', 'items marked Discontinued or To Be Advised that took orders in the '
                             'last 6 months — either the status is out of date, or the sale '
                             'should not have happened'),
           ('Other issues', 'master records that are incomplete or unused'),
           ('Cleaning log', 'what was dropped on the way in, and why')])

    sheet(wb, 'Still selling', ['Item', 'Category', 'Status', 'Revenue last 180 days (THB)'],
          [[m['code'], m['cat1'], m['status'], m['recent_revenue_6mo']] for m in d['mismatches']],
          [None, None, None, THB], [10, 20, 16, 24],
          title=f'Dead on paper, alive in the orders — {len(d["mismatches"]):,} items',
          note='Sorted by revenue. The ones at the top are costing the most confusion.')

    item, sales, cust = d['cleaned']['item'], d['cleaned']['sales'], d['cleaned']['customer']
    sold = set(sales['Item Internal ID'].dropna())
    ordered = set(sales['Customer Internal ID'].dropna())
    other = []
    for _, r in item[~item['Has List Price']].iterrows():
        other.append(['No list price', 'Item', int(r['Item Internal ID']), r['Item Code'],
                      'cannot be priced, so it is left out of the dashboard catalogue'])
    for _, r in item[~item['Item Internal ID'].isin(sold)].iterrows():
        other.append(['Never sold', 'Item', int(r['Item Internal ID']), r['Item Code'],
                      f'in the item master, no order line in {len(d["agg"]["monthly_labels"])} months'])
    for _, r in cust[~cust['Customer Internal ID'].isin(ordered)].iterrows():
        other.append(['Never ordered', 'Customer', int(r['Customer Internal ID']),
                      r['Company Name'], 'on the books, has never placed an order'])
    sheet(wb, 'Other issues', ['Issue', 'Record', 'Internal ID', 'Code / name', 'What it means'],
          other, [None, None, None, None, None], [16, 10, 13, 30, 56],
          title=f'Master records to review — {len(other):,}')

    log = d['cleaned'].get('log', [])
    rows = [[name, n_in, n_out, n_in - n_out, ' · '.join(notes)] for name, n_in, n_out, notes in log]
    sheet(wb, 'Cleaning log', ['File', 'Rows in', 'Rows kept', 'Rows removed', 'Why'],
          rows, [None, NUM, NUM, NUM, None], [22, 11, 11, 13, 90],
          title='What the cleaning step removed',
          note='Run on the raw exports every time the dashboard is rebuilt.')
    return _save(wb, folder, EXPORT_DIR, 'Data_Issues.xlsx')


# ---------------------------------------------------------------- 3. inventory plan
def inventory_plan(d, folder, as_of):
    """the catalogue with everything the pipeline knows about how each item behaves"""
    sales = d['cleaned']['sales']
    last = sales.groupby('Item Internal ID')['Sales Order Date'].max()
    units, tier, cls = d['units']['units_by_item'], d['units']['volume_tier'], d['classification']
    abc = {'A': 'High Value', 'B': 'Medium Value', 'C': 'Low Value'}
    vol = {'A': 'High Volume', 'B': 'Medium Volume', 'C': 'Low Volume'}
    xyz = {'X': 'Stable', 'Y': 'Variable', 'Z': 'Irregular', 'Insufficient Data': 'Not enough data'}

    rows = []
    for i in d['inventory']:
        iid = int(i['id'])
        when = last.get(iid)
        c = cls.get(i['id'], {})
        rows.append([iid, i['code'], i['cat1'], i['cat2'], i['status'], i['price'],
                     units.get(i['id'], 0), abc.get(c.get('abc'), ''),
                     vol.get(tier.get(i['id']), 'No sales'), xyz.get(c.get('xyz'), ''),
                     when.date() if pd.notna(when) else None,
                     (as_of - when).days if pd.notna(when) else None])
    rows.sort(key=lambda r: -r[6])

    headers = ['Item ID', 'Code', 'Category 1', 'Category 2', 'Status', 'List price (THB)',
               'Units sold (4Y)', 'Value tier', 'Volume tier', 'Demand pattern',
               'Last sold', 'Days since']
    fmts = [None, None, None, None, None, THB, NUM, None, None, None, 'DD MMM YYYY', NUM]
    widths = [10, 9, 18, 20, 14, 15, 13, 13, 14, 15, 13, 11]

    wb = Workbook(); wb.remove(wb.active)
    cover(wb, 'Inventory plan', as_of,
          'Every SKU with a list price, and what four years of orders say about it. The three '
          'tiers are meant to be read together: High Value + Stable deserves tight stock '
          'control, Low Value + Irregular rarely deserves standing stock at all.',
          [('Value tier', 'share of revenue — High = the SKUs making up the top 80%'),
           ('Volume tier', 'the same split on pieces shipped rather than baht'),
           ('Demand pattern', 'how steady monthly demand is: Stable, Variable or Irregular'),
           ('Days since', 'days between the last order for this item and the data date')],
          ['This file has no stock-on-hand column because the item master does not carry one. '
           'Add it and this becomes a real purchasing plan.',
           'The forecast is for the company as a whole, not per SKU, so no per-item FY25 number '
           'is shown here rather than show an invented one.'])
    sheet(wb, 'All SKUs', headers, rows, fmts, widths,
          title=f'{len(rows):,} SKUs', note='Sorted by units sold over the four years.')
    quiet = [r for r in rows if r[4] == 'Continuing' and (r[11] is None or r[11] > DEAD_STOCK_DAYS)]
    sheet(wb, 'No sales 12 months', headers, quiet, fmts, widths,
          title=f'Still listed as Continuing, but quiet — {len(quiet):,} SKUs',
          note='Candidates for review: still supplied and sellable, no order in the last year.')
    return _save(wb, folder, EXPORT_DIR, 'Inventory_Plan.xlsx')


# ---------------------------------------------------------------- 4. customers
def customers(d, folder, as_of):
    by_cust = d['units']['units_by_cust']
    rows = [[c['id'], c['name'], c['type'], c['industry'], _group(c['segment']), c['segment'],
             c['revenue'], c['orders'], by_cust.get(str(c['id']), 0), c['productsOverdue']]
            for c in d['customers']]
    headers = ['Customer ID', 'Company', 'Type', 'Industry', 'Segment', 'RFM box',
               'Revenue (THB)', 'Orders', 'Units bought', 'Products overdue']
    fmts = [None, None, None, None, None, None, THB, NUM, NUM, NUM]
    widths = [12, 28, 11, 28, 17, 16, 18, 10, 13, 15]

    total = sum(r[6] for r in rows) or 1
    groups = {}
    for r in rows:
        g = groups.setdefault(r[4], [0, 0, 0])
        g[0] += 1; g[1] += r[6]; g[2] += r[9]
    summary = sorted(([k, v[0], v[1], v[1] / total, round(v[1] / v[0]), v[2]]
                      for k, v in groups.items()), key=lambda r: -r[2])

    wb = Workbook(); wb.remove(wb.active)
    cover(wb, 'Customer base', as_of,
          'Every customer ranked by what they have spent, with the behaviour group the system '
          'put them in and how many of their products are overdue for a repeat order.',
          [('Segment', 'from how recently, how often and how much they buy — Best Customers, '
                       'Regular, New & Growing, At Risk, Inactive. The same names the dashboard uses'),
           ('RFM box', 'the finer eight-way split the five groups are rolled up from'),
           ('Products overdue', 'how many of their regular products are past their usual '
                                'reorder date — the same pairs listed in Call_List.xlsx'),
           ('Units bought', 'pieces, not baht: a customer can be small in revenue and large in volume')],
          ['A Best Customer with a high overdue count is the easiest money on this list: they '
           'still trust us, nobody has called.'])
    sheet(wb, 'All customers', headers, rows, fmts, widths,
          title=f'{len(rows):,} customers', note='Sorted by revenue.')
    sheet(wb, 'By segment',
          ['Segment', 'Customers', 'Revenue (THB)', '% of revenue', 'Avg per customer (THB)',
           'Products overdue'],
          summary, [None, NUM, THB, PCT, THB, NUM], [18, 12, 20, 13, 21, 16],
          title='Where the revenue sits')
    return _save(wb, folder, EXPORT_DIR, 'Customers.xlsx')


# ---------------------------------------------------------------- 5. the forecast
ORDER = ['Ensemble', 'SARIMA', 'XGBoost', 'LSTM', 'LightGBM']


def forecast_workbook(forecasts, folder, as_of, out_dir='cleaned', name='Forecast_FY25.xlsx'):
    """FY25 by month and model — kept beside the cleaned data, as the figures it is derived from"""
    rev, qty = forecasts['revenue'], forecasts['units']
    month = lambda lab: pd.Period(lab, freq='M').strftime('%b %Y')

    wb = Workbook(); wb.remove(wb.active)
    cover(wb, 'FY25 forecast', as_of,
          'Apr 2025 – Mar 2026 projected by five models from 209 weeks of real sales-order '
          'history. Every model was then scored on a year it was never shown.',
          [('Off %', 'average monthly error on the held-out year Apr 2024 – Mar 2025 — lower is better'),
           ('Ensemble', 'not a trained model: the plain average of the four others, month by month')],
          ['Read the range between the models as the direction, not a plan number: even the best '
           'of them is off by around 12% in a month.',
           'Revenue and units are forecast separately, on the same data and the same protocol.'])

    rows = [[m, rev['models'][m]['mape'] / 100, sum(rev['models'][m]['forecast_values_m']) * 1e6,
             qty['models'][m]['mape'] / 100, sum(qty['models'][m]['forecast_values_m'])]
            for m in ORDER]
    sheet(wb, 'Accuracy', ['Model', 'Off % (revenue)', 'FY25 revenue (THB)', 'Off % (units)',
                           'FY25 units'], rows, [None, PCT, THB, PCT, NUM], [13, 15, 21, 14, 15],
          title='Model scorecard',
          note=f'FY24 actual: {rev["fy24_actual_m"] * 1e6:,.0f} THB · {qty["fy24_actual_m"]:,.0f} units')

    for title, src, unit, fmt, scale in [('FY25 Revenue', rev, 'revenue in baht', THB, 1e6),
                                         ('FY25 Units', qty, 'units shipped', NUM, 1)]:
        rows = [[month(lab)] + [src['models'][m]['forecast_values_m'][i] * scale for m in ORDER]
                for i, lab in enumerate(src['forecast_labels'])]
        ws = sheet(wb, title, ['Month'] + ORDER, rows, [None] + [fmt] * 5, [13] + [15] * 5,
                   title=f'FY25 forecast — {unit}', note='Apr 2025 – Mar 2026, one column per model.')
        end = ws.max_row + 1
        ws.cell(end, 1, 'FY25 total').font = _bold
        ws.cell(end, 1).border = _rule
        for c in range(2, 7):                         # a real formula, so edits recalculate
            col = get_column_letter(c)
            cell = ws.cell(end, c, f'=SUM({col}{end - 12}:{col}{end - 1})')
            cell.font = _bold; cell.number_format = fmt; cell.border = _rule

    for title, src, fmt, scale in [('Backtest Revenue', rev, THB, 1e6),
                                   ('Backtest Units', qty, NUM, 1)]:
        rows = [[month(lab), src['backtest_actual_monthly'][i] * scale] +
                [src['models'][m]['backtest_pred_monthly'][i] * scale for m in ORDER]
                for i, lab in enumerate(src['backtest_labels'])]
        sheet(wb, title, ['Month', 'Actual'] + ORDER, rows, [None] + [fmt] * 6, [13] + [15] * 6,
              title=f'{title} — the held-out year',
              note='Trained on Apr 2021 – Mar 2024 only, then asked to predict this year.')
    return _save(wb, folder, out_dir, name)


# ---------------------------------------------------------------- entry point
def export_all(d, folder='.'):
    as_of = d['as_of']
    print('  exports:')
    return [forecast_workbook(d['forecasts'], folder, as_of),
            call_list(d, folder, as_of),
            data_issues(d, folder, as_of),
            inventory_plan(d, folder, as_of),
            customers(d, folder, as_of)]
