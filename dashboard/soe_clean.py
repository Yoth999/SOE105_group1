"""
STAGE 1 of 3 — IMPORT AND CLEAN
===============================
Drop the three raw exports next to this script and run it:

    Sale Order History.xlsx      order lines
    Item.xlsx                    item master
    Customer.xlsx                customer master

Files are found by pattern, newest first, so the exact filename does not matter.
Every stage after this one reads the DataFrames returned here, so the raw files are
only ever touched once.

    python soe_clean.py          # cleans and writes the *_CLEANED.xlsx copies
"""
import fnmatch, os
from pathlib import Path
import pandas as pd

CLEANED_DIR = 'cleaned'       # the *_CLEANED.xlsx copies go in this subfolder

RAW_PATTERNS = {'sales':    ['*sale*order*.xls*', '*sales*.xls*', '*order*.xls*'],
                'item':     ['*item*.xls*'],
                'customer': ['*customer*.xls*']}

SALES_STATUS_KEEP = ['billed', 'closed']      # anything else is not a confirmed sale

STATUS_MAP = {'continue': 'Continuing', 'continuing': 'Continuing', 'active': 'Continuing',
              'discontinue': 'Discontinued', 'discontinued': 'Discontinued',
              'inactive': 'Discontinued',
              'to be advised': 'To Be Advised', 'tba': 'To Be Advised'}


def _find_raw(folder, kind):
    """newest raw file for a kind, case-insensitive. Only the project folder itself is
    searched, so the cleaned copies in CLEANED_DIR can never be picked up as raw input."""
    hits = [f for f in Path(folder).glob('*.xls*')
            if 'cleaned' not in f.name.lower() and not f.name.startswith('~$')
            and any(fnmatch.fnmatch(f.name.lower(), p) for p in RAW_PATTERNS[kind])]
    return max(hits, key=lambda f: f.stat().st_mtime) if hits else None


def _read(path):
    df = pd.read_excel(path, sheet_name=pd.ExcelFile(path).sheet_names[0])
    df.columns = [str(c).replace('\xa0', ' ').strip() for c in df.columns]
    return df


def clean_sales(path, log):
    df = _read(path); n0 = len(df)
    df = df.drop_duplicates(); n_dup = n0 - len(df)

    status = df['Sales Order Status'].astype(str).str.strip().str.lower()
    df = df[status.isin(SALES_STATUS_KEEP)]; n_status = n0 - n_dup - len(df)

    df['Sales Order Date'] = pd.to_datetime(df['Sales Order Date'], errors='coerce')
    for c in ['Base Price', 'Quantity']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in ['Item Code', 'Company Name', 'Sales Order Number']:
        df[c] = df[c].astype(str).str.strip()
    for c in ['Item Internal ID', 'Customer Internal ID']:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')

    n = len(df); df = df.dropna(subset=['Sales Order Date', 'Base Price', 'Quantity'])
    n_missing = n - len(df)

    n = len(df); df = df[(df['Quantity'] > 0) & (df['Base Price'] >= 0)]   # returns / junk
    n_bad = n - len(df)

    df['Revenue'] = df['Base Price'] * df['Quantity']
    df = df.sort_values('Sales Order Date').reset_index(drop=True)
    log.append(('Sales order history', n0, len(df), [
        f'{n_dup:,} duplicate rows removed',
        f'{n_status:,} rows dropped (status not Billed/Closed)',
        f'{n_missing:,} rows dropped (missing date / price / qty)',
        f'{n_bad:,} rows dropped (qty <= 0 or negative price)']))
    return df


def clean_items(path, log):
    df = _read(path); n0 = len(df)
    df = df.drop_duplicates(); n_dup = n0 - len(df)

    df['Item Internal ID'] = pd.to_numeric(df['Item Internal ID'], errors='coerce').astype('Int64')
    df['List price'] = pd.to_numeric(df['List price'], errors='coerce')
    for c in ['Item Code', 'Product Category 1', 'Product Category 2', 'Production Status']:
        df[c] = df[c].astype(str).str.strip().replace({'nan': '', '- None -': ''})

    n = len(df)      # one row per item; keep the row that actually carries a price
    df = df.sort_values('List price', na_position='last').drop_duplicates('Item Internal ID')
    n_dupid = n - len(df)

    n = len(df); df = df.dropna(subset=['Item Internal ID']); n_noid = n - len(df)

    df['Production Status'] = (df['Production Status'].str.lower()
                               .map(STATUS_MAP).fillna('To Be Advised'))
    n_noprice = int(df['List price'].isna().sum())
    df['Has List Price'] = df['List price'].notna()
    df['List price'] = df['List price'].fillna(0)
    df = df.sort_values('Item Code').reset_index(drop=True)

    log.append(('Item master', n0, len(df), [
        f'{n_dup:,} duplicate rows removed',
        f'{n_dupid:,} duplicate item IDs collapsed',
        f'{n_noid:,} rows dropped (no item ID)',
        f'{n_noprice:,} items have no list price (kept, flagged Has List Price = False)']))
    return df


def clean_customers(path, log):
    df = _read(path); n0 = len(df)
    df = df.drop_duplicates(); n_dup = n0 - len(df)

    df['Customer Internal ID'] = pd.to_numeric(df['Customer Internal ID'], errors='coerce').astype('Int64')
    for c in ['Company Name', 'Customer Type', 'FA Sub Industries']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace({'nan': ''})

    n = len(df)
    df = df.dropna(subset=['Customer Internal ID']).drop_duplicates('Customer Internal ID')
    log.append(('Customer master', n0, len(df), [
        f'{n_dup:,} duplicate rows removed',
        f'{n - len(df):,} rows dropped (no / duplicate customer ID)']))
    return df.reset_index(drop=True)


def clean_all(folder='.', write=True):
    """clean whatever raw files are present; returns {'sales','item','customer'} DataFrames"""
    jobs = [('sales', clean_sales, 'Sale_Order_History_CLEANED.xlsx'),
            ('item', clean_items, 'Item_CLEANED.xlsx'),
            ('customer', clean_customers, 'Customer_CLEANED.xlsx')]
    log, out = [], {}
    out_dir = os.path.join(folder, CLEANED_DIR)
    if write:
        os.makedirs(out_dir, exist_ok=True)

    print('=' * 62); print('STAGE 1 — DATA CLEANING'); print('=' * 62)
    for kind, fn, out_name in jobs:
        raw = _find_raw(folder, kind)
        if raw is None:
            print(f'  - no raw file found for "{kind}" - skipped'); continue
        print(f'  reading {raw.name}')
        out[kind] = df = fn(raw, log)
        if write:
            df.to_excel(os.path.join(out_dir, out_name), sheet_name='Sheet1', index=False)
            print(f'    -> {CLEANED_DIR}/{out_name}')

    print('-' * 62)
    for name, n_in, n_out, notes in log:
        print(f'{name}: {n_in:,} rows in -> {n_out:,} rows out')
        for note in notes: print(f'    - {note}')

    # cross-file checks: order lines pointing at records that do not exist
    for kind, key, what in [('item', 'Item Internal ID', 'an item'),
                            ('customer', 'Customer Internal ID', 'a customer')]:
        if 'sales' in out and kind in out:
            known = set(out[kind][key].dropna())
            n = int((~out['sales'][key].isin(known)).sum())
            print(f'Cross-check: {n:,} order lines reference {what} not in the master file')
    print('=' * 62)
    out['log'] = log            # stage 3 exports this as the data-quality record
    return out


if __name__ == '__main__':
    clean_all(os.path.dirname(os.path.abspath(__file__)) or '.')
