"""
SOE Mitsubishi dashboard — run the whole pipeline.

    python app.py            drag the three exports onto a web page (easiest)
    python run_all.py        use the files already sitting in this folder

        Sale Order History.xlsx   Item.xlsx   Customer.xlsx

    stage 1  soe_clean.py      read and clean the raw Excel        -> cleaned/*_CLEANED.xlsx
    stage 2  soe_forecast.py   five models x two targets           -> forecast_*.json
    stage 3  soe_dashboard.py  analytics + dashboard_template.html -> dashboard.html

Nothing is recomputed that does not have to be. Every run fingerprints the three raw
files (SHA-256) and compares them with the last run, stored in .pipeline_cache.json:

    all three unchanged      nothing runs, the existing dashboard opens      instant
    only item/customer new   clean + dashboard, the forecast is reused       ~1 min
    sales file is new        everything runs                                 ~2 min

    --force           recompute even if nothing changed
    --skip-forecast   reuse the forecast files whatever the fingerprint says
    --no-open         do not open the dashboard in a browser when it is done

    pip install pandas numpy scipy scikit-learn openpyxl
"""
import hashlib, importlib.util, json, os, pathlib, sys, time, webbrowser

# Windows consoles default to a legacy code page, and the pipeline prints '·' and '฿'.
# Force UTF-8 on the way out so a console or log file can never kill a run.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# check the environment before anything else, so a missing package gives one clear line
# instead of a stack trace from whichever module happened to import it first
MISSING = [pip for mod, pip in [('pandas', 'pandas'), ('numpy', 'numpy'), ('scipy', 'scipy'),
                                ('sklearn', 'scikit-learn'), ('openpyxl', 'openpyxl')]
           if importlib.util.find_spec(mod) is None]
if MISSING:
    sys.exit(f'\n  missing package(s): {", ".join(MISSING)}\n'
             f'  run:  pip install {" ".join(MISSING)}\n'
             f'  (python in use: {sys.executable})\n')

import soe_clean, soe_forecast, soe_dashboard, soe_export

FOLDER = os.path.dirname(os.path.abspath(__file__)) or '.'
CACHE_FILE = '.pipeline_cache.json'
CHROME_PATHS = [r'C:\Program Files\Google\Chrome\Application\chrome.exe',
                r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
                os.path.expandvars(r'%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe'),
                '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
                '/usr/bin/google-chrome', '/usr/bin/chromium']


# ---------------------------------------------------------------- change detection
def fingerprint(folder=FOLDER):
    """SHA-256 of each raw file, so an identical re-upload is recognised as identical"""
    out = {}
    for kind in ('sales', 'item', 'customer'):
        path = soe_clean._find_raw(folder, kind)
        if path is None:
            continue
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        out[kind] = h.hexdigest()
    return out


def _cache(folder=FOLDER, write=None):
    path = os.path.join(folder, CACHE_FILE)
    if write is not None:
        json.dump(write, open(path, 'w'), indent=1)
        return write
    try:
        return json.load(open(path))
    except Exception:
        return {}


def plan(folder=FOLDER, force=False, skip_forecast=False):
    """decide what actually needs to run this time"""
    now, before = fingerprint(folder), _cache(folder).get('files', {})
    have = lambda *n: all(os.path.exists(os.path.join(folder, *f.split('/'))) for f in n)
    dash = have('dashboard.html', *[f'{sub}/{n}' for sub, n, _ in soe_export.OUTPUTS])
    fc = have('forecast_revenue.json', 'forecast_units.json')
    if force:
        return 'full', now
    if now and now == before and dash and fc:
        return 'nothing', now
    if skip_forecast or (now.get('sales') and now.get('sales') == before.get('sales') and fc):
        return 'no-forecast', now
    return 'full', now


# ---------------------------------------------------------------- browser
def open_dashboard(folder=FOLDER):
    url = pathlib.Path(os.path.join(folder, 'dashboard.html')).resolve().as_uri()
    for exe in CHROME_PATHS:                       # Chrome first, it is what the UI is built for
        if os.path.exists(exe):
            try:
                webbrowser.get(f'"{exe}" %s').open(url); return 'chrome'
            except Exception:
                break
    webbrowser.open(url)
    return 'default browser'


# ---------------------------------------------------------------- the run
def main(force=False, skip_forecast=False, open_browser=True, folder=FOLDER):
    t0 = time.time()
    what, files = plan(folder, force, skip_forecast)

    if what == 'nothing':
        print('=' * 62)
        print('NOTHING TO DO — the three raw files are identical to the last run')
        print(f'  reusing the dashboard built {_cache(folder).get("built_at", "earlier")}')
        print('=' * 62)
        summary = dict(_cache(folder).get('summary', {}), reused='everything', elapsed=0)
    else:
        cleaned = soe_clean.clean_all(folder)
        print('=' * 62); print('STAGE 2 — FORECAST'); print('=' * 62)
        if what == 'no-forecast':
            print('  the sales file has not changed — reusing the forecast from the last run')
        forecasts = {}
        for target in ('revenue', 'units'):
            path = os.path.join(folder, f'forecast_{target}.json')
            if what == 'no-forecast' and os.path.exists(path):
                forecasts[target] = json.load(open(path))
            else:
                forecasts[target] = soe_forecast.forecast(cleaned['sales'], target)
                json.dump(forecasts[target], open(path, 'w'), indent=1)
                print(f'    -> forecast_{target}.json')

        summary = soe_dashboard.build(cleaned, forecasts, folder)
        summary['reused'] = 'the forecast' if what == 'no-forecast' else None
        summary['elapsed'] = round(time.time() - t0)
        _cache(folder, write={'files': files, 'built_at': time.strftime('%Y-%m-%d %H:%M'),
                              'summary': summary})
        print(f'done in {summary["elapsed"]}s')

    if open_browser:
        print(f'  opening dashboard.html in {open_dashboard(folder)}')
    return summary


if __name__ == '__main__':
    main(force='--force' in sys.argv,
         skip_forecast='--skip-forecast' in sys.argv,
         open_browser='--no-open' not in sys.argv)
