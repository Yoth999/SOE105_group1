"""
THE FRONT DOOR — drag the three Excel files onto a web page and wait.

    python app.py

Opens http://127.0.0.1:8765 in the browser. Drop Sale Order History / Item / Customer on
the page, press Build, and the three pipeline stages run on this computer while the page
shows what they are doing. When it finishes the dashboard opens by itself.

Nothing leaves the machine: this is a plain local server (Python standard library only,
no Flask, no internet). Files are written straight into this folder, so a run started here
and a run started with `python run_all.py` are exactly the same thing.

    upload.html      the page                     (edit this to change how it looks)
    app.py           the server + wiring          (this file)
    run_all.py       what the Build button runs
"""
import contextlib, fnmatch, io, json, os, sys, threading, time, urllib.parse, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Windows consoles default to a legacy code page, and the pipeline prints '·' and '฿'.
# Force UTF-8 on the way out so a console or log file can never kill a run.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import run_all, soe_clean, soe_export

HERE = os.path.dirname(os.path.abspath(__file__)) or '.'
PORT = 8765
CANONICAL = {'sales': 'Sale Order History.xlsx', 'item': 'Item.xlsx', 'customer': 'Customer.xlsx'}
MAX_UPLOAD = 400 << 20                         # 400 MB per file is far beyond any real export
DOWNLOADS = soe_export.OUTPUTS             # one list, defined where the files are written
XLSX = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

STATE = {'running': False, 'done': False, 'error': None, 'lines': [], 'stage': 0,
         'reusing_forecast': False, 'summary': None}
LOCK = threading.Lock()


# ---------------------------------------------------------------- pipeline output -> page
class _Tee(io.TextIOBase):
    """every line the pipeline prints goes to the console and to the page's log box"""

    def __init__(self):
        self.buf = ''

    def write(self, s):
        try:
            sys.__stdout__.write(s); sys.__stdout__.flush()
        except Exception:
            pass                               # a console that cannot show a character

        self.buf += s
        *lines, self.buf = self.buf.split('\n')
        with LOCK:
            for ln in lines:
                if set(ln.strip()) in ({'='}, {'-'}) or not ln.strip():
                    continue
                STATE['lines'].append(ln.rstrip())
                up = ln.upper()
                if 'STAGE 1' in up or 'DATA CLEANING' in up: STATE['stage'] = 1
                elif 'STAGE 2' in up or 'FORECAST' == up.strip(): STATE['stage'] = 2
                elif 'STAGE 3' in up or 'DASHBOARD' in up: STATE['stage'] = 3
                if 'reusing the forecast' in ln: STATE['reusing_forecast'] = True
            del STATE['lines'][:-400]
        return len(s)


def _run():
    with LOCK:
        STATE.update(running=True, done=False, error=None, lines=[], stage=1,
                     reusing_forecast=False, summary=None)
    try:
        with contextlib.redirect_stdout(_Tee()):
            summary = run_all.main(open_browser=False, folder=HERE)
        with LOCK:
            STATE['summary'] = summary
            STATE['stage'] = 4
    except BaseException as e:                 # SystemExit included: missing packages, bad data
        with LOCK:
            STATE['error'] = f'{type(e).__name__}: {e}'
    finally:
        with LOCK:
            STATE['running'] = False; STATE['done'] = True


# ---------------------------------------------------------------- server
def classify(name):
    """which of the three exports a dropped file is, by the same patterns stage 1 uses"""
    low = os.path.basename(name).lower()
    if 'cleaned' in low or low.startswith('~$'):
        return None
    for kind, pats in soe_clean.RAW_PATTERNS.items():
        if any(fnmatch.fnmatch(low, p) for p in pats):
            return kind
    return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):                 # keep the console for the pipeline only
        pass

    def _send(self, code, body=b'', ctype='application/json'):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _file(self, name, ctype='text/html; charset=utf-8'):
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            return self._send(404, b'not built yet', 'text/plain')
        with open(path, 'rb') as f:
            self._send(200, f.read(), ctype)

    def do_GET(self):
        route = urllib.parse.urlparse(self.path).path
        if route in ('/', '/index.html'):
            return self._file('upload.html')
        if route == '/dashboard.html':
            return self._file('dashboard.html')
        if route == '/download':
            want = os.path.basename((urllib.parse.parse_qs(
                urllib.parse.urlparse(self.path).query).get('f') or [''])[0])
            for sub, n, _ in DOWNLOADS:              # only ever the files we advertise
                if n == want:
                    path = os.path.join(HERE, sub, n)
                    if not os.path.exists(path):
                        break
                    with open(path, 'rb') as f:
                        data = f.read()
                    self.send_response(200)
                    self.send_header('Content-Type', XLSX)
                    self.send_header('Content-Disposition', f'attachment; filename="{n}"')
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers(); self.wfile.write(data)
                    return
            return self._send(404, b'no such export', 'text/plain')

        if route == '/status':
            with LOCK:
                s = dict(STATE)
            s['has_dashboard'] = os.path.exists(os.path.join(HERE, 'dashboard.html'))
            s['built_at'] = run_all._cache(HERE).get('built_at')
            s['files'] = [{'name': n, 'what': what, 'kb': round(os.path.getsize(p) / 1024)}
                          for sub, n, what in DOWNLOADS
                          if os.path.exists(p := os.path.join(HERE, sub, n))]
            return self._send(200, json.dumps(s).encode())
        self._send(404, b'{}')

    def do_POST(self):
        route = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(route.query)

        if route.path == '/upload':
            size = int(self.headers.get('Content-Length', 0))
            if size <= 0 or size > MAX_UPLOAD:
                return self._send(400, json.dumps({'error': 'bad file size'}).encode())
            name = (q.get('name') or ['upload.xlsx'])[0]
            kind = (q.get('kind') or [None])[0] or classify(name)
            if kind not in CANONICAL:
                return self._send(400, json.dumps({'error': f'unrecognised file: {name}'}).encode())
            # Saved under the canonical name, so the folder never collects near-duplicates.
            # Written to a temporary file first and swapped in at the end: the browser may
            # well be streaming the body straight out of the file we are about to replace.
            path = os.path.join(HERE, CANONICAL[kind])
            tmp = path + '.part'
            with open(tmp, 'wb') as f:
                left = size
                while left:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    f.write(chunk); left -= len(chunk)
            if left:
                os.remove(tmp)
                return self._send(400, json.dumps({'error': 'upload was cut short'}).encode())
            os.replace(tmp, path)
            return self._send(200, json.dumps({'kind': kind, 'saved': CANONICAL[kind]}).encode())

        if route.path == '/run':
            with LOCK:
                busy = STATE['running']
            if not busy:
                threading.Thread(target=_run, daemon=True).start()
            return self._send(200, json.dumps({'started': not busy}).encode())

        self._send(404, b'{}')


def main():
    for port in range(PORT, PORT + 10):         # in case something else already has 8765
        try:
            server = ThreadingHTTPServer(('127.0.0.1', port), Handler); break
        except OSError:
            continue
    else:
        sys.exit(f'could not open a port in {PORT}-{PORT + 9}')
    url = f'http://127.0.0.1:{port}/'
    print(f'SOE Mitsubishi — dashboard builder\n  {url}\n  (close this window to stop)\n')
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nstopped.')


if __name__ == '__main__':
    main()
