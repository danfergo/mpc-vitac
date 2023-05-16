import glob
import os
import time
from threading import Thread
import re
from os.path import join, isdir, isfile
from os import listdir
from socketserver import ThreadingMixIn
from http.server import HTTPServer, BaseHTTPRequestHandler, SimpleHTTPRequestHandler

from ....experimenter import e
import json


def replace_all(x, config):
    # not very pretty code, but it works
    # replaces all ${key} -> config[key] in the attr.

    wrap = {'txt': x, 'offset': 0}
    regex = "(\$\{(\w(\w|\\n)*)\})"

    def replace(match, o):
        txt = o['txt']
        offset = o['offset']
        k = match.group(2)
        s = match.span(2)
        if k not in config:
            raise Exception('Config "' + k + '" not found !')

        o['txt'] = txt[: offset + s[0] - 2] + config[k] + txt[offset + s[1] + 1:]
        # offset is used to compensate the "original" spans
        # for the differences in the string before and after
        o['offset'] += len(config[k]) - (s[1] - s[0]) - 3

    ''
    [replace(x, wrap) for x in re.finditer(regex, x)]
    return wrap['txt']


def HandlerFactory(data_):
    class CustomHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.routes = {
                '/': self.index,
                '/index.html': self.index,
                '/api': self.api,
                '/404': self.not_found,
            }
            self.data = data_
            super().__init__(*args, directory=e.ws('outputs'), **kwargs)

        # def __call__(self, *args, **kwargs):
        #     """Handle a request."""
        #     super().__init__(*args, directory=e.out(), **kwargs)

        def send_template(self, path, data=None):
            full_path = os.path.dirname(__file__) + '/html/' + path
            f = open(full_path, 'r')

            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html = replace_all(f.read(), data if data is not None else {})
            self.wfile.write(bytes(html, "utf-8"))

        def send_json(self, data=None):
            data = {} if data is None else data
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(bytes(json.dumps(data), 'utf-8'))

        ## routes

        def not_found(self):
            self.send_template('/404.html')

        def api(self):
            self.send_json(self.data)

        def index(self):

            # card_template = """
            #     <div class="card">
            #       <div class="card-body">
            #         <h5 class="card-title">{src}</h5>
            #         <img src="/{src}">
            #       </div>
            #     </div>
            # """

            # plots = os.listdir(e.out())
            # data = {
            #     'main_content': "\n".join([card_template.format(src=p) for p in plots])
            # }
            self.send_template('/index.html')

        def do_GET(self):
            path = self.path
            if path in self.routes:
                self.routes[path]()
            else:
                super(CustomHandler, self).do_GET()

        def log_message(self, format, *args):
            return

    return CustomHandler


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class EBoard:

    def __init__(self):
        hostName = "0.0.0.0"
        serverPort = 8080
        self.server = None
        self.data = {}
        self.thread = Thread(target=self.serve_on_port, args=[
            hostName,
            serverPort,
            self.data
        ])
        self.thread.start()

    def serve_on_port(self, hostname, port, data):
        handler = HandlerFactory(data)
        self.server = ThreadingHTTPServer((hostname, port), handler)
        print("Server started http://%s:%s" % (hostname, port))
        self.server.serve_forever()

    def on_epoch_start(self, ev):
        def assign(k, v):
            self.data[k] = v

        assign('epoch', ev['epoch'])
        assign('key', e.config['__key__'])
        assign('run_key', e.config['__run_key__'])

        def ff_tree(p):
            ls = list(listdir(p))
            files = [f for f in ls if isfile(join(p, f))]
            return {
                **{f: ff_tree(join(p, f)) for f in ls if f != '__pycache__' and isdir(join(p, f))},
                **({f: None for f in files} if len(files) > 0 else {})
            }

        assign('outputs', ff_tree(e.ws('outputs')))

    def on_e_end(self, ev):
        self.server.shutdown()
        self.thread.join()


if __name__ == '__main__':
    EBoard()
