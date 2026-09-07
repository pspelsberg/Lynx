from __future__ import annotations
import asyncio,sys,tempfile
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from lynx_harness.lsp import LspClient
from lynx_harness.playwright import PlaywrightCli
from lynx_harness.tools import ToolError
LSP=r"""import sys,json
def read():
 h=b''
 while not h.endswith(b'\r\n\r\n'): h+=sys.stdin.buffer.read(1)
 n=int([x for x in h.decode().split('\r\n') if x.lower().startswith('content-length')][0].split(':')[1])
 return json.loads(sys.stdin.buffer.read(n))
def send(x):
 b=json.dumps(x).encode(); sys.stdout.buffer.write(f'Content-Length: {len(b)}\r\n\r\n'.encode()+b); sys.stdout.buffer.flush()
while True:
 try: m=read()
 except: break
 if 'id' in m: send({'jsonrpc':'2.0','id':m['id'],'result':{}})
"""
class SensorTests(unittest.TestCase):
 def test_playwright_requires_installed_binary(self):
  with self.assertRaises(ToolError): PlaywrightCli(Path(tempfile.gettempdir()),"definitely-not-installed")

 def test_playwright_bounds_process_output(self):
  with tempfile.TemporaryDirectory() as d:
   executable=Path(d)/"browser.sh"
   executable.write_text("#!/bin/sh\nprintf '%0100001d' 0\n")
   executable.chmod(0o700)
   browser=PlaywrightCli(Path(d), str(executable), {"example.test"})
   async def run():
    with self.assertRaises(ToolError): await browser.run("snapshot")
   asyncio.run(run())
 def test_lsp_rejects_oversized_header(self):
  bad = r"""import sys
sys.stdout.write('X'*9000 + '\n'); sys.stdout.flush()
while True: sys.stdin.buffer.read(1)
"""
  with tempfile.TemporaryDirectory() as d:
   script=Path(d)/"lsp.py"; script.write_text(bad)
   async def run():
    c=LspClient((sys.executable,str(script)),Path(d))
    with self.assertRaises(ToolError): await c.start()
    self.assertIsNone(c.process)
   asyncio.run(run())

 def test_lsp_content_length_transport(self):
  with tempfile.TemporaryDirectory() as d:
   script=Path(d)/"lsp.py"; script.write_text(LSP)
   async def run():
    c=LspClient((sys.executable,str(script)),Path(d)); await c.start(); result=await c.request("x",{}); await c.close(); return result
   self.assertEqual(asyncio.run(run()),{})
if __name__=="__main__": unittest.main()
