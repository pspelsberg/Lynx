from __future__ import annotations
import asyncio
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from lynx_harness.mcp import McpServerConfig,McpStdioClient

SERVER="""import sys,json
for line in sys.stdin:
 m=json.loads(line)
 if m.get('id'):
  if m['method']=='initialize': r={'protocolVersion':'2025-06-18','capabilities':{}}
  else: r={'tools':[]}
  print(json.dumps({'jsonrpc':'2.0','id':m['id'],'result':r}),flush=True)
"""
class McpTests(unittest.TestCase):
 def test_initialize_and_request(self):
  with tempfile.TemporaryDirectory() as d:
   script=Path(d)/"server.py"; script.write_text(SERVER)
   async def run():
    c=McpStdioClient(McpServerConfig((sys.executable,str(script)),"test")); await c.start(); result=await c.request("tools/list"); await c.close(); return c.protocol_version,result
   self.assertEqual(asyncio.run(run()),("2025-06-18",{"tools":[]}))
 def test_version_mismatch_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   script=Path(d)/"server.py"; script.write_text(SERVER.replace("2025-06-18","bogus"))
   async def run():
    c=McpStdioClient(McpServerConfig((sys.executable,str(script)),"test"))
    with self.assertRaises(Exception): await c.start()
   asyncio.run(run())

 def test_oversized_jsonl_frame_is_rejected_before_materializing_it(self):
  server = SERVER.replace("  else: r={'tools':[]}\n  print(json.dumps({'jsonrpc':'2.0','id':m['id'],'result':r}),flush=True)", "  else: print('x'*2100000,flush=True); break\n  print(json.dumps({'jsonrpc':'2.0','id':m['id'],'result':r}),flush=True)")
  with tempfile.TemporaryDirectory() as d:
   script=Path(d)/"server.py"; script.write_text(server)
   async def run():
    c=McpStdioClient(McpServerConfig((sys.executable,str(script)),"test")); await c.start()
    with self.assertRaises(Exception): await c.request("tools/list")
    self.assertIsNone(c.process)
   asyncio.run(run())

 def test_request_cancellation_closes_server(self):
  hanging = SERVER.replace("  else: r={'tools':[]}\n  print(json.dumps({'jsonrpc':'2.0','id':m['id'],'result':r}),flush=True)", "  else: time.sleep(60)\n  print(json.dumps({'jsonrpc':'2.0','id':m['id'],'result':r}),flush=True)").replace("import sys,json", "import sys,json,time")
  with tempfile.TemporaryDirectory() as d:
   script=Path(d)/"server.py"; script.write_text(hanging)
   async def run():
    c=McpStdioClient(McpServerConfig((sys.executable,str(script)),"test")); await c.start()
    request=asyncio.create_task(c.request("tools/list")); await asyncio.sleep(0.05); request.cancel()
    with self.assertRaises(asyncio.CancelledError): await request
    self.assertIsNone(c.process)
   asyncio.run(run())
if __name__=="__main__": unittest.main()
