from __future__ import annotations
import asyncio,tempfile
from pathlib import Path
import unittest,sys
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from lynx_harness.kernel import PythonKernel
class KernelTests(unittest.TestCase):
 def test_state_persists_across_calls(self):
  with tempfile.TemporaryDirectory() as d:
   async def run():
    k=PythonKernel(Path(d)); await k.execute("x=40"); result=await k.execute("_=x+2"); await k.close(); return result
   self.assertEqual(asyncio.run(run())["value"],"42")
 def test_filesystem_metadata_cannot_escape_workspace(self):
  with tempfile.TemporaryDirectory() as d:
   async def run():
    k=PythonKernel(Path(d)); result=await k.execute("import os\nos.listdir('/tmp')"); await k.close(); return result
   self.assertFalse(asyncio.run(run())["ok"])
 def test_output_is_bounded_without_pipe_deadlock(self):
  with tempfile.TemporaryDirectory() as d:
   async def run():
    k=PythonKernel(Path(d), max_output=10_000); result=await k.execute("print('x' * 200000)"); await k.close(); return result
   self.assertLessEqual(len(asyncio.run(run())["output"]), 10_000)

 def test_network_disabled_by_default(self):
  with tempfile.TemporaryDirectory() as d:
   async def run():
    k=PythonKernel(Path(d)); result=await k.execute("import socket\nsocket.create_connection((\"example.com\",80),1)"); await k.close(); return result
   self.assertFalse(asyncio.run(run())["ok"])
 def test_code_cannot_self_enable_networking(self):
  with tempfile.TemporaryDirectory() as d:
   async def run():
    k=PythonKernel(Path(d)); result=await k.execute("import os\nos.environ['LYNX_KERNEL_NETWORK']='1'\nimport socket\n_=socket.socket()\n"); await k.close(); return result
   self.assertFalse(asyncio.run(run())["ok"])
 def test_cancellation_closes_worker(self):
  with tempfile.TemporaryDirectory() as d:
   async def run():
    k=PythonKernel(Path(d),timeout_s=30); task=asyncio.create_task(k.execute("import time; time.sleep(20)"))
    await asyncio.sleep(0.1); task.cancel()
    with self.assertRaises(asyncio.CancelledError): await task
    return k.process
   self.assertIsNone(asyncio.run(run()))
 def test_timeout_closes_worker(self):
  with tempfile.TemporaryDirectory() as d:
   async def run():
    k=PythonKernel(Path(d),timeout_s=1); result=None
    try: result=await k.execute("while True: pass")
    except Exception: return k.process
   self.assertIsNone(asyncio.run(run()))
if __name__=="__main__": unittest.main()
