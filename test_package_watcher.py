#!/usr/bin/env python3
"""Focused unit tests for Package Watcher (stubs Pi/Ollama packages when run off-device)."""

import os
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch


def _install_stubs():
    if "cv2" not in sys.modules:
        cv2 = types.ModuleType("cv2")
        cv2.IMWRITE_JPEG_QUALITY = 1
        cv2.FONT_HERSHEY_SIMPLEX = 0
        cv2.COLOR_BGR2GRAY = 0
        cv2.THRESH_BINARY = 0
        cv2.CV_64F = 0
        cv2.resize = lambda img, sz: img
        cv2.imencode = lambda ext, img, *a: (True, b"jpg")
        def _imwrite(path, img, *a):
            with open(path, "wb") as f:
                f.write(b"jpg")
            return True

        cv2.imwrite = _imwrite
        cv2.imread = lambda path: b"img"
        sys.modules["cv2"] = cv2
    if "requests" not in sys.modules:
        req = types.ModuleType("requests")
        req.RequestException = Exception
        sys.modules["requests"] = req
    if "tinydb" not in sys.modules:
        tdb = types.ModuleType("tinydb")
        _STORE = {}

        class TinyDB:
            def __init__(self, path):
                self.path = path
                _STORE.setdefault(path, [])

            def insert(self, row):
                with open(self.path, "a") as f:
                    pass
                _STORE[self.path].append(dict(row))

            def all(self):
                return list(_STORE[self.path])

            def remove(self, cond):
                _STORE[self.path] = [r for r in _STORE[self.path] if not cond(r)]

        tdb.TinyDB = TinyDB
        sys.modules["tinydb"] = tdb
    if "langgraph" not in sys.modules:
        lg = types.ModuleType("langgraph")
        lgg = types.ModuleType("langgraph.graph")
        lgg.START = "__start__"
        lgg.END = "__end__"

        class Compiled:
            def __init__(self, nodes, edges, conds):
                self.nodes, self.edges, self.conds = nodes, edges, conds

            def invoke(self, state):
                s = dict(state)
                cur = lgg.START
                while cur != lgg.END:
                    if cur in self.conds:
                        fn, mapping = self.conds[cur]
                        nxt = fn(s)
                        cur = mapping[nxt] if isinstance(mapping, dict) else nxt
                    elif cur in self.edges:
                        cur = self.edges[cur]
                    else:
                        break
                    if cur == lgg.END:
                        break
                    res = self.nodes[cur](s)
                    if res:
                        s.update(res)
                return s

        class StateGraph:
            def __init__(self, _):
                self.nodes, self.edges, self.conds = {}, {}, {}

            def add_node(self, name, fn):
                self.nodes[name] = fn

            def add_edge(self, src, dst):
                self.edges[src] = dst

            def add_conditional_edges(self, src, fn, mapping):
                self.conds[src] = (fn, mapping)

            def compile(self):
                return Compiled(self.nodes, self.edges, self.conds)

        lgg.StateGraph = StateGraph
        sys.modules["langgraph"] = lg
        sys.modules["langgraph.graph"] = lgg
    if "langchain_ollama" not in sys.modules:
        lo = types.ModuleType("langchain_ollama")

        class ChatOllama:
            def __init__(self, **kw):
                pass

            def invoke(self, prompt):
                return types.SimpleNamespace(content="Bring it inside soon.")

        lo.ChatOllama = ChatOllama
        sys.modules["langchain_ollama"] = lo


_install_stubs()
import package_watcher as pw  # noqa: E402
import tool_agent as ta       # noqa: E402


class TestPackageWatcher(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        pw.BASE_DIR = self.tmp.name
        pw.EVENTS_DIR = os.path.join(self.tmp.name, "events")
        pw.THUMBS_DIR = os.path.join(pw.EVENTS_DIR, "thumbs")
        pw.DB_PATH = os.path.join(self.tmp.name, "events.json")
        pw.TEMP_DB_PATH = os.path.join(self.tmp.name, "temperature.json")
        pw.STATE_PATH = os.path.join(self.tmp.name, "state.json")
        pw.LABELS_PATH = os.path.join(self.tmp.name, "labels.json")
        os.makedirs(pw.EVENTS_DIR, exist_ok=True)
        pw.runtime["active_package"] = None
        pw.runtime["snooze_until"] = 0.0
        pw.runtime["last_temp_alert"] = 0.0
        pw._dbs.clear()
        pw.runtime["temp"] = {"c": 30.0, "f": 86.0, "humidity": 55.0, "time": time.time()}
        pw.runtime["temp_recent"].clear()

    def tearDown(self):
        self.tmp.cleanup()

    def test_parse_package_response(self):
        self.assertEqual(pw.parse_package_response("Yes, a brown cardboard box sits by the door."),
                         (True, "A brown cardboard box sits by the door."))
        self.assertEqual(pw.parse_package_response("No, the porch is empty."),
                         (False, "The porch is empty."))
        self.assertFalse(pw.parse_package_response("There is no package on the mat.")[0])
        self.assertTrue(pw.parse_package_response("A small Amazon parcel is on the steps.")[0])

    def test_delivery_sitting_and_pickup_lifecycle(self):
        img = os.path.join(pw.EVENTS_DIR, "20261126_120000_001.jpg")
        with open(img, "wb") as f:
            f.write(b"jpg")

        with patch.object(pw, "send_ntfy") as ntfy, patch.object(pw, "publish_mqtt"), \
             patch.object(pw, "llm_answer") as llm:
            llm.invoke.return_value = types.SimpleNamespace(content="Bring it inside soon because it is 86F.")

            # 1. Arrival
            d1 = pw.decide({
                "event_id": "20261126_120000_001", "mode": "auto", "trigger": "motion",
                "image_path": img, "package_present": True, "description": "Brown cardboard box.",
            })
            self.assertEqual(d1["outcome"], "delivered")
            self.assertIsNotNone(pw.runtime["active_package"])
            self.assertIn("86F", d1["advice"])

            # 2. Still sitting outside -> no duplicate delivery alert
            d2 = pw.decide({
                "event_id": "20261126_120500_001", "mode": "auto", "trigger": "motion",
                "image_path": img, "package_present": True, "description": "Brown cardboard box.",
            })
            self.assertEqual(d2["outcome"], "still_present")
            self.assertFalse(d2["alert"])

            # 3. Picked up
            d3 = pw.decide({
                "event_id": "20261126_121000_001", "mode": "auto", "trigger": "motion",
                "image_path": img, "package_present": False, "description": "Empty porch.",
            })
            self.assertEqual(d3["outcome"], "picked_up")
            self.assertIsNone(pw.runtime["active_package"])
            ntfy.assert_not_called()

    def test_graph_end_to_end_and_web_status(self):
        with patch.object(pw, "describe_image", return_value="Yes, a parcel sits on the mat."), \
             patch.object(pw, "send_ntfy") as ntfy, \
             patch.object(pw, "publish_mqtt"), \
             patch.object(pw, "UPLOAD_TO_S3", False), \
             patch.object(pw, "llm_answer") as llm:
            llm.invoke.return_value = types.SimpleNamespace(content="Bring it in soon.")
            out = pw.run_event("auto", "motion", frame=b"img", motion=0.25)
            self.assertEqual(out["outcome"], "delivered")
            self.assertEqual(ntfy.call_count, 1)

            st = pw.web_status()
            self.assertEqual(st["outside_now"], 1)
            self.assertEqual(st["today"]["deliveries"], 1)
            self.assertEqual(len(pw.web_events("delivered")), 1)

            self.assertTrue(pw.mark_package_picked_up("dashboard", notify_phone=False))
            self.assertEqual(pw.web_status()["outside_now"], 0)
            self.assertEqual(pw.web_status()["today"]["picked_up"], 1)

    def test_stale_temp_alert_skipped_after_pickup_and_combined_ask(self):
        pkg = {
            "id": "pkg2", "arrived_at": time.time() - 300, "description": "Box",
            "image": "", "notes": "", "advice": "", "temp_c": 22.0, "humidity": 50.0,
            "last_temp_alert": 0.0,
        }
        pw.runtime["active_package"] = pkg

        def clear_during_llm(_prompt):
            pw.mark_package_picked_up("dashboard", notify_phone=False)
            return types.SimpleNamespace(content="Bring it inside.")

        with patch.object(pw, "llm_answer") as llm, patch.object(pw, "send_ntfy") as ntfy, \
             patch.object(pw, "publish_mqtt"):
            llm.invoke.side_effect = clear_during_llm
            pw.check_package_temp(31.0, 55.0, time.time())
            self.assertIsNone(pw.runtime["active_package"])
            ntfy.assert_not_called()

        # Combined history + bring-in question still calls advise_package
        with patch.object(pw, "advise_package", return_value="Bring it in now.") as adv:
            r = pw.ask("When did the package arrive and should I bring it inside?")
            self.assertEqual(r["answer"], "Bring it in now.")
            adv.assert_called_once()

    def test_snooze_endpoint_validation_and_event_pruning(self):
        import io
        import json as _json

        # Insert an expired event and verify record_event prunes it
        with pw.db_lock:
            pw._db(pw.DB_PATH).insert({"id": "old", "timestamp": "2020-01-01T00:00:00+00:00", "expires_at": 1})
        with patch.object(pw, "publish_mqtt"):
            pw.record_event("snapshot", "snapshot", "manual")
        ids = [e["id"] for e in pw.db_all()]
        self.assertNotIn("old", ids)

        # Exercise Handler.do_POST /api/snooze validation
        handler_cls = pw.make_web_handler("k")

        def post_snooze(payload):
            raw = _json.dumps(payload).encode()
            h = handler_cls.__new__(handler_cls)
            h.path = "/api/snooze?key=k"
            h.headers = {"Content-Length": str(len(raw))}
            h.rfile = io.BytesIO(raw)
            h.wfile = io.BytesIO()
            sent = []
            h.send_response = lambda code: sent.append(code)
            h.send_header = lambda *a: None
            h.end_headers = lambda: None
            h.do_POST()
            return sent[0]

        self.assertEqual(post_snooze({"minutes": "abc"}), 400)
        self.assertEqual(post_snooze({"minutes": 999}), 400)
        self.assertEqual(post_snooze({"minutes": 30}), 200)

    def test_advise_persists_user_notes_and_temp_alert(self):
        pw.runtime["active_package"] = {
            "id": "pkg1", "arrived_at": time.time() - 600, "description": "Medium box",
            "image": "", "notes": "", "advice": "", "temp_c": 22.0, "humidity": 50.0,
            "last_temp_alert": 0.0,
        }
        with patch.object(pw, "llm_answer") as llm, patch.object(pw, "send_ntfy") as ntfy, \
             patch.object(pw, "publish_mqtt"):
            llm.invoke.return_value = types.SimpleNamespace(
                content="Bring it inside now; 86F will melt chocolate."
            )
            res = pw.tool_advise_bring_in({"package_info": "chocolate"}, "It has chocolate, bring it in?")
            self.assertIn("chocolate", res["answer"])
            self.assertEqual(pw.runtime["active_package"]["notes"], "chocolate")

            # Follow-up temperature spike triggers warning once within cooldown
            pw.check_package_temp(30.0, 55.0, time.time())
            self.assertEqual(ntfy.call_count, 1)
            pw.check_package_temp(31.0, 55.0, time.time() + 60)
            self.assertEqual(ntfy.call_count, 1)

    def test_tool_agent_safety_and_routing(self):
        agent = pw.build_tool_agent()
        self.assertIn("Package Watcher", agent.system_prompt())
        self.assertIn("advise_bring_in", agent.system_prompt())

        with patch.object(agent, "choose_calls", return_value=([("snooze_alerts", {"minutes": 30})], "")):
            r = agent.ask("Snooze alerts for 30 minutes", session="test")
            self.assertTrue(r["pending"])
            self.assertEqual(r["tool_calls"][0]["status"], "needs_confirmation")

        with patch.object(agent, "choose_calls", return_value=([("snooze_alerts", {"minutes": 999})], "")):
            r = agent.ask("Snooze for 999 minutes", session="test2")
            self.assertEqual(r["tool_calls"][0]["status"], "refused")

        # Validate parse_calls fallback for text-embedded tool JSON
        calls = ta.parse_calls({"content": '{"name": "advise_bring_in", "arguments": {"package_info": "milk"}}'})
        self.assertEqual(calls, [("advise_bring_in", {"package_info": "milk"})])


if __name__ == "__main__":
    unittest.main(verbosity=2)
