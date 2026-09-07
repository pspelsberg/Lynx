from __future__ import annotations
import unittest

try:
    import fastapi  # noqa: F401
except ImportError:
    fastapi = None


@unittest.skipUnless(fastapi is not None, "install the optional api extra")
class ApiTests(unittest.TestCase):
    def test_run_request_is_body_model_with_strict_extra_policy(self):
        from lynx_harness.api import create_app
        app = create_app()
        route = next(route for route in app.routes if getattr(route, "path", None) == "/runs")
        self.assertEqual([param.name for param in route.dependant.body_params], ["request"])
        self.assertEqual(route.dependant.query_params, [])
        schema = app.openapi()["components"]["schemas"]["RunRequest"]
        self.assertFalse(schema["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
