import unittest
from fastapi.testclient import TestClient
from apitopic import app


class TestAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_predict_endpoint(self):
        response = self.client.get("/status")

        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())


if __name__ == "__main__":
    unittest.main()
