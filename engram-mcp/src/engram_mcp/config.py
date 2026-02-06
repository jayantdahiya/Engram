import os


def get_config() -> dict:
    url = os.environ.get("ENGRAM_API_URL", "http://localhost:8000")
    username = os.environ.get("ENGRAM_USERNAME")
    password = os.environ.get("ENGRAM_PASSWORD")
    if not username or not password:
        raise RuntimeError("ENGRAM_USERNAME and ENGRAM_PASSWORD environment variables are required")
    return {"api_url": url.rstrip("/"), "username": username, "password": password}
