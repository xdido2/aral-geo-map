"""Access to the Planetary Computer blob store.

The imagery is open data, but the container itself is private: every request
needs a short-lived SAS token. Tokens are issued per container, not per file,
so the token is fetched once and reused - asking for one per asset earns an
HTTP 429 within a few dozen files.
"""
import json
import threading
import time
import urllib.parse
import urllib.request

from . import config as C

_cache = {"token": None, "expires": 0.0}
_lock = threading.Lock()
TTL = 45 * 60          # tokens last about an hour; refresh early


def token(href):
    """A SAS query string valid for the container that holds `href`.

    Locked: with a thread pool, every worker finds the cache empty at the same
    instant, and without the lock all of them go to the signing endpoint at
    once - sixteen requests for one token, and a 429 on a shared address.
    """
    if _cache["token"] and time.time() < _cache["expires"]:
        return _cache["token"]
    with _lock:
        if _cache["token"] and time.time() < _cache["expires"]:
            return _cache["token"]
        return _refresh(href)


def _refresh(href):
    now = time.time()
    url = C.PC_SIGN + urllib.parse.quote(href, safe="")
    with urllib.request.urlopen(url, timeout=60) as r:
        signed = json.load(r)["href"]
    _cache["token"] = urllib.parse.urlsplit(signed).query
    _cache["expires"] = now + TTL
    return _cache["token"]


def sign(href):
    """Same URL, with the token appended. This is what Azure will serve."""
    parts = urllib.parse.urlsplit(href)
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path, token(href), ""))


def asset_url(base_url, product_id, key):
    """Signed URL of one asset of one scene, straight from a manifest row."""
    return sign(f"{base_url}/{C.asset_filename(product_id, key)}")
