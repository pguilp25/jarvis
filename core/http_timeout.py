import aiohttp

_NIM = "integrate.api.nvidia.com"

def http_timeout(url, payload=None):
    """Network-drop-resilient timeout (ckpt-176). A WiFi->LTE handoff silently drops
    the socket; without a guard the call hangs up to `total` (was 1800-3600s).
    NIM is slow -> 5-min window; other providers -> 1-min window.
    STREAMING calls (payload has stream=True): use sock_read (DATA-INACTIVITY) = window —
    a silent drop = no token for that window = dead -> fail fast, never false-killing a
    long LIVE stream (tokens keep the timer alive). NON-streaming: the whole body arrives
    after generation, so a short read-timeout would false-kill a slow gen -> use a total
    ceiling instead. sock_connect=30 catches connect-time drops everywhere."""
    nim = _NIM in (url or "")
    window = 300 if nim else 60
    streaming = bool((payload or {}).get("stream"))
    if streaming:
        return aiohttp.ClientTimeout(total=1200, sock_connect=30, sock_read=window)
    return aiohttp.ClientTimeout(total=(300 if nim else 120), sock_connect=30)
