"""IAM token provider — fetches and caches Huawei-Cloud IAM tokens.

Vendored from the submit-host tooling (2026-08-29, M1) and cleaned for the
public repo: no credentials, no orchestra dependencies, requests imported
lazily (module import stays cheap and laptop-safe).

Token acquisition priority (per auth config dict):
  1. auth['x_auth_token']  — static token (if configured, used as-is)
  2. local cache           — indexed by cache key, validated against
                             expires_at with a 5-minute early-refresh buffer
  3. IAM JWT token         — needs account / secret (+ project / enterprise)
  4. IAM password auth     — needs domain_name / username / password /
                             project_id

Endpoints:
  JWT:     POST <internal-iam-jwt-endpoint> (auth.jwt_url in platform config)
           token in the response body's access_token field (~24h validity)
  password POST https://iam.{region}.myhuaweicloud.com/v3/auth/tokens
           token in the X-Subject-Token response header (24h validity)

The cache lives in ~/.cache/climbmix/iam_tokens.json (XDG-style user cache,
NEVER inside the repo). Every API request goes through get_token(): a hit is
a cheap file read, and the expiry buffer transparently rolls the token over
mid-run (production runs are ~32h > 24h token lifetime).

SECURITY: the auth dict contains secrets — it is loaded from the gitignored
platform config (config/remote_ma.example.json documents the schema; the
real file lives at ~/.config/climbmix/remote_ma.json). Never log its values.
"""

import datetime
import json
import os
from logging import getLogger
from typing import Any, Dict, Optional, Tuple

logger = getLogger(__name__)

IAM_CACHE_FILE = "iam_tokens.json"

# Refresh this many minutes BEFORE the recorded expiry (clock skew + fetch
# latency headroom).
_TOKEN_EXPIRY_BUFFER_MINUTES = 5

_DEFAULT_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "climbmix")


class IamTokenProvider:
    """Fetches and caches X-Auth-Token style IAM tokens.

    Thread-safety: get_token() may be called concurrently (the remote
    executor submits jobs from several worker threads). Worst case two
    threads both fetch a fresh token and race the cache write — both tokens
    are valid, so the last-write-wins outcome is harmless.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        self._cache_file = os.path.join(cache_dir, IAM_CACHE_FILE)

    # ── public API ────────────────────────────────────────────────────────

    def get_token(self, auth_config: Dict[str, Any],
                  *, force_refresh: bool = False) -> str:
        """Return a usable auth token per the priority chain in the module
        docstring. Raises Exception when no acquisition method is complete."""
        static_token = str(auth_config.get("x_auth_token") or "").strip()
        if static_token:
            logger.debug("using static x_auth_token")
            return static_token

        account = str(auth_config.get("account") or "").strip()
        jwt_cache_key = f"jwt_{account}" if account else None
        project_id = str(auth_config.get("project_id") or "").strip()
        region = str(auth_config.get("region") or "cn-north-4").strip()
        iam_cache_key = f"{region}_{project_id}"

        if not force_refresh:
            if jwt_cache_key:
                cached = self._load_cached(jwt_cache_key)
                if cached and self._is_valid(cached):
                    return cached["token"]
            cached = self._load_cached(iam_cache_key)
            if cached and self._is_valid(cached):
                return cached["token"]

        secret = str(auth_config.get("secret") or "").strip()
        if account and secret:
            jwt_url = str(auth_config.get("jwt_url") or "").strip()
            if not jwt_url:
                raise ValueError(
                    "auth.jwt_url is missing in the platform config — the "
                    "JWT acquisition path needs it (see "
                    "config/remote_ma.example.json)")
            jwt_project = (str(auth_config.get("project") or "").strip()
                           or account)
            enterprise = str(auth_config.get("enterprise") or "").strip()
            logger.info("fetching IAM JWT token (account configured)")
            token, expires_at = self._fetch_jwt_token(
                account=account, secret=secret, project=jwt_project,
                enterprise=enterprise, jwt_url=jwt_url)
            if jwt_cache_key:
                self._save_cache(jwt_cache_key, token, expires_at)
            logger.info("JWT token valid until %s", expires_at)
            return token

        domain_name = str(auth_config.get("domain_name") or "").strip()
        username = str(auth_config.get("username") or "").strip()
        password = str(auth_config.get("password") or "").strip()

        missing = [k for k, v in {
            "domain_name": domain_name, "username": username,
            "password": password, "project_id": project_id,
        }.items() if not v]
        if missing:
            raise Exception(
                "cannot acquire an IAM token, auth config incomplete "
                f"(missing: {missing}). Configure one of: "
                "auth.x_auth_token (static), auth.account + auth.secret "
                "(JWT), or auth.domain_name + auth.username + "
                "auth.password (password auth)")

        logger.info("fetching IAM token via password auth (region=%s)",
                    region)
        token, expires_at = self._fetch_from_iam(
            domain_name=domain_name, username=username, password=password,
            project_id=project_id, region=region)
        self._save_cache(iam_cache_key, token, expires_at)
        logger.info("IAM token valid until %s", expires_at)
        return token

    def invalidate_cached_token(self, auth_config: Dict[str, Any]) -> None:
        """Drop every cached token for this auth config (used when a
        previously returned token is rejected with HTTP 401/403)."""
        project_id = str(auth_config.get("project_id") or "").strip()
        region = str(auth_config.get("region") or "cn-north-4").strip()
        self._delete_cache(f"{region}_{project_id}")
        account = str(auth_config.get("account") or "").strip()
        if account:
            self._delete_cache(f"jwt_{account}")

    # ── fetchers ──────────────────────────────────────────────────────────

    def _fetch_jwt_token(self, account: str, secret: str, project: str,
                         enterprise: str, jwt_url: str) -> Tuple[str, str]:
        """POST the internal IAM JWT endpoint (URL supplied by the caller
        from the platform config — it is deployment-specific and must not
        live in this public repo); token lands in the response body's
        access_token field. Returns (token, expires_at ISO 8601)."""
        import requests

        payload = {
            "data": {
                "type": "jwt-token",
                "attributes": {
                    "account": account,
                    "secret": secret,
                    "project": project,
                    "enterprise": enterprise,
                },
            },
        }
        try:
            resp = requests.post(
                jwt_url, json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30, verify=False,
                proxies={"http": None, "https": None})
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            err_msg = str(e)
            status = e.response.status_code if e.response is not None else "?"
            try:
                body = e.response.json()
                err_msg = (body.get("message")
                           or body.get("error", {}).get("message")
                           or str(e))
            except Exception:
                pass
            raise Exception(
                f"IAM JWT token fetch failed (HTTP {status}): {err_msg}")
        except requests.exceptions.ConnectionError:
            raise Exception(
                "cannot reach the IAM JWT token service — check network")
        except requests.exceptions.Timeout:
            raise Exception("IAM JWT token request timed out (30s)")

        data = resp.json()
        token = str(data.get("access_token", "")).strip()
        if not token:
            raise Exception(
                "IAM JWT token response has no access_token field")
        return token, str(data.get("expires_at", "")).strip()

    def _fetch_from_iam(self, domain_name: str, username: str, password: str,
                        project_id: str, region: str) -> Tuple[str, str]:
        """POST the public IAM password-auth endpoint; token lands in the
        X-Subject-Token response header. Returns (token, expires_at)."""
        import requests

        iam_url = (f"https://iam.{region}.myhuaweicloud.com"
                   f"/v3/auth/tokens?nocatalog=true")
        payload = {
            "auth": {
                "identity": {
                    "methods": ["password"],
                    "password": {"user": {
                        "name": username, "password": password,
                        "domain": {"name": domain_name},
                    }},
                },
                "scope": {"project": {"id": project_id}},
            },
        }
        try:
            resp = requests.post(
                iam_url, json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30, verify=False,
                proxies={"http": None, "https": None})
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            err_msg = str(e)
            status = e.response.status_code if e.response is not None else "?"
            try:
                err_msg = (e.response.json()
                           .get("error", {}).get("message") or str(e))
            except Exception:
                pass
            raise Exception(
                f"IAM password auth failed (HTTP {status}): {err_msg}")
        except requests.exceptions.ConnectionError:
            raise Exception(
                f"cannot reach iam.{region}.myhuaweicloud.com — check "
                "network / auth.region")
        except requests.exceptions.Timeout:
            raise Exception("IAM request timed out (30s)")

        token = resp.headers.get("X-Subject-Token", "").strip()
        if not token:
            raise Exception("IAM response has no X-Subject-Token header")
        expires_at = ""
        try:
            expires_at = resp.json().get("token", {}).get("expires_at", "")
        except Exception:
            pass
        return token, str(expires_at).strip()

    # ── cache management ──────────────────────────────────────────────────

    def _load_cached(self, cache_key: str) -> Optional[Dict[str, Any]]:
        try:
            if not os.path.exists(self._cache_file):
                return None
            with open(self._cache_file, "r", encoding="utf-8") as f:
                return json.load(f).get(cache_key)
        except Exception:
            return None

    def _save_cache(self, cache_key: str, token: str, expires_at: str) -> None:
        try:
            all_cached: Dict[str, Any] = {}
            if os.path.exists(self._cache_file):
                try:
                    with open(self._cache_file, "r", encoding="utf-8") as f:
                        all_cached = json.load(f)
                except Exception:
                    pass
            all_cached[cache_key] = {
                "token": token,
                "expires_at": expires_at,
                "cached_at": datetime.datetime.now(
                    datetime.timezone.utc).isoformat(),
            }
            os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(all_cached, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning("IAM token cache write failed: %s", e)

    def _delete_cache(self, cache_key: str) -> None:
        try:
            if not os.path.exists(self._cache_file):
                return
            with open(self._cache_file, "r", encoding="utf-8") as f:
                all_cached = json.load(f)
            if cache_key not in all_cached:
                return
            all_cached.pop(cache_key, None)
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(all_cached, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning("IAM token cache delete failed: %s", e)

    def _is_valid(self, cached: Dict[str, Any]) -> bool:
        """True when the cached token outlives now + the refresh buffer."""
        expires_at_str = cached.get("expires_at", "")
        if not expires_at_str:
            return False
        try:
            expires_dt = datetime.datetime.fromisoformat(expires_at_str)
            if expires_dt.tzinfo is None:
                expires_dt = expires_dt.replace(
                    tzinfo=datetime.timezone.utc)
            now = datetime.datetime.now(datetime.timezone.utc)
            buffer = datetime.timedelta(minutes=_TOKEN_EXPIRY_BUFFER_MINUTES)
            return now + buffer < expires_dt
        except Exception as e:
            logger.warning("token expiry parse failed (%s): %s",
                           expires_at_str, e)
            return False
