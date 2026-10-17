"""TradingView datafeed helpers using env-based credentials."""

from __future__ import annotations

import os


def get_tv_client():
    """
    Build a ``TvDatafeed`` client from ``TV_USERNAME`` / ``TV_PASSWORD``.

    Raises
    ------
    EnvironmentError
        If credentials are missing.
    ImportError
        If ``tvDatafeed`` is not installed.
    """
    try:
        from tvDatafeed import TvDatafeed
    except ImportError as exc:
        raise ImportError(
            "Install tvDatafeed from GitHub (not on PyPI): "
            "pip install \"tvdatafeed @ git+https://github.com/rongardF/tvdatafeed.git\""
        ) from exc

    username = os.environ.get("TV_USERNAME", "")
    password = os.environ.get("TV_PASSWORD", "")
    if not username or not password:
        raise EnvironmentError(
            "Set TV_USERNAME and TV_PASSWORD in the environment "
            "(or a local .env file). Do not hardcode credentials."
        )
    return TvDatafeed(username, password)
