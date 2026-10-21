from jarviseo.api.__main__ import HOST, parse_args


def test_defaults_open_browser_on_8000():
    args = parse_args([])
    assert args.port == 8000 and not args.no_browser and not args.reload


def test_options():
    args = parse_args(["--port", "8100", "--no-browser", "--reload"])
    assert args.port == 8100 and args.no_browser and args.reload


def test_server_listens_only_on_this_pc():
    """각자 PC 에서만 쓴다. 같은 와이파이의 다른 PC 에서는 들어올 수 없다."""
    assert HOST == "127.0.0.1"
