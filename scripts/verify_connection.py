"""verify_connection.py — Validate IBKR TWS/Gateway connectivity.

Reads credentials from .env and tests connection to the IBKR socket.
Directive: 01_environment_setup.md
"""

import os
import sys
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
except ImportError:
    print("ERROR: python-dotenv is not installed. Run `uv sync` first.")
    sys.exit(1)

try:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
except ImportError:
    print("ERROR: ibapi is not installed. Run `uv sync` first.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env")

IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", 4002))
IBKR_CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", 1))


class VerifyApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.success = False

    def nextValidId(self, orderId: int):
        print("=" * 50)
        print(f"  IBKR Connection Verified ✓ [Port: {IBKR_PORT}]")
        print("=" * 50)
        self.success = True
        self.disconnect()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        if errorCode != 2104 and errorCode != 2106:  # ignore info messages
            print(f"IBKR Error [{errorCode}]: {errorString}")


def main() -> None:
    print(
        f"Connecting to IBKR Gateway/TWS on {IBKR_HOST}:{IBKR_PORT} (Client ID: {IBKR_CLIENT_ID})..."
    )

    app = VerifyApp()

    try:
        app.connect(IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID)
    except Exception as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)

    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()

    # Wait for connection
    timeout = 5
    start_time = time.time()
    while not app.success and time.time() - start_time < timeout:
        time.sleep(0.1)

    if not app.success:
        print("\n❌ ERROR: Connection timed out.")
        print(
            "Ensure TWS or IB Gateway is running and configured to accept socket connections on the configured port."
        )
        sys.exit(1)

    print("\n✅ All checks passed. You are ready to go.\n")


if __name__ == "__main__":
    main()
