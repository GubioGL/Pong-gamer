"""Atalho para o jogo: `python pongamer.py` (o mesmo que `python -m games.pong.play`)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from games.pong.play import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
