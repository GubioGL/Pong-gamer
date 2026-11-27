"""Atalho para o jogo da cobrinha: `python cobrinha.py` (o mesmo que `python -m games.snake.play`)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from games.snake.play import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
