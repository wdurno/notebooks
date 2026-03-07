from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "demo" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.experiment_interface import build_parser


def test_parser_defaults_deterministic_coding_to_false():
    parser = build_parser()
    args = parser.parse_args(["--phase", "init"])

    assert args.deterministic_coding is False


def test_parser_accepts_deterministic_coding_flag():
    parser = build_parser()
    args = parser.parse_args(["--phase", "init", "--deterministic-coding"])

    assert args.deterministic_coding is True

