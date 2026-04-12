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
    assert args.history_window == 12
    assert args.all_images is False
    assert args.prompt_token_window == 512
    assert args.init_t == 0.0
    assert args.epochs == 1
    assert args.fit_iters is None
    assert args.update_mode == "auto"


def test_parser_accepts_deterministic_coding_flag():
    parser = build_parser()
    args = parser.parse_args(["--phase", "init", "--deterministic-coding"])

    assert args.deterministic_coding is True


def test_parser_accepts_history_window_and_image_mode_flags():
    parser = build_parser()
    args = parser.parse_args(
        ["--phase", "init", "--history-window", "18", "--all-images", "--prompt-token-window", "256"],
    )

    assert args.history_window == 18
    assert args.all_images is True
    assert args.prompt_token_window == 256


def test_parser_accepts_epochs_and_fit_iters_flags():
    parser = build_parser()
    args = parser.parse_args(["--phase", "tune", "--epochs", "3", "--fit-iters", "9", "--init-t", "0.35"])

    assert args.epochs == 3
    assert args.fit_iters == 9
    assert args.init_t == 0.35


def test_parser_accepts_update_mode_flag():
    parser = build_parser()
    args = parser.parse_args(["--phase", "retask", "--update-mode", "online"])

    assert args.update_mode == "online"
