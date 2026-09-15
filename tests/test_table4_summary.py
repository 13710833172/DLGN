"""Sanity checks for Table 4 summary helpers."""

from pathlib import Path

from src.utils.table4_summary import (
    table4_model_param_ops,
    table4_param_and_ops_display,
    write_table4_csv_md,
)


def test_paper_small_param_ops():
    mcfg = {"num_layers": 6, "neurons_per_layer": 8000}
    name, p, o = table4_model_param_ops(mcfg)
    assert name == "Diff Logic Net (small)"
    assert p == "48000"
    assert o == "48K"
    p2, o2 = table4_param_and_ops_display(mcfg)
    assert (p2, o2) == (p, o)


def test_paper_normal_param_ops():
    mcfg = {"num_layers": 6, "neurons_per_layer": 64000}
    name, p, o = table4_model_param_ops(mcfg)
    assert name == "Diff Logic Net"
    assert p == "384000"
    assert o == "384K"


def test_paper_cifar10_small_param_ops():
    mcfg = {"num_layers": 4, "neurons_per_layer": 12000}
    name, p, o = table4_model_param_ops(mcfg)
    assert name == "Diff Logic Net (small)"
    assert p == "48000"
    assert o == "48K"


def test_paper_cifar10_large_param_ops():
    mcfg = {"num_layers": 5, "neurons_per_layer": 256000}
    name, p, o = table4_model_param_ops(mcfg)
    assert name == "Diff Logic Net (large)"
    assert p == "1280000"
    assert o == "1.28M"


def test_debug_param_ops():
    mcfg = {"num_layers": 3, "neurons_per_layer": 260}
    p, o = table4_param_and_ops_display(mcfg)
    assert p == "12480"
    assert o == "12480"


def test_write_table4_without_space(tmp_path: Path):
    final_test = {
        "best_epoch": 1,
        "best_val_soft_acc": 0.9,
        "best_val_hard_acc": 0.88,
        "best_val_gap": 0.02,
        "test_soft_acc": 0.91,
        "test_hard_acc": 0.9,
        "test_gap": 0.01,
    }
    mcfg = {"num_layers": 2, "neurons_per_layer": 8}
    mname, pstr, ostr = table4_model_param_ops(mcfg)
    csv_p, md_p = write_table4_csv_md(
        run_dir=tmp_path,
        final_test=final_test,
        train_wall_seconds=123.4,
        t_gpu_sec_per_image=None,
        t_gpu_note="not_cuda_device",
        model_name=mname,
        param_str=pstr,
        ops_str=ostr,
    )
    assert csv_p.read_text(encoding="utf-8").splitlines()[0].startswith("Model,")
    assert mname in md_p.read_text(encoding="utf-8")

    # CSV row parses as single data line + header
    lines = csv_p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
