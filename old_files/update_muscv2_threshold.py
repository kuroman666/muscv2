import os
import re
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 更新 1_infer_muscv2_no_gt.py 的 argparse 設定
def patch_muscv2_py():
    py_path = os.path.join(SCRIPT_DIR, "1_infer_muscv2_no_gt.py")
    if not os.path.isfile(py_path):
        print(f"[略過] 找不到 {py_path}")
        return

    with open(py_path, "r", encoding="utf-8") as f:
        s = f.read()

    s = s.replace(
        'choices=["hybrid", "mean_std", "percentile", "top_k", "fixed"]',
        'choices=["auto", "hybrid", "mean_std", "percentile", "top_k", "fixed"]',
    )
    s = s.replace('default="hybrid"', 'default="auto"')
    s = re.sub(r'("--threshold",\s*type=float,\s*default=)0\.25', r"\g<1>0.0", s)
    s = re.sub(r'("--min_threshold",\s*type=float,\s*default=)0\.27', r"\g<1>0.0", s)

    with open(py_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)
    print("[✓ 已更新] 1_infer_muscv2_no_gt.py (argparse 支援 auto 且預設 threshold=0.0)")


# 2. 批次更新目錄下所有 run_*.sh 執行檔
PARAM_PATTERNS = [
    (
        r'THRESH_MODE="(\$\{THRESH_MODE:-)[^"]+("|\})"',
        'THRESH_MODE="${THRESH_MODE:-auto}"',
    ),
    (
        r'THRESH_MODE="(fixed|hybrid|min_3sigma_modz)"',
        'THRESH_MODE="auto"',
    ),
    (r'DEFAULT_THRESHOLD="0\.18"', 'DEFAULT_THRESHOLD="0.0"'),
    (r'THRESHOLD="\$\{THRESHOLD:-\}"', 'THRESHOLD="${THRESHOLD:-0.0}"'),
    (r'THRESHOLD="\$\{THRESHOLD:-0\}"', 'THRESHOLD="${THRESHOLD:-0.0}"'),
    (r"THRESHOLD=0\.25", "THRESHOLD=0.0"),
    (r"MIN_THRESHOLD=0\.27", "MIN_THRESHOLD=0.0"),
]

OLD_STATS_BLOCK_RE = re.compile(
    r"(m1std\s*=\s*mean_s\s*\+\s*1\.0\s*\*\s*std_s\n"
    r"\s*m2std\s*=\s*mean_s\s*\+\s*2\.0\s*\*\s*std_s\n"
    r"\s*m3std\s*=\s*mean_s\s*\+\s*3\.0\s*\*\s*std_s)"
)

NEW_STATS_CALC = """_all_eval_scores = np.array([float(r["Max_Score"]) for r in records], dtype=np.float64)
_mean_all = float(np.mean(_all_eval_scores))
_std_all = float(np.std(_all_eval_scores))
_th_3sigma = _mean_all + 3.0 * _std_all
_med_all = float(np.median(_all_eval_scores))
_mad_all = float(np.median(np.abs(_all_eval_scores - _med_all)))
_th_modz = _med_all + (3.5 / 0.6745) * _mad_all
if _th_3sigma <= _th_modz:
    _low_th, _low_name = _th_3sigma, "3-Sigma"
    _high_th, _high_name = _th_modz, "Mod-Z(3.5)"
else:
    _low_th, _low_name = _th_modz, "Mod-Z(3.5)"
    _high_th, _high_name = _th_3sigma, "3-Sigma"
_n_all = int(_all_eval_scores.size)
_low_cnt = int(np.sum(_all_eval_scores >= _low_th))
_high_cnt = int(np.sum(_all_eval_scores >= _high_th))
_high_pct = (_high_cnt / max(1, _n_all)) * 100.0
_high_p = float(np.mean(_all_eval_scores < _high_th) * 100.0)
_high_cut_str = (
    f"高於最高分({float(np.max(_all_eval_scores)):.4f})，切出 0 張"
    if _high_cnt == 0
    else f"切在 Rank {_high_cnt}/{_n_all} 之後 (共 {_high_cnt} 張, {_high_pct:.2f}%, 約 P{_high_p:.2f})"
)
m1std = mean_s + 1.0 * std_s
m2std = mean_s + 2.0 * std_s
m3std = mean_s + 3.0 * std_s"""

OLD_PRINT_LIST_RE = re.compile(
    r'([ \t]*)f"([ \t]*📐[^"]*門檻參考值\s*:\s*)mean\+1\*std=\{m1std:\.4f\}\s*\|\s*mean\+2\*std=\{m2std:\.4f\}\s*\|\s*mean\+3\*std=\{m3std:\.4f\}",'
)

NEW_PRINT_LIST_REPL = (
    r'\1f"\2 3-Sigma(Shewhart 1931)={_th_3sigma:.4f} | Mod-Z 3.5(Iglewicz 1993)={_th_modz:.4f}",\n'
    r'\1f"  🎯 兩者較小值參考         : [{_low_name}] = {_low_th:.4f} (切出 {_low_cnt}/{_n_all} 張, {_low_cnt/max(1,_n_all)*100.0:.2f}%)",\n'
    r'\1f"  📍 兩者較高值切點         : [{_high_name}] = {_high_th:.4f} -> {_high_cut_str}",'
)

OLD_PRINT_DIRECT_RE = re.compile(
    r'([ \t]*)print\(f"([ \t]*📐[^"]*門檻參考值\s*:\s*)mean\+1\*std=\{m1std:\.4f\}\s*\|\s*mean\+2\*std=\{m2std:\.4f\}\s*\|\s*mean\+3\*std=\{m3std:\.4f\}"\)'
)

NEW_PRINT_DIRECT_REPL = (
    r'\1print(f"\2 3-Sigma(Shewhart 1931)={_th_3sigma:.4f} | Mod-Z 3.5(Iglewicz 1993)={_th_modz:.4f}")\n'
    r'\1print(f"  🎯 兩者較小值參考            : [{_low_name}] = {_low_th:.4f} (切出 {_low_cnt}/{_n_all} 張, {_low_cnt/max(1,_n_all)*100.0:.2f}%)")\n'
    r'\1print(f"  📍 兩者較高值切點            : [{_high_name}] = {_high_th:.4f} -> {_high_cut_str}")'
)


def patch_all_sh():
    sh_files = sorted(glob.glob(os.path.join(SCRIPT_DIR, "run_*.sh")))
    if not sh_files:
        print("[提示] 目錄下未找到任何 run_*.sh")
        return

    for filepath in sh_files:
        fname = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            original = f.read()

        content = original
        for pat, repl in PARAM_PATTERNS:
            content = re.sub(pat, repl, content)

        content = OLD_STATS_BLOCK_RE.sub(NEW_STATS_CALC, content)
        content = OLD_PRINT_LIST_RE.sub(NEW_PRINT_LIST_REPL, content)
        content = OLD_PRINT_DIRECT_RE.sub(NEW_PRINT_DIRECT_REPL, content)

        if content != original:
            with open(filepath, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)
            print(f"[✓ 已更新] {fname}")
        else:
            print(f"[無變動] {fname}")


if __name__ == "__main__":
    patch_muscv2_py()
    patch_all_sh()
    print("\n🔍 執行 `grep -n \"THRESH_MODE=\" run_*.sh` 即可雙重確認！")