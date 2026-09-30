#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════
#  test_cluster_semantics.py — cluster_peek 数据面/紧凑渲染 + recipe_report
#  §2b 接线的回归测试 (stub 池, 零 NPU 零真实数据)
#
#  覆盖: collect 的统计/域交叉/抽样口径 (含无域 doc 的分母排除)、
#  render_full 与历史 CLI 输出同构的关键结构、render_brief 紧凑表、
#  recipe_report._cluster_semantics 的缓存生成/复用/优雅跳过路径。
# ═══════════════════════════════════════════════════════════════════════
import json
import os
import sys
import tempfile
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ── stub 池: cluster_peek 只消费这五个属性/方法 ──


class _Schema:
    domain_names = ["数学", "化学", "生物学", "物理"]
    quality_cols = ["stem_relevance", "knowledge_value",
                    "notation_fidelity", "rigor_coherence", "noise_level"]


class FakeMM:
    """8 docs / 2 簇 / 4 域, doc 6 无域标签 (-1) — 域分母须排除它。"""

    def __init__(self, pool_dir):
        assert os.path.isdir(pool_dir)
        self.num_docs = 8
        # 域标签 (池级): doc6 = -1 (无域)
        self.cluster_labels = np.array([0, 0, 0, 3, 1, 1, -1, 2])
        self.doc_char_counts = np.array(
            [400, 800, 1200, 1600, 2000, 2400, 2800, 100], dtype=np.float64)
        q = np.tile(np.array([3.0, 4.0, 5.0, 4.0, 5.0]), (8, 1))
        q[4:] += 1.0            # 簇 1 质量分整体 +1 → 可断言
        self.quality_scores = q
        self._schema = _Schema()

    def read_texts(self, idx, verbose=False):
        return [f"doc {int(i)}: " + "content " * 8 for i in idx]


# 注入 stub (须在 import cluster_peek 之前)
_stub = types.ModuleType("climbmix.data.metadata_manager")
_stub.ShardMetadataManager = FakeMM
sys.modules.setdefault("climbmix", types.ModuleType("climbmix"))
sys.modules["climbmix.data"] = types.ModuleType("climbmix.data")
sys.modules["climbmix.data.metadata_manager"] = _stub

sys.path.insert(0, HERE)
import cluster_peek as cp           # noqa: E402
import recipe_report as rr          # noqa: E402


def main():
    tmp = tempfile.mkdtemp(prefix="cluster_sem_test_")
    pool_dir = os.path.join(tmp, "pool")
    os.makedirs(pool_dir)
    cache_npz = os.path.join(tmp, "cluster_cache.npz")
    # 簇 0 = docs 0-3, 簇 1 = docs 4-7
    np.savez(cache_npz, final_labels=np.array([0, 0, 0, 0, 1, 1, 1, 1]))

    # ── 1) collect: 统计 / 域交叉 / 分母排除 ──
    info, profs = cp.collect(pool_dir, cache_npz, n_per=2, chars=40, seed=7)
    assert info["num_docs"] == 8 and info["K"] == 2, info
    p0, p1 = profs
    assert p0["n_docs"] == 4 and p1["n_docs"] == 4
    assert p0["med_chars"] == 1000.0 and p1["med_chars"] == 2200.0
    assert abs(p0["tok_pct"] + p1["tok_pct"] - 100.0) < 1e-9
    # 簇 0 域: 数学 3 + 物理 1 (id 序全 4 行, 含 0 行); est_tok = char/4
    assert p0["domains"] == [("数学", 3, 600.0), ("化学", 0, 0.0),
                             ("生物学", 0, 0.0), ("物理", 1, 400.0)], p0["domains"]
    assert p0["dom_docs_total"] == 4
    # 簇 1 域: 化学 2 + 生物学 1, doc6 (无域) 不进分母
    assert p1["dom_docs_total"] == 3
    dom1 = {n: dd for n, dd, _t in p1["domains"]}
    assert dom1 == {"数学": 0, "化学": 2, "生物学": 1, "物理": 0}, p1["domains"]
    # 质量分: 簇 1 = 基础+1
    assert abs(p1["quality"]["stem_relevance"] - 4.0) < 1e-9
    assert abs(p0["quality"]["knowledge_value"] - 4.0) < 1e-9
    # 抽样: n_per=2, snippet 截断到 chars
    assert len(p0["samples"]) == 2 and len(p1["samples"]) == 2
    assert all(len(s["snippet"]) <= 40 for s in p0["samples"])
    print("PASS  collect: stats / domain cross / denominators / samples")

    # ── 2) render_full: 关键结构与历史 CLI 输出同构 ──
    full = cp.render_full(
        types.SimpleNamespace(cluster_cache=cache_npz, n=2, seed=7, chars=40),
        info, profs)
    assert full.startswith("# 簇语义抽样 — cluster_cache.npz")
    assert "## C0" in full and "## C1" in full
    assert "(占池 35.4% token, 50.0% docs)" in full
    assert "域构成 (本簇内):" in full and "| 数学 | 3 | 75.0% |" in full
    assert "| 化学 | 0 | 0.0% |" in full          # 0 行保留 (id 序)
    assert "质量列均值: stem_relevance 4.00" in full
    assert "### 抽样 2 篇 (首 40 字符)" in full
    print("PASS  render_full: structure matches the historical CLI output")

    # ── 3) render_brief: 紧凑表 (报告 §2b 内嵌体) ──
    brief = cp.render_brief(profs)
    assert brief.count("\n") < 15                  # 紧凑: 表头2 + 2行 + 空 + caption + 2 样本
    assert "| C0 | 4 | 35.4% | 数学 75.0% / 物理 25.0% | 1000 |" in brief
    assert "| C1 | 4 | 64.6% | 化学 66.7% / 生物学 33.3% | 2200 |" in brief
    assert "stem_relevance 4.00, knowledge_value 5.00" in brief
    assert brief.count("- **C0**") == 1 and brief.count("- **C1**") == 1
    print("PASS  render_brief: compact table + one excerpt per cluster")

    # ── 4) recipe_report._cluster_semantics: 生成 → 缓存复用 → 跳过路径 ──
    run_dir = os.path.join(tmp, "run")
    os.makedirs(run_dir)
    np.savez(os.path.join(run_dir, "cluster_cache.npz"),
             final_labels=np.array([0, 0, 0, 0, 1, 1, 1, 1]))
    with open(os.path.join(run_dir, "launch_env.json"), "w") as f:
        json.dump({"DATA_DIR": pool_dir}, f)

    sem_path = os.path.join(run_dir, "cluster_semantics.md")
    block, note = rr._cluster_semantics(run_dir, pool_dir)
    assert block and note is None
    assert "| C0 | 4 | 35.4% |" in block and "- **C1**" in block
    assert os.path.isfile(sem_path)
    mtime = os.path.getmtime(sem_path)
    # 二次调用走缓存 (不重生成 — mtime 不变)
    block2, note2 = rr._cluster_semantics(run_dir, pool_dir)
    assert block2 == block and note2 is None
    assert os.path.getmtime(sem_path) == mtime
    # cluster_cache 变新 → 重生成
    time.sleep(0.05)
    os.utime(os.path.join(run_dir, "cluster_cache.npz"))
    block3, _ = rr._cluster_semantics(run_dir, pool_dir)
    assert block3 == block                        # 内容确定性 (同 seed)
    assert os.path.getmtime(sem_path) > mtime
    print("PASS  _cluster_semantics: generate / cache reuse / mtime refresh")

    # 缺件优雅跳过
    b, n = rr._cluster_semantics(run_dir, "/nonexistent/pool")
    assert b is None and "不可解析" in n
    b, n = rr._cluster_semantics(run_dir, "")
    assert b is None and "DATA_DIR" in n
    empty_dir = os.path.join(tmp, "empty_run")     # 独立空目录: 无 cluster_cache.npz
    os.makedirs(empty_dir)
    b, n = rr._cluster_semantics(empty_dir, pool_dir)
    assert b is None and "cluster_cache.npz 缺失" in n
    print("PASS  _cluster_semantics: graceful skip paths")

    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
