#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  基于已有 d20 实验结果，进行增量实验
#
#  本脚本做什么 (依次执行):
#    1. 复用资格核验: 数据池身份(内容级 key) + 聚类 K + 训练/评测参数
#       逐项与源 run 比对, 任一不符拒绝发射 (docs/reuse_design.md §2)
#    2. 继承源 run 的池缓存 (cluster_cache 双文件) + 注入其全部已测点
#    3. 调 run_search.sh 跑主实验:
#       新实验轮次的 d20 搜索 (guided, 从历史地基上继续)
#       → d28 两臂 (climb/random) → 报告
#
#  怎么复用: 源 run 的全部已测 d20 实验点【整体放进新实验的第一轮】
#  作地基 — 不重跑、不占新预算; 新实验从 guided 轮开始 (跳过随机
#  探索轮), exp id 从历史数续排, 采样自动避开历史配比.
#
#  新实验轮数 = CONFIGS_PER_ITER, 指【新增】的实验数, 不含历史:
#    例: "20,10" = 新增第 1 轮 20 个 + 第 2 轮 10 个 d20 实验;
#        历史 30 点地基 + 30 新点 → 总实验数 60, 其中要跑的 30.
#
#  用法: 编辑下方 EDIT 块 → ./runs/run_extend_search.sh
#        干跑: LAUNCH=0 ./runs/run_extend_search.sh (校验+注入+打印, 不发射)
#  · 注入只发生一次 (新 run 无 search_state 时); 中断后重跑本命令 = 续跑
#  · 其余发射参数 (NPU_PER_EXP/REMOTE_*/...) 在 run_search.sh 的 EDIT 块
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

# ─── EDIT HERE (env 可临时覆盖, 文件值为默认) ──────────────────────
EXP_NAME="${EXP_NAME:-prod3}"        # 新 run 名
HISTORY_RUN="${HISTORY_RUN:-}"       # 复用源 run; 留空 = 运行时列出可用的
CONFIGS_PER_ITER="${CONFIGS_PER_ITER:-20,10}"   # 新增实验轮次 (不含历史)
K_ENHANCED="${K_ENHANCED:-15}"       # 须与源池一致 (自动校验, 不一致拒绝发射)
LAUNCH="${LAUNCH:-1}"                # 0=干跑
# 传给 run_search.sh (其余发射参数在它的 EDIT 块):
export EXP_NAME K_ENHANCED
# ───────────────────────────────────────────────────────────────────

CLIMBMIX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CLIMBMIX_DIR"
OUTPUT_DIR="$CLIMBMIX_DIR/result/${EXP_NAME}_current"

# HISTORY_RUN 缺失 → 列出可用历史 run (点数 + 池 K + 池身份 + 可复用性)
if [ -z "$HISTORY_RUN" ]; then
    echo "✗ HISTORY_RUN 必填 (复用源 run)。可用历史 run (result/ 下):"
    for s in "$CLIMBMIX_DIR"/result/*/search_state.json; do
        [ -f "$s" ] || continue
        d="${s%/*}"; d="${d#"$CLIMBMIX_DIR"/}"
        info=$(python3 - "$s" "$d" "$K_ENHANCED" <<'PY'
import json, os, re, sys
state, rundir, want_k = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    s = json.load(open(state))
    n = len(s.get("accumulated_configs") or [])
    k = ""
    prof = os.path.join(rundir, "balanced_profile.json")
    if os.path.isfile(prof):
        try:
            k = json.load(open(prof)).get("K_final")
        except ValueError:
            pass
    if not k:
        k = s.get("n_clusters") or "?"
    # 池身份: search.log 记录的池 key 优先, 退回 launch_env 的 DATA_DIR
    pool_id = "?"
    log = os.path.join(rundir, "search.log")
    if os.path.isfile(log):
        m = re.search(r"Pool-level embedding/kmeans cache:.*embeddings/([0-9a-f]{12})",
                      open(log, errors="replace").read())
        if m:
            pool_id = f"key={m.group(1)}"
    if pool_id == "?":
        envf = os.path.join(rundir, "launch_env.json")
        if os.path.isfile(envf):
            dd = json.load(open(envf)).get("DATA_DIR")
            if dd:
                pool_id = f"dir={dd}"
    ok = "✓ 可复用" if str(k) == want_k else f"✗ K={k} != K_ENHANCED={want_k}, 不可复用"
    print(f"{n}|{k}|{pool_id}|{ok}")
except Exception as e:
    print(f"?|?|?|state 解析失败: {e}")
PY
) || info="?|?|?|state 解析失败"
        IFS='|' read -r N KK POOL VERDICT <<EOF
$info
EOF
        echo "  ${d}  (${N} 个 d20 实验, 池 K=${KK}, 池 ${POOL})  ${VERDICT}"
    done
    exit 1
fi
HISTORY_RUN="${HISTORY_RUN#"$CLIMBMIX_DIR"/}"   # repo 相对路径也可
echo "═══ 增量实验: 复用 ${HISTORY_RUN} → 新实验 ${EXP_NAME} ═══"

# 槽位规范化: CONFIGS_PER_ITER 只写新增轮次 → 历史数自动补进列表头
# (底层 bootstrapper 语义: 地基占 iter1; 兼容写法 = 第 1 槽恰好等于历史数)
normalize_slots() {
    N_HIST=$(python3 -c "import json;print(json.load(open('$OUTPUT_DIR/search_state.json'))['realized_configs_per_iter'][0])")
    if [ "${CONFIGS_PER_ITER%%,*}" != "$N_HIST" ]; then
        CONFIGS_PER_ITER="${N_HIST},${CONFIGS_PER_ITER}"
    fi
    NEW_PLAN="${CONFIGS_PER_ITER#*,}"
    N_NEW=$(python3 -c "print(sum(int(x) for x in '${NEW_PLAN}'.split(',')))")
    ROUNDS=$(python3 -c "
ps = [int(x) for x in '${NEW_PLAN}'.split(',')]
print(' + '.join(f'第 {i+1} 轮 {p} 个' for i, p in enumerate(ps)))")
    echo "    地基:   ${HISTORY_RUN} 的 ${N_HIST} 个历史 d20 实验 (已测完, 不占新预算)"
    echo "    新实验: ${ROUNDS} = ${N_NEW} 个新 d20 实验"
    echo "    总实验数 = $((N_HIST + N_NEW))  [底层 CONFIGS_PER_ITER=${CONFIGS_PER_ITER}]"
    export CONFIGS_PER_ITER
}

# 当前发射参数的解析: env 覆盖 > run_climbmix.sh EDIT 块默认值
# (默认值里的 $VAR 引用递归展开; 输出 "key=value" 行, 供下面两个核验函数共用)
_current_launch_params() {
    python3 - "$CLIMBMIX_DIR/runs/run_climbmix.sh" <<'PY'
import os, re, sys
pat = re.compile(r'^([A-Z][A-Z0-9_]+)="\$\{[A-Z0-9_]+:-(.*?)}"', re.M)
defaults = dict(pat.findall(open(sys.argv[1], encoding="utf-8").read()))

def expand(v, depth=0):
    if depth > 3 or "$" not in v:
        return v
    def sub(m):
        name = m.group(1) or m.group(2)
        if name in os.environ:
            return os.environ[name]
        if name in defaults:
            return expand(defaults[name], depth + 1)
        return m.group(0)
    return re.sub(r"\$\{([A-Z0-9_]+)\}|\$([A-Z0-9_]+)", sub, v)

for k in sorted(defaults):
    print(f"{k}={expand(os.environ.get(k, defaults[k]))}")
PY
}

# 池身份核验 (内容级): K 相同 ≠ 池相同 — 权重向量活在簇空间里,
# 池不同 = 同 K 也是不同空间。池 key = pool_embedding_cache_key
# (sha256 of 分片清单+大小, 嵌入模型, 采样数) — 与 pool-keyed 缓存同源。
# 源侧: search.log 里发射时的真实记录; 新侧: 现场扫 DATA_DIR 计算。
verify_pool_identity() {
    echo "  池身份核验 (内容级):"
    local src_key=""
    if [ -f "$HISTORY_RUN/search.log" ]; then
        src_key=$(grep -oE "Pool-level embedding/kmeans cache:.*embeddings/[0-9a-f]{12}" \
            "$HISTORY_RUN/search.log" | grep -oE "[0-9a-f]{12}$" | head -1)
    fi
    if [ -n "$src_key" ]; then
        local verdict
        verdict=$(_current_launch_params | python3 -c "
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from climbmix.utils.embed_cache import pool_embedding_cache_key
cur = dict(l.rstrip('\n').split('=', 1) for l in sys.stdin if '=' in l)
data_dir = cur.get('DATA_DIR', '')
if not data_dir or not os.path.isdir(data_dir):
    print('?|新侧 DATA_DIR 不存在: ' + (data_dir or '(未设置)'))
    sys.exit(0)
shards = sorted(f for f in os.listdir(data_dir) if f.endswith('.parquet'))
if not shards:
    print('?|新侧 DATA_DIR 无 parquet: ' + data_dir)
    sys.exit(0)
key = pool_embedding_cache_key(
    ((n, os.path.getsize(os.path.join(data_dir, n))) for n in shards),
    cur.get('EMBEDDING_MODEL', ''), 512,
    int(cur.get('EMBEDDING_SAMPLE_SIZE') or 0))
print(f'{key}|{len(shards)}')
" 2>&1) || verdict="?|解析失败"
        case "$verdict" in
            "?|"*)
                echo "    ⚠ 池核验跳过 (${verdict#?|})"
                echo "      请人工确认 DATA_DIR 与源 run 的数据池一致" ;;
            *)
                local new_key n_shards
                new_key="${verdict%%|*}"; n_shards="${verdict#*|}"
                if [ "$new_key" = "$src_key" ]; then
                    echo "    ✓ 池一致 (key=${src_key}, ${n_shards} 个分片)"
                else
                    echo "    ✗ 池不一致: 源 key=${src_key} vs 当前 key=${new_key}"
                    echo "      不同数据池 = 簇空间不同, 历史点权重无意义 (docs/reuse_design.md §2)"
                    exit 1
                fi ;;
        esac
    else
        # 回退: 源 run 无 search.log 记录 (老 run) → 只能路径级比对
        local src_dir
        src_dir=$(python3 -c "
import json
try:
    print(json.load(open('$HISTORY_RUN/launch_env.json')).get('DATA_DIR', ''))
except (OSError, ValueError):
    print('')")
        if [ -z "$src_dir" ]; then
            echo "    ⚠ 源 run 无 search.log 也无 launch_env.json — 池身份无法核验, 请人工确认"
        else
            local cur_dir
            cur_dir=$(_current_launch_params | grep -m1 '^DATA_DIR=' | cut -d= -f2-)
            if [ "$src_dir" = "$cur_dir" ]; then
                echo "    ⚠ 仅路径级核验通过 (DATA_DIR=${src_dir}; 源 run 无 search.log, 无内容级 key)"
            else
                echo "    ✗ 池不一致: 源 DATA_DIR=${src_dir} vs 当前 DATA_DIR=${cur_dir}"
                echo "      且源 run 无 search.log 无法做内容级核验 — 拒绝 (docs/reuse_design.md §2)"
                exit 1
            fi
        fi
    fi
}

# 训练/评测语义参数核验: 源 run 发射时的参数 (launch_env.json) vs
# 当前发射参数。路径/执行形状键排除; DATA_DIR/GENERAL_DATA_DIR 单列
# (路径不同仅警告 — 内容终审由上面的池 key 把关)。
verify_immutable_layer() {
    echo "  训练/评测参数核验 (源 run 发射参数 vs 当前):"
    local params_tmp
    params_tmp="$(mktemp)"
    _current_launch_params > "$params_tmp"
    python3 - "$HISTORY_RUN/launch_env.json" "$params_tmp" <<'PY'
import json, os, sys
src_path, params_path = sys.argv[1], sys.argv[2]
if not os.path.isfile(src_path):
    print("    ⚠ 源 run 无 launch_env.json — 无法自动核对, 请人工比对源 run 的 search.log")
    sys.exit(0)
src = json.load(open(src_path))
cur = dict(l.rstrip('\n').split('=', 1) for l in open(params_path) if '=' in l)
# 硬核验: 语义键 (训练/评测协议) — 不等即拒
SKIP = {"EXP_NAME", "OUTPUT_DIR", "NUM_NPU", "NPU_PER_EXP",
        "CLIMBMIX_DIR", "NANOCHAT_DIR", "NANOCHAT_BASE_DIR",
        "TARGET_ARM_NODES", "HF_ENDPOINT", "REMOTE_D28_ASSET_URI",
        "DATA_DIR", "GENERAL_DATA_DIR"}
# 警告级: 数据路径 — 路径不同内容可能相同, 终审在池 key
WARN = {"DATA_DIR", "GENERAL_DATA_DIR"}
mismatch, warns, checked = [], [], 0
for k, src_v in sorted(src.items()):
    if k in SKIP and k not in WARN:
        continue
    cur_v = cur.get(k, "")
    if k in WARN:
        if str(src_v) != str(cur_v):
            warns.append((k, src_v, cur_v))
        continue
    checked += 1
    if str(src_v) != str(cur_v):
        mismatch.append((k, src_v, cur_v))
for k, src_v, cur_v in mismatch:
    print(f"    ✗ {k}: 源={src_v}  当前={cur_v or '(默认)'}")
for k, src_v, cur_v in warns:
    print(f"    ⚠ {k}: 源={src_v}  当前={cur_v or '(默认)'} (路径不同, 由池 key 终审)")
if mismatch:
    print(f"    ✗ {len(mismatch)}/{checked} 个语义参数不一致 — 历史点在新配置下的测量")
    print("      可比性存疑 (docs/reuse_design.md §2 不可变层)。修正后再来, 或确认有意为之")
    sys.exit(1)
print(f"    ✓ {checked} 个语义参数全部一致" + (f" (+{len(warns)} 个路径警告)" if warns else ""))
PY
    rm -f "$params_tmp"
}

if [ -f "$OUTPUT_DIR/search_state.json" ]; then
    # 续跑: 注入早已完成, 只补槽 (同一命令同一写法, 重跑即续)
    echo "  → 检测到已有 search_state — 续跑 (不重注入)"
    if python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUTPUT_DIR/search_state.json')).get('history_seed') else 1)" 2>/dev/null; then
        normalize_slots
    fi
    verify_pool_identity
    verify_immutable_layer
else
    # 首次增量: 校验 → 继承池缓存 → 注入 (docs/reuse_design.md §4.2/§8)
    echo "  → 复用注入: ${HISTORY_RUN} 的历史点"
    [ -f "$HISTORY_RUN/search_state.json" ] || { echo "✗ ${HISTORY_RUN}/search_state.json 不存在 (用上面列出的可用 run)"; exit 1; }
    [ -f "$HISTORY_RUN/cluster_cache.npz" ] || { echo "✗ ${HISTORY_RUN}/cluster_cache.npz 不存在 (池缓存来源)"; exit 1; }
    if [ -f "$HISTORY_RUN/balanced_profile.json" ]; then
        SRC_K=$(python3 -c "import json;print(json.load(open('${HISTORY_RUN}/balanced_profile.json'))['K_final'])")
        [ "$SRC_K" = "$K_ENHANCED" ] || { echo "✗ 源池 K_final=${SRC_K} != K_ENHANCED=${K_ENHANCED} — 聚类空间不同, 历史点不可复用 (docs/reuse_design.md §2)"; exit 1; }
        echo "    K 校验: 源池 K_final=${SRC_K} == K_ENHANCED ✓"
    fi
    verify_pool_identity
    mkdir -p "$OUTPUT_DIR"
    if [ ! -f "$OUTPUT_DIR/cluster_cache.npz" ]; then
        # 池缓存 = 两个文件 (climb_pipeline 缓存命中条件: npz + info json 同在):
        # 缺任一个都会触发重新聚类 → 历史点静默作废
        cp "$HISTORY_RUN/cluster_cache.npz" "$OUTPUT_DIR/"
        if [ -f "$HISTORY_RUN/cluster_info_cache.json" ]; then
            cp "$HISTORY_RUN/cluster_info_cache.json" "$OUTPUT_DIR/"
        else
            echo "    ⚠ 源 run 缺 cluster_info_cache.json — 缓存不完整, Step 1 可能重新聚类!"
            echo "      (重新聚类 = 历史点作废; 源归档不完整请先补齐)"
        fi
        if [ -f "$HISTORY_RUN/balanced_profile.json" ]; then
            cp "$HISTORY_RUN/balanced_profile.json" "$OUTPUT_DIR/"
        fi
        echo "    池缓存已继承 (继承, 不重新聚类 — 重新聚类 = 历史点静默作废)"
    fi
    python3 scripts/inject_history.py \
        --source "$HISTORY_RUN/search_state.json" \
        --target-dir "$OUTPUT_DIR" \
        --pool "$OUTPUT_DIR/cluster_cache.npz"
    normalize_slots
    verify_immutable_layer
fi

if [ "$LAUNCH" != "1" ]; then
    echo
    echo "[dry-run] 注入/校验完成。正式发射: 去掉 LAUNCH=0 重跑本命令"
    exit 0
fi

# 注入完成后进入主流程 (state 已在 → run_search.sh 走续跑路径, 从新实验第 1 轮开始)
exec bash runs/run_search.sh
