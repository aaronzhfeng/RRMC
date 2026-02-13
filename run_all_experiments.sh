#!/bin/bash
# =============================================================================
# RRMC Experiment Runner — 100-Puzzle Scaling
# =============================================================================
#
# Runs all experiments from docs/12_experiment_plan_100puzzles.md
#
# Usage:
#   ./run_all_experiments.sh              # Run all phases
#   ./run_all_experiments.sh phase1       # Run only Phase 1 (DC)
#   ./run_all_experiments.sh phase2       # Run only Phase 2 (SP)
#   ./run_all_experiments.sh phase3       # Run only Phase 3 (GN)
#   ./run_all_experiments.sh smoke        # 5-puzzle smoke test of all methods
#
# Requirements:
#   - OPENROUTER_API_KEY set in environment
#   - Sufficient OpenRouter credits ($20+ recommended)
#
# Rate limits: OpenRouter has no RPM limit for paid models (only DDoS
# protection). We use max_workers=32 for aggressive parallelism.
# =============================================================================

set -euo pipefail

cd /root/RRMC

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
WORKERS=32              # Concurrent API calls per experiment
N_PUZZLES=100           # Puzzles per experiment
GN_PUZZLES=20           # GN starts with 20 (gate before scaling)
LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)"
MAX_PARALLEL_EXPS=3     # Max experiments running in background simultaneously

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
mkdir -p "$LOG_DIR"

log() {
    echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"
}

success() {
    echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $*"
}

warn() {
    echo -e "${YELLOW}[$(date +%H:%M:%S)] !${NC} $*"
}

fail() {
    echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} $*"
}

# Run a single experiment
# Usage: run_exp <config_name> <n_puzzles> <extra_args...>
run_exp() {
    local config="$1"
    local n="$2"
    shift 2
    local logfile="${LOG_DIR}/${config//\//_}.log"

    log "Starting: $config (n=$n, workers=$WORKERS)"
    if python run.py "$config" \
        --n_puzzles "$n" \
        --max_workers "$WORKERS" \
        "$@" \
        > "$logfile" 2>&1; then
        success "Done: $config → $logfile"
        return 0
    else
        fail "FAILED: $config → $logfile"
        return 1
    fi
}

# Run experiment in background, tracking PID
# Usage: run_bg <config_name> <n_puzzles> <extra_args...>
PIDS=()
EXP_NAMES=()

run_bg() {
    local config="$1"
    local n="$2"
    shift 2
    local logfile="${LOG_DIR}/${config//\//_}.log"

    log "Launching background: $config (n=$n)"
    python run.py "$config" \
        --n_puzzles "$n" \
        --max_workers "$WORKERS" \
        "$@" \
        > "$logfile" 2>&1 &
    PIDS+=($!)
    EXP_NAMES+=("$config")
}

# Wait for all background experiments and report
wait_all() {
    local any_failed=0
    for i in "${!PIDS[@]}"; do
        local pid="${PIDS[$i]}"
        local name="${EXP_NAMES[$i]}"
        if wait "$pid"; then
            success "Done: $name (pid $pid)"
        else
            fail "FAILED: $name (pid $pid)"
            any_failed=1
        fi
    done
    PIDS=()
    EXP_NAMES=()
    return $any_failed
}

# Throttle: wait until running background jobs drop below limit
throttle() {
    while (( $(jobs -rp | wc -l) >= MAX_PARALLEL_EXPS )); do
        sleep 5
    done
}

# Check prerequisites
check_prereqs() {
    # API key is loaded from configs/providers/openrouter.yaml automatically
    # Quick import check
    if ! python -c "from pipeline import Pipeline" 2>/dev/null; then
        fail "Python import failed. Check dependencies."
        exit 1
    fi
    success "Prerequisites OK"
}

# Print summary of results directory
print_summary() {
    local task="$1"
    log "Results for $task:"
    local dir="results/baseline/${task}"
    if [[ -d "$dir" ]]; then
        find "$dir" -name "*.json" -newer "$LOG_DIR" -exec basename {} \; 2>/dev/null | head -20
    fi
}

# =============================================================================
# Phase 1: DC Validation (100 puzzles, qwen-2.5-7b)
# =============================================================================
phase1() {
    echo ""
    echo "============================================================"
    echo " Phase 1: DC — 100 puzzles, qwen/qwen-2.5-7b-instruct"
    echo "============================================================"
    echo ""

    # --- HIGH priority: run in parallel ---

    # 1a. Zero-shot (very fast — no NPC calls)
    run_bg "dc_methods/zero_shot" "$N_PUZZLES"

    # 1b. Fixed Turns 10 (moderate — 10 turns × NPC)
    run_bg "dc_methods/fixed_turns" "$N_PUZZLES"

    # 1c. CIP-Lite (fast — stops at T1 mostly)
    run_bg "dc_methods/cip_lite" "$N_PUZZLES"

    wait_all
    log "Phase 1 batch 1/3 complete (zero_shot, fixed_turns, cip_lite)"

    # --- HIGH priority, expensive: DQS runs ---

    # 1d. DQS + Fixed(10) — the proven winner
    run_bg "dc_methods/dqs_fixed_turns" "$N_PUZZLES"

    # 1e. DQS + MI(min_turns=3) — key new experiment
    run_bg "dc_methods/dqs_mi_only" "$N_PUZZLES"

    # 1f. MI-Only(min_turns=3) — ablation (no DQS)
    run_bg "dc_methods/mi_only_min3" "$N_PUZZLES"

    wait_all
    log "Phase 1 batch 2/3 complete (dqs_fixed, dqs_mi, mi_only_min3)"

    # --- LOW priority: comparison baselines ---

    run_bg "dc_methods/knowno" "$N_PUZZLES"
    run_bg "dc_methods/self_consistency" "$N_PUZZLES"

    wait_all
    log "Phase 1 batch 3/3 complete (knowno, self_consistency)"

    success "Phase 1 COMPLETE"
    echo ""
}

# =============================================================================
# Phase 2: SP Evaluation (100 puzzles, qwen-2.5-7b)
# =============================================================================
phase2() {
    echo ""
    echo "============================================================"
    echo " Phase 2: SP — 100 puzzles, qwen/qwen-2.5-7b-instruct"
    echo "============================================================"
    echo ""

    # SP has no DQS support yet, so just baselines + MI

    # Batch 1: fast runs
    run_bg "sp_methods/zero_shot" "$N_PUZZLES"
    run_bg "sp_methods/fixed_turns" "$N_PUZZLES"
    run_bg "sp_methods/cip_lite" "$N_PUZZLES"

    wait_all
    log "Phase 2 batch 1/2 complete (zero_shot, fixed_turns, cip_lite)"

    # Batch 2: MI (heavier)
    run_bg "sp_methods/mi_only_min3" "$N_PUZZLES"

    wait_all
    success "Phase 2 COMPLETE"
    echo ""
}

# =============================================================================
# Phase 3: GN with stronger model (20 puzzles gate)
# =============================================================================
phase3() {
    echo ""
    echo "============================================================"
    echo " Phase 3: GN — ${GN_PUZZLES} puzzles"
    echo "============================================================"
    echo ""
    warn "GN uses the default model (qwen-2.5-7b)."
    warn "To test with qwen3-coder-30b, edit policy_model in the config"
    warn "or override via base.yaml before running."
    echo ""

    # GN baseline + DQS (entropy-optimal) + MI
    run_bg "gn_methods/fixed_turns" "$GN_PUZZLES"
    run_bg "gn_methods/dqs_fixed_turns" "$GN_PUZZLES"

    wait_all
    log "Phase 3 batch 1/2 complete (fixed_turns, dqs)"

    run_bg "gn_methods/mi_only_min5" "$GN_PUZZLES"

    wait_all
    success "Phase 3 COMPLETE"
    echo ""
}

# =============================================================================
# Smoke test: 5 puzzles, all methods, quick validation
# =============================================================================
smoke() {
    local SMOKE_N=5
    echo ""
    echo "============================================================"
    echo " Smoke Test — ${SMOKE_N} puzzles per method"
    echo "============================================================"
    echo ""

    # DC smoke (3 key methods in parallel)
    run_bg "dc_methods/zero_shot" "$SMOKE_N"
    run_bg "dc_methods/fixed_turns" "$SMOKE_N"
    run_bg "dc_methods/dqs_mi_only" "$SMOKE_N"

    wait_all
    log "DC smoke complete"

    # SP smoke
    run_bg "sp_methods/zero_shot" "$SMOKE_N"
    run_bg "sp_methods/fixed_turns" "$SMOKE_N"
    run_bg "sp_methods/mi_only_min3" "$SMOKE_N"

    wait_all
    log "SP smoke complete"

    # GN smoke
    run_bg "gn_methods/fixed_turns" "$SMOKE_N"

    wait_all
    success "Smoke test COMPLETE"
    echo ""
}

# =============================================================================
# Main
# =============================================================================
main() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║         RRMC Experiment Runner — 100-Puzzle Scale           ║"
    echo "╠══════════════════════════════════════════════════════════════╣"
    echo "║  Workers: $WORKERS concurrent API calls                         ║"
    echo "║  Puzzles: DC=$N_PUZZLES  SP=$N_PUZZLES  GN=$GN_PUZZLES                          ║"
    echo "║  Logs:    $LOG_DIR            ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""

    check_prereqs

    local phase="${1:-all}"

    case "$phase" in
        phase1|dc)
            phase1
            ;;
        phase2|sp)
            phase2
            ;;
        phase3|gn)
            phase3
            ;;
        smoke|test)
            smoke
            ;;
        all)
            local start=$(date +%s)
            phase1
            phase2
            phase3
            local end=$(date +%s)
            local elapsed=$(( end - start ))
            echo ""
            echo "============================================================"
            success "ALL PHASES COMPLETE in $(( elapsed / 60 ))m $(( elapsed % 60 ))s"
            echo "  Logs: $LOG_DIR"
            echo "  Results: results/baseline/{dc,sp,gn}/"
            echo "============================================================"
            ;;
        *)
            echo "Usage: $0 [phase1|phase2|phase3|smoke|all]"
            exit 1
            ;;
    esac
}

main "$@"
