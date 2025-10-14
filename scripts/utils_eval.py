# utils_eval.py
"""
Evaluation utilities for reaction mechanism step alignment (oMeS Benchmark).
"""

from typing import List, Tuple, Callable
from dataclasses import dataclass
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

TAU: float = 0.60  # you can modify this value
FP_RADIUS: int = 2
FP_NBITS: int = 2048
NON_MATCH_PENALTY = 1e-6 
_morgan_gen = GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_NBITS)

def canonical_smiles(smi: str) -> str:
    """
    Convert SMILES to canonical form using RDKit.
    Return empty string if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smi)
    return "" if mol is None else Chem.MolToSmiles(mol, canonical=True)


def tanimoto_morgan(smi1: str, smi2: str) -> float:
    """
    Compute the Tanimoto similarity between two SMILES using Morgan fingerprints.
    """
    mol1, mol2 = Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)
    if mol1 is None or mol2 is None:
        return 0.0
    fp1 = _morgan_gen.GetFingerprint(mol1)
    fp2 = _morgan_gen.GetFingerprint(mol2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def step_sigma(raw_sim: float, tau: float = TAU) -> float:
    return 0.0 if raw_sim < tau else raw_sim


GoldStep = Tuple[str, str, float]
PredStep = Tuple[str, str]

@dataclass
class oMeSResult:
    S_total: float
    S_partial: float
    alignment: List[str]
    V: float  # Valid SMILES rate
    L: float  # Logical fidelity score

# ------- 只需要替换 oMeS 函数本体 -------

def oMeS(
    gold: List[GoldStep],
    pred: List[PredStep],
    sim_fn: Callable[[str, str], float] = tanimoto_morgan,
    tau: float = TAU,
) -> oMeSResult:
    N, M = len(gold), len(pred)

    DP_tot = [[0.0]*(M+1) for _ in range(N+1)]
    DP_par = [[0.0]*(M+1) for _ in range(N+1)]
    DP_pen = [[0.0]*(M+1) for _ in range(N+1)]
    DP_rnk = [[0]*(M+1)   for _ in range(N+1)]   # 显式优先级
    trace  = [[None]*(M+1) for _ in range(N+1)]

    rank = {
        "match": 3,
        "type_mismatch": 2,
        "skip_gold": 1,
        "skip_pred": 1,
    }

    trace[0][0] = "end"

    # ---------- 初始化第一列（j = 0）：全是 skip_gold ----------
    for i in range(1, N+1):
        DP_tot[i][0] = DP_tot[i-1][0]
        DP_par[i][0] = DP_par[i-1][0]
        DP_pen[i][0] = DP_pen[i-1][0] - NON_MATCH_PENALTY
        DP_rnk[i][0] = rank["skip_gold"]
        trace[i][0]  = "skip_gold"

    # ---------- 初始化第一行（i = 0）：全是 skip_pred ----------
    for j in range(1, M+1):
        DP_tot[0][j] = DP_tot[0][j-1]
        DP_par[0][j] = DP_par[0][j-1]
        DP_pen[0][j] = DP_pen[0][j-1] - NON_MATCH_PENALTY
        DP_rnk[0][j] = rank["skip_pred"]
        trace[0][j]  = "skip_pred"

    gold_can = [canonical_smiles(s) for _, s, _ in gold]
    pred_can = [canonical_smiles(s) for _, s in pred]

    for i in range(1, N+1):
        g_type, g_smi, w = gold[i-1]
        g_can = gold_can[i-1]

        for j in range(1, M+1):
            p_type, p_smi = pred[j-1]
            p_can = pred_can[j-1]

            if g_type == p_type:
                m_tot   = w if (g_can == p_can and g_can) else 0.0
                m_par   = w * step_sigma(sim_fn(g_smi, p_smi), tau) if (g_can and p_can) else 0.0
                action_diag = "match"
                pen_diag    = 0.0
                r_diag      = rank["match"]
            else:
                m_tot = m_par = 0.0
                action_diag = "type_mismatch"
                pen_diag    = -NON_MATCH_PENALTY
                r_diag      = rank["type_mismatch"]

            candidates = [
                # 上方 -> skip_gold
                (DP_tot[i-1][j],   DP_par[i-1][j],   DP_pen[i-1][j] - NON_MATCH_PENALTY, rank["skip_gold"], "skip_gold",   (i-1, j)),
                # 左方 -> skip_pred
                (DP_tot[i][j-1],   DP_par[i][j-1],   DP_pen[i][j-1] - NON_MATCH_PENALTY, rank["skip_pred"], "skip_pred",   (i, j-1)),
                # 对角线
                (DP_tot[i-1][j-1] + m_tot,
                 DP_par[i-1][j-1] + m_par,
                 DP_pen[i-1][j-1] + pen_diag,
                 r_diag, action_diag, (i-1, j-1))
            ]

            # 关键：把 rank 放到排序 key 里，确保优先级（match > mismatch > skip）
            best = max(candidates, key=lambda x: (x[0], x[1], x[3], x[2]))
            DP_tot[i][j], DP_par[i][j], DP_pen[i][j], DP_rnk[i][j], trace[i][j], _ = best

    # ---------- 回溯 ----------
    align = []
    i, j = N, M
    while not (i == 0 and j == 0):
        # 防御：如果仍出现 None（说明边界没处理好），用显式分支兜底
        if i == 0:
            align.append("skip_pred")
            j -= 1
            continue
        if j == 0:
            align.append("skip_gold")
            i -= 1
            continue

        act = trace[i][j]
        if act in ("match", "type_mismatch"):
            align.append(act)
            i, j = i - 1, j - 1
        elif act == "skip_gold":
            align.append(act)
            i -= 1
        elif act == "skip_pred":
            align.append(act)
            j -= 1
        elif act == "end":
            break
        else:
            # 不应该到这里，如果到了，说明还有逻辑问题
            raise RuntimeError(f"Unexpected trace action: {act} at ({i}, {j})")
    align.reverse()
    # --------- 计算 V / L ----------
    V = sum(1 for _, s in pred if canonical_smiles(s)) / max(1, M)
    L = sum(1 for act in align if act == "match") / max(1, N)
    return oMeSResult(round(DP_tot[N][M], 2), round(DP_par[N][M], 2), align, round(V, 2), round(L, 2))


def alignment_table(
    gold: List[GoldStep],
    pred: List[PredStep],
    align: List[str],
) -> pd.DataFrame:
    """display the alignment result as a DataFrame"""
    rows = []
    gi = pi = 0
    for tag in align:
        g = gold[gi] if gi < len(gold) else ("-", "-", 0.0)
        p = pred[pi] if pi < len(pred) else ("-", "-")
        rows.append({
            "Gold type": g[0], "Gold SMILES": g[1],
            "Pred type": p[0], "Pred SMILES": p[1],
            "Action": tag,
        })
        if tag in ("match", "type_mismatch"):
            gi += 1; pi += 1
        elif tag == "skip_gold":
            gi += 1
        else:  # skip_pred
            pi += 1
    return pd.DataFrame(rows)