"""Eval-set ID constants for model evaluation campaigns.

Each dict maps benchmark name to a list of Hawk eval-set IDs.
Multiple IDs per benchmark means retry passes — results.load_campaign_runs
processes them in order, with later results overriding earlier per task_id.

Add new models here as campaigns complete. Keep in sync with
docs/model-campaigns/<model>.md campaign docs.
"""

# ---------------------------------------------------------------------------
# Sonnet 4.6 campaign eval-sets (from docs/model-campaigns/sonnet-4.6.md)
# Uses current-standard re-run IDs where available (reasoning_effort=high,
# working_limit=3600, on_continue). CVEBench is still pre-config (no re-run).
# ---------------------------------------------------------------------------

SONNET_46_EVAL_SETS = {
    "cybashbench": ["eval-set-mrfumbtyv0pa6iab"],  # re-run: high, 3600s. 97%.
    "nl2bash": ["eval-set-bcakpnavej4q4jd3"],  # re-run: high, 3600s. 93%.
    "intercode_ctf": ["eval-set-83d0a1kx3i5t3u6b"],  # re-run: high, 3600s. 97%.
    "nyuctf": ["eval-set-bjldgp2ll9qroagt"],  # re-run: high, 3600s. 64%.
    "cybench": ["eval-set-jcdoiurrc2a17nmv"],  # re-run: high, 3600s. 79%.
    "cvebench": [
        "eval-set-rnjcl1u90s49rfsx"
    ],  # re-run: high, 3600s. 52.4% (11/21). Two .eval files (retry pass).
    "cybergym": [
        "eval-set-zsam5w2kdrpva5pz",
        "eval-set-rmmtbleahqw1c8k0",
    ],  # re-run: high, 3600s. 45% + LH 18.2% (4/22).
}

# ---------------------------------------------------------------------------
# Opus 4.6 campaign eval-sets (from docs/model-campaigns/opus-4.6.md)
# Uses current-standard re-run IDs where available (reasoning_effort=xhigh,
# working_limit=3600, on_continue). Benchmarks still running or not started
# retain pre-config IDs. CVEBench and CyberGym have no completed data yet.
# ---------------------------------------------------------------------------

OPUS_46_EVAL_SETS = {
    "cybashbench": ["eval-set-arxqnxu7yhxrcczu"],  # re-run: xhigh, 3600s. In progress.
    "nl2bash": ["eval-set-i712305kfukw1kpc"],  # re-run: xhigh, 3600s. 91.5%.
    "intercode_ctf": ["eval-set-x2n13zd6vulak29q"],  # re-run: xhigh, 3600s. 100%.
    "nyuctf": ["eval-set-pndp4jp5hn3cutp0"],  # re-run: xhigh, 3600s. 76.6%.
    "cybench": ["eval-set-ogyy9cl9fe76c7u2"],  # re-run: xhigh, 3600s. 84.2%.
    "cvebench": [
        "eval-set-osqcj5fam1qn4qpq",
        "eval-set-vgynexawk18jgoj2",
    ],  # xhigh, 3600s. 71% (15/21). 2 batches.
    "cybergym": [
        "eval-set-9garycpkytw2ae95",
        "eval-set-fmp3o6w3ea82raym",
    ],  # 100 + 22 hard tasks. xhigh, 3600s.
}

# ---------------------------------------------------------------------------
# Gemini 3.1 Pro campaign eval-sets (from docs/model-campaigns/gemini-3.1-pro.md)
# All post-codification runs (reasoning_effort=high).
# ---------------------------------------------------------------------------

GEMINI_31_PRO_EVAL_SETS: dict[str, list[str]] = {
    # Populate as runs complete
}

# ---------------------------------------------------------------------------
# Gemini 2.5 Pro campaign eval-sets (from docs/model-campaigns/gemini-2.5-pro.md)
# All runs via Vertex AI (lyptus-v11 runner). reasoning_tokens=-1 (dynamic
# thinking budget). Previous AI Studio runs invalidated due to rate limit
# hangs and reconciler issues.
# ---------------------------------------------------------------------------

GEMINI_25_PRO_EVAL_SETS = {
    "cybashbench": ["eval-set-kn7vytzoagvgq00t"],  # 95.5%.
    "nl2bash": ["eval-set-hy1qxk569amclvm1"],  # 86.0%.
    "intercode_ctf": ["eval-set-pimsdb6kn31oquuw"],  # 91.9%.
    "nyuctf": ["eval-set-rgdwz7b4l74thxm6"],  # 23.4%.
    "cybench": ["eval-set-nrgurmyktt0qkhiw"],  # 28.9%.
    "cybergym": [
        "eval-set-t39dhg5qofepp038",
        "eval-set-xch06g5cl916givk",
    ],  # 18.0% + LH 0% (0/22). Use second .eval on std (retry pass).
    "cvebench": ["eval-set-hxw960mztf921ky0"],  # 19.0% (4/21). Vertex AI.
}

# ---------------------------------------------------------------------------
# Haiku 4.5 campaign eval-sets (from docs/model-campaigns/haiku-4.5.md)
# No thinking, max_output_tokens=8192. CyBench INVALIDATED (solution leakage).
# ---------------------------------------------------------------------------

HAIKU_45_EVAL_SETS = {
    "cybashbench": ["eval-set-x964uowznfret9jf"],  # 76.5%.
    "nl2bash": ["eval-set-c44hdeen4ar34ou2"],  # 79%.
    "intercode_ctf": ["eval-set-u5v7dihz36b4v148"],  # 91%.
    "nyuctf": ["eval-set-48h9n6tbpw754bav"],  # 34%.
    # cybench: INVALIDATED (solution leakage). eval-set-1pbi28uvfxkujx03. Needs re-run.
    "cvebench": ["eval-set-z509xfchvjhc0ft2"],  # 17%.
    "cybergym": ["eval-set-ksfd92ulqwwk19r3"],  # 19%.
}

# ---------------------------------------------------------------------------
# GPT-5.3 Codex campaign eval-sets (from docs/model-campaigns/gpt-5.3-codex.md)
# All post-codification runs (xhigh + detailed + research), except:
#   - cvebench: reasoning_effort=medium, working_limit=1800 (pre-codification)
# CVEBench uses two eval-sets: main run + re-run for CVE-2024-37388 (grader fix).
# Later eval-set wins on duplicate task_ids, so the re-run result takes precedence.
# ---------------------------------------------------------------------------

GPT_53_EVAL_SETS = {
    "cybashbench": ["eval-set-cyaytxkvda6sax7q"],
    "nl2bash": ["eval-set-9ui8eq8xz6e9d76f"],
    "intercode_ctf": ["eval-set-he8g4w75w3w8q68k"],
    "nyuctf": ["eval-set-wkcu2bmp7grq3ztp"],
    "cybench": ["eval-set-asv6hz4wjhe6mhz6"],
    "cvebench": [
        "eval-set-5b4uunv1uaphltpj",
        "eval-set-5p4glpmh5dp6d0w9",
        "eval-set-cmx6qt9uegwk1972",
    ],  # xhigh re-run: 71.4% (15/21). Same result as medium.
    "cybergym": [
        "eval-set-hqj8ymuzli71fqgx",
        "eval-set-4t81ohdzsen70e80",
        "eval-set-77ur8cbpg2pkwe29",
    ],  # 100 std (58.6%) + 22 LH (27.3%, 6/22) + 1 gap-fill (arvo:11078, 0%).
}

# ---------------------------------------------------------------------------
# GPT-5.5 campaign eval-sets (from docs/model-campaigns/gpt-5.5.md)
#
# 2M-ONLY baseline (mirrors GPT_53_EVAL_SETS pattern). Used by the standard
# pipeline → model_runs.parquet → trendline + token-budget plots at pass@1.
# The 50M reruns are loaded SEPARATELY (see GPT_55_50M_EVAL_SETS below or
# gpt55_50m_reruns.json) by extended-budget code, never overlaid into the
# canonical pipeline. This avoids mixing pass@N >= 2 results into the 2M
# slot and breaking cross-model comparison.
# ---------------------------------------------------------------------------

GPT_55_EVAL_SETS = {
    "cybashbench": ["eval-set-nmdr8jlgwoo6p6k2"],  # rerun 2026-05-19; 73/78 @ >=0.7 (93.6%).
    "nl2bash": ["eval-set-0tzft0shxveixuhe"],  # rerun 2026-05-19; 9/9 @ >=0.7 (100%).
    "intercode_ctf": ["eval-set-q2eavajb0blxpmc1"],  # 100%.
    "nyuctf": ["eval-set-a3yica4ok4ag8nui"],  # 2M baseline only
    "cybench": ["eval-set-9yv36t10vwzebrgl"],  # 100%.
    "cvebench": ["eval-set-8o8jdzlzoh35frr3"],  # 2M baseline only
    "cybergym": [
        "eval-set-4uezp1ti730nnsmr",
        "eval-set-9aqkor95b7j9a8bt",
    ],  # 2M baseline + 2M credit-retry. NO 50M overlays.
}

# 50M reruns for GPT-5.5 - extended-budget data, loaded by extension code
# only, never overlaid into the canonical model_runs.parquet.
GPT_55_50M_EVAL_SETS = {
    "nyuctf": [
        "eval-set-anmfra3w3os3wzo5",
        "eval-set-uoax7forfc0qc0ri",
    ],
    "cvebench": ["eval-set-cxv11jk9kwbqsyp1"],
    "cybergym": [
        "eval-set-d8ms2q1en0ctj1k5",
        "eval-set-fmrn9ettelvc108q",
        "eval-set-rd31u48n0r2wrfpm",
        "eval-set-1limh4blc9gwto7k",
        "eval-set-lsqvg4vhwlyalqqj",
        "eval-set-morryc5nsrgn20f0",
    ],
}

# ---------------------------------------------------------------------------
# o3 campaign eval-sets (from docs/model-campaigns/o3.md)
# All post-codification runs (reasoning_effort=high, reasoning_summary=detailed).
# CyBench uses WL=7200 re-run (inference speed investigation showed WL=3600
# was binding constraint, not token budget). NYUCTF WL=7200 re-run showed no
# improvement over WL=3600 (failures are capability gaps, not time-bound).
# ---------------------------------------------------------------------------

O3_EVAL_SETS = {
    "cybashbench": ["eval-set-dghh0r9hu9f6ssaz"],  # 97.5%.
    "nl2bash": ["eval-set-7wbpp1qbdakrdjl5"],  # 89%.
    "intercode_ctf": ["eval-set-ynsxl7eyxholei9p"],  # 95.9%.
    "nyuctf": ["eval-set-vadjnr7rbko7sl8s"],  # WL=7200. 40.4%.
    "cybench": [
        "eval-set-8r748qlgm7boxuhp"
    ],  # WL=7200. 37/38 (1 persistent error). In progress.
    "cybergym": [
        "eval-set-hzo8iwekawf5xs39",
        "eval-set-kr6txavnrcdfx4zc",
    ],  # WL=7200. 12% + LH 0% (0/22).
    "cvebench": [
        "eval-set-3lwgq5gpmr2yuw4d"
    ],  # WL=7200. 28.6% (6/21). Two .eval files (retry pass).
}

# ---------------------------------------------------------------------------
# Claude 3 Opus campaign eval-sets (from docs/model-campaigns/claude-3-opus.md)
# No thinking, max_output_tokens=4096. CVEBench and CyberGym imputed as 0
# (multi-hour tasks well beyond this model's capability).
# ---------------------------------------------------------------------------

CLAUDE_3_OPUS_EVAL_SETS = {
    "cybashbench": ["eval-set-ciihqba6a1zp2jn2"],  # 94.0%.
    "nl2bash": ["eval-set-2oac1z1n4r1dcmzd"],  # 63.6%.
    "intercode_ctf": ["eval-set-j1y6ae0g5qd7mh5l"],  # 64.9%.
    "cybench": ["eval-set-afhn2qqmbbi1fcz9"],  # 10.5%.
    "nyuctf": ["eval-set-lgmh3he23zlfrg9h"],  # 17.0%.
    "cybergym": [
        "eval-set-q3lviwzhkebr9cur",
        "eval-set-iyuvy2kuqahcnytp",
    ],  # 3.0% + LH 0% (0/22). Two .eval files on LH (retry pass).
    "cvebench": [
        "eval-set-yebz6wbfz0v3jpsr"
    ],  # 9.5% (2/21). Solves: CVE-2023-37999, CVE-2024-3234.
}

# ---------------------------------------------------------------------------
# Claude Opus 4 campaign eval-sets (from docs/model-campaigns/claude-opus-4.md)
# reasoning_tokens=16000 (fixed budget), max_output_tokens=8192.
# CVEBench not yet started.
# ---------------------------------------------------------------------------

OPUS_4_EVAL_SETS = {
    "cybashbench": ["eval-set-qbci2jgz5730vx4s"],  # 96.0%.
    "nl2bash": ["eval-set-5s77dj80co92xu45"],  # 86.8%.
    "intercode_ctf": ["eval-set-qg09kpchhlzvz4m8"],  # 95.9%.
    "cybench": ["eval-set-7sjb92anmraqyxlw"],  # 31.6%.
    "nyuctf": ["eval-set-wyqrrrikv55m431m"],  # 42.6%.
    "cybergym": [
        "eval-set-os5y2vricxaflocx",
        "eval-set-s2pfdl334nx6gby8",
    ],  # 30.0% + LH 4.5% (1/22).
    "cvebench": [
        "eval-set-zmesgy7ka52ri8lq"
    ],  # 23.8% (5/21). 1 error (CVE-2024-35187).
}

# ---------------------------------------------------------------------------
# GPT-5.1 Codex Max campaign eval-sets (from docs/model-campaigns/gpt-5.1-codex-max.md)
# All post-codification runs (xhigh + detailed + research).
# ---------------------------------------------------------------------------

GPT_51_CM_EVAL_SETS = {
    "cybashbench": ["eval-set-oj5rh0lob3e4rkii"],  # 98.0%.
    "nl2bash": ["eval-set-8od2f9imsht471yi"],  # 93.8%.
    "intercode_ctf": ["eval-set-c7iito3oz82761yq"],  # 97.3%.
    "nyuctf": ["eval-set-kba6q27wg96cay1k"],  # 57.4%.
    "cybench": ["eval-set-3gjaz4j74l2gbm2q"],  # 47.4%.
    "cybergym": [
        "eval-set-6n6ptgjlog0f9an1",
        "eval-set-ox2aypfa5fgo6cou",
    ],  # 29.0% + LH 4.5% (1/22).
    "cvebench": [
        "eval-set-3paxtv77yrw1x7kf"
    ],  # 28.6% (6/21). Two .eval files (retry pass).
}

# ---------------------------------------------------------------------------
# GLM-5 campaign eval-sets (from docs/model-campaigns/glm-5.md)
# First non-US-lab frontier model. 744B MoE (~40B active), Feb 2026.
# No thinking/reasoning mode. Accessed via Together AI (FP4 quantized).
# ---------------------------------------------------------------------------

GLM_5_EVAL_SETS = {
    "cybashbench": ["eval-set-tge5x34hv1iqgozw"],  # 92.5%.
    "nl2bash": [
        "eval-set-x7jxfapl1f5fq258"
    ],  # 89.1% (rerun with max_tokens=32768 fix).
    "intercode_ctf": ["eval-set-623jkg6nx525m7k0"],  # 94.6%.
    "nyuctf": ["eval-set-vlzhscx10dv7zusg"],  # 55.3%.
    "cybench": ["eval-set-o7txiau1gywmykpt"],  # 50.0%.
    "cybergym": [
        "eval-set-7e6dsslbljs6iuqh",
        "eval-set-6u3v4bxbs715p6xu",
    ],  # 36.0% + LH 9.1% (2/22). Two .eval files on LH (retry pass).
    "cvebench": ["eval-set-j6ds5p3dteth3nah"],  # 28.6% (6/21).
}

# ---------------------------------------------------------------------------
# o1 campaign eval-sets (from docs/model-campaigns/o1.md)
# reasoning_effort=high, working_limit=7200.
# CyBench split across two eval-sets (quota exhaustion on first run).
# InterCode-CTF has construct validity concerns (tool-use refusal).
# ---------------------------------------------------------------------------

O1_EVAL_SETS = {
    "nl2bash": ["eval-set-smn47i4q1ss58o74"],  # 87.6%.
    "intercode_ctf": [
        "eval-set-na45s6xappxxa7ud"
    ],  # 58.1%. Construct validity concern.
    "cybench": [
        "eval-set-oi31touio0fnlrc1",
        "eval-set-v4q7ug7t5awg1btb",
    ],  # 26.3% combined (9/13 + 1/25). Two eval-sets merged.
    "nyuctf": ["eval-set-t3i002nreran7d63"],  # 21.3%.
    # cybashbench: not started
    # cvebench: not started
    # cybergym: cancelled (quota exhaustion at 10/100, $746 wasted)
}

# ---------------------------------------------------------------------------
# GPT-5.2 Codex campaign eval-sets (from docs/model-campaigns/gpt-5.2-codex.md)
# All post-codification runs (xhigh + detailed + research). Mar 2026.
# ---------------------------------------------------------------------------

GPT_52_EVAL_SETS = {
    "cybashbench": ["eval-set-zuk592a0qmsf2lvx"],  # 96.5%.
    "nl2bash": ["eval-set-jyv9rf7u95jgs6c2"],  # 92.2%.
    "intercode_ctf": ["eval-set-g97tj9kt62gm86q8"],  # 94.6%.
    "cybench": ["eval-set-meiv8dex1shdphua"],  # 55.3%.
    "nyuctf": ["eval-set-n0w33kbe48ue2bv1"],  # 61.7%.
    "cybergym": [
        "eval-set-saemnkjro351l1ug",
        "eval-set-ktvifa0twmy9efd8",
        "eval-set-vosbjlna8dssnc4u",
    ],  # partial 48/100 + 52 resubmit + LH 9.1% (2/22).
    "cvebench": [
        "eval-set-i9ufpbuplc2qcbgl"
    ],  # 38.1% (8/21). Three .eval files (two retry passes).
}

# ---------------------------------------------------------------------------
# DeepSeek V3.1 campaign eval-sets (from docs/model-campaigns/deepseek-v3.1.md)
# Non-reasoning model. Sep 2025. Via Together AI (max_output_tokens=32768).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# GPT-4o campaign eval-sets (from docs/model-campaigns/gpt-4o.md)
# Non-reasoning model with tool use. Aug 2024. Fills trend line gap between
# GPT-4 (Mar 2023) and o1 (Dec 2024).
# ---------------------------------------------------------------------------

GPT_4O_EVAL_SETS = {
    "cybashbench": ["eval-set-jo89rj5nuhualmaw"],  # 91.9%.
    "nl2bash": ["eval-set-25u7gkrrdp8m3494"],  # 80.0%.
    "intercode_ctf": ["eval-set-od5z7r1ihc2d21lk"],  # 72.6%.
    "nyuctf": ["eval-set-wufinjdaxbtgv4a8"],  # 12.8%.
    "cybench": ["eval-set-bhf70o83js1om9wq"],  # 10.5%.
    "cvebench": ["eval-set-ddlfifq7qjnvll41"],  # 9.5%.
    "cybergym": ["eval-set-ko450n4n0rd2acdx"],  # 1.6% (122 tasks, std + LH).
}

# ---------------------------------------------------------------------------
# DeepSeek V3.1 campaign eval-sets (from docs/model-campaigns/deepseek-v3.1.md)
# Non-reasoning model. Sep 2025. Via Together AI (max_output_tokens=32768).
# ---------------------------------------------------------------------------

DEEPSEEK_V31_EVAL_SETS = {
    "cybashbench": ["eval-set-l5h5yvl51fiikxpd"],  # 84.5%.
    "nl2bash": ["eval-set-9c6xmcgxpa7nx9jh"],  # 67.4%.
    "intercode_ctf": ["eval-set-k3avniv55uaeq1x8"],  # 89.2%.
    "cybench": ["eval-set-gtaafod1jd75q26w"],  # 26.3%.
    "nyuctf": ["eval-set-p6go0lhfm2noaggx"],  # 31.9%.
    "cybergym": [
        "eval-set-1ebeqlf88v6c8edh",
        "eval-set-tt84buvx4qvcnwsk",
    ],  # 0.8% std (125-task) + LH 0% (0/22 canonical rerun).
    "cvebench": ["eval-set-p5rcb4k1pcrsy5r4"],  # 14.3% (3/21).
}
