# Arm 3c, stratified: does the donor's number change play?

`09_dose_response.py` pools all turns, so its gradient is confounded: `lie = donor_score - true_score` and `true_score` is the recipient's own cumulative payoff, so a large lie marks an agent that was *already* defecting.

`cdx/donor.py` selects uniformly at random among live episodes with a distinct fingerprint, so **conditional on the recipient's state the donor is randomly assigned**. Stratifying on `(turn, true_score)` holds the recipient's history nearly fixed and makes the contrast causal.

Both are shown. **The gap between them is the confound, measured.**

Bootstrap over episodes, 2,000 resamples, seed 20260811. Strata with fewer than 30 rows or missing either direction are dropped — they carry no information about the contrast.


`effect = P(defect | donor < true) - P(defect | donor > true)`

Positive means a block understating the score drives more defection.


==============================================================================
## `exp3_llama_abs`
==============================================================================

```
opp        naive              95% CI   stratified              95% CI  strata        n
allc     +0.0334  [+0.0236,+0.0430]      -0.0162  [-0.0293,+0.0005]     101   31,156
tft      -0.0880  [-0.0987,-0.0773]      -0.0908  [-0.1075,-0.0759]     219   29,533
```

==============================================================================
## `exp3_llama_sem`
==============================================================================

```
opp        naive              95% CI   stratified              95% CI  strata        n
allc     +0.0673  [+0.0573,+0.0780]      -0.0231  [-0.0389,-0.0040]      51   15,199
tft      -0.0263  [-0.0361,-0.0172]      +0.0230  [+0.0116,+0.0361]      93   30,198
```

==============================================================================
## `exp3_llama_swap`
==============================================================================

```
opp        naive              95% CI   stratified              95% CI  strata        n
allc     +0.2593  [+0.2478,+0.2706]      -0.0005  [-0.0145,+0.0191]     132   27,846
tft      -0.2170  [-0.2335,-0.1995]      +0.0240  [+0.0110,+0.0369]     203   30,127
```

==============================================================================
## `exp3_mistral_abs`
==============================================================================

```
opp        naive              95% CI   stratified              95% CI  strata        n
allc     -0.0164  [-0.0222,-0.0108]      +0.0092  [+0.0014,+0.0168]      56   28,505  VOID
tft      -0.0984  [-0.1096,-0.0885]      -0.1567  [-0.1692,-0.1416]     175   27,694  VOID
```

==============================================================================
## `exp3_mistral_sem`
==============================================================================

```
opp        naive              95% CI   stratified              95% CI  strata        n
allc     -0.0000  [-0.0001,+0.0000]         +nan  [+nan,+nan]       0        0
tft      +0.0001  [+0.0000,+0.0003]      +0.0000  [+0.0000,+0.0000]       3    5,982
```

==============================================================================
## `exp3_mistral_swap`
==============================================================================

```
opp        naive              95% CI   stratified              95% CI  strata        n
allc     -0.0384  [-0.0441,-0.0327]      +0.0060  [-0.0157,+0.0176]      25   17,721
tft      -0.0088  [-0.0121,-0.0056]      -0.0004  [-0.0020,+0.0006]      11   19,476
```

==============================================================================
## `exp3_qwen_abs`
==============================================================================

```
opp        naive              95% CI   stratified              95% CI  strata        n
allc     +0.0733  [+0.0659,+0.0804]      -0.0111  [-0.0220,-0.0048]      64   25,560
tft      -0.0442  [-0.0521,-0.0367]      -0.0371  [-0.0508,-0.0261]     171   27,438
```

==============================================================================
## `exp3_qwen_sem`
==============================================================================

```
opp        naive              95% CI   stratified              95% CI  strata        n
allc     +0.2995  [+0.2783,+0.3213]      +0.1089  [+0.0387,+0.1672]      16    2,133
tft      +0.1226  [+0.1116,+0.1326]      +0.1289  [+0.1086,+0.1526]     115   25,625
```

==============================================================================
## `exp3_qwen_swap`
==============================================================================

```
opp        naive              95% CI   stratified              95% CI  strata        n
allc     +0.1111  [+0.1007,+0.1215]      +0.0207  [-0.0053,+0.0483]      85   29,368
tft      -0.1336  [-0.1451,-0.1218]      +0.0510  [+0.0136,+0.0684]     145   30,033
```

==============================================================================
## How to read it
==============================================================================

**Stratified effect close to the naive one** — the gradient survives randomisation. The model responds to what the block says, graded by how wrong it is. That is the strongest positive evidence of content use in the study, and it makes the replicated qwen-vs-TFT content effect (−0.2407 in exp2, −0.2375 in exp3) a dose-response rather than an anomaly.

**Stratified effect near zero while the naive one is large** — the gradient was the agent's own trajectory. It agrees with `08_decomposition_ci.py`, which finds content ≈ 0 and schema large, by a completely independent route. Two routes to the same conclusion is worth more than either alone.


Whichever it is, report both numbers. The naive figure is what a study without the stratification would have published, and the distance between them is the size of the mistake it would have made.
