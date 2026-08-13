# Prerequisites for a new-opponent experiment

Two gates, both from data already on disk. Either can stop the experiment; **B can stop it for a reason no opponent design fixes**.


==============================================================================
## Gate A — is arm 3c already dead?
==============================================================================

`d = donor_score − true_score` is the falsification each arm-3c row carries. Degenerate-donor rows are excluded — there the donor *is* the recipient's state and `d = 0` by construction.

`select_donor` samples from `live_states`, and `run_cell` runs per (arm, opponent), so donors share the recipient's opponent and round. **If the sampling is symmetric, `E[d] = 0`** and a linear response gives a zero mean contrast even for a model that follows the lie perfectly.

At the measured slope of ≈0.01 defection per point of error, `|d| ≥ 15` is the range that moves a decision by ≈15pp.

```
database                  opp           n     E[d]    sd(d)    min    max  P|d|>=15  P d<=-15  P d>=+15
exp3_llama_abs            allc     38,000    -0.12     4.95    -24     24     0.006     0.003     0.003
exp3_llama_abs            tft      38,000     0.00     5.40    -28     25     0.017     0.009     0.008
exp3_llama_sem            allc     38,000     0.98     4.17    -32     32     0.011     0.004     0.006
exp3_llama_sem            tft      38,000    -0.02     3.23    -26     25     0.006     0.003     0.003
exp3_llama_swap           allc     38,000    -0.24     7.48    -32     28     0.056     0.028     0.028
exp3_llama_swap           tft      38,000     0.53     8.84    -34     36     0.116     0.053     0.063
exp3_mistral_abs          allc     36,000    -0.15     3.26    -12     14     0.000     0.000     0.000
exp3_mistral_abs          tft      36,000    -0.03     4.62    -21     22     0.006     0.003     0.003
exp3_mistral_sem          allc     34,000     1.99     0.19     -2      2     0.000     0.000     0.000
exp3_mistral_sem          tft      34,000    -0.80     0.75     -2      2     0.000     0.000     0.000
exp3_mistral_swap         allc     36,000    -0.43     2.36    -12     12     0.000     0.000     0.000
exp3_mistral_swap         tft      36,000     3.26     2.43    -12     12     0.000     0.000     0.000
exp3_qwen_abs             allc     38,000     0.75     4.84    -16     14     0.000     0.000     0.000
exp3_qwen_abs             tft      38,000     0.32     7.56    -29     29     0.052     0.024     0.028
exp3_qwen_sem             allc     36,000     3.00     3.30    -32     30     0.007     0.001     0.006
exp3_qwen_sem             tft      36,000     0.21     3.30    -21     19     0.001     0.000     0.000
exp3_qwen_swap            allc     36,000    -0.27     5.53    -20     20     0.005     0.003     0.003
exp3_qwen_swap            tft      38,000     1.14     9.08    -32     30     0.145     0.058     0.086
```

**Reading it.** `sd(d)` near zero means arm 3c is numerically almost arm 3 and has no power whatever the opponent — that alone would end the design. `E[d]` near zero with large `sd(d)` means the *mean* contrast is weak but a **signed** split (3c− vs 3c+) recovers the power, provided `P(d ≤ −15)` and `P(d ≥ +15)` are both non-trivial. Compare `E[d]` against `sd(d)/sqrt(n)` before calling it zero.


==============================================================================
## Gate B — does any model plan toward the horizon?
==============================================================================

Measured in **arm 3**, the true-state arm — the condition most favourable to planning. If defection is flat in the round index here, there is no end-game switch point, nothing for a falsified score to move, and **no opponent design creates one**.

`slope` is OLS of P(defect) on round. `last−first` compares the final quarter of the game against the first, which survives non-linearity.

```
database                      opp        slope  first q   last q  last-first    off
exp2_llama                    allc    +0.00488    0.040    0.113      +0.074  0.000
exp2_llama                    tft     +0.00667    0.046    0.145      +0.099  0.000
exp2_qwen                     allc    +0.00984    0.004    0.156      +0.152  0.000
exp2_qwen                     tft     +0.01318    0.003    0.205      +0.202  0.000
exp3_llama_abs                allc    +0.00068    0.704    0.703      -0.001  0.000
exp3_llama_abs                tft     +0.00071    0.589    0.592      +0.003  0.000
exp3_llama_sem                allc    +0.00434    0.046    0.109      +0.063  0.000
exp3_llama_sem                tft     +0.00693    0.043    0.148      +0.105  0.000
exp3_llama_swap               allc    -0.03400    0.726    0.202      -0.524  0.000
exp3_llama_swap               tft     -0.00767    0.840    0.723      -0.117  0.000
exp3_mistral_abs              allc    +0.02960    0.578    0.996      +0.418  1.000  VOID
exp3_mistral_abs              tft     +0.02423    0.405    0.780      +0.375  1.000  VOID
exp3_mistral_sem              allc    -0.00001    0.000    0.000      -0.000  0.000
exp3_mistral_sem              tft     -0.00002    0.000    0.000      -0.000  0.000
exp3_mistral_swap             allc    -0.00517    0.942    0.890      -0.053  0.000
exp3_mistral_swap             tft     +0.00051    0.990    0.999      +0.009  0.000
exp3_qwen_abs                 allc    +0.03468    0.329    0.912      +0.583  0.000
exp3_qwen_abs                 tft     +0.03338    0.382    0.952      +0.570  0.000
exp3_qwen_sem                 allc    +0.00993    0.004    0.159      +0.155  0.000
exp3_qwen_sem                 tft     +0.01383    0.003    0.220      +0.217  0.000
exp3_qwen_swap                allc    -0.00017    0.668    0.681      +0.013  0.000
exp3_qwen_swap                tft     +0.01075    0.815    0.973      +0.158  0.000
exp4_llama_abs_logit          allc    +0.00275    0.677    0.714      +0.037  0.000
exp4_llama_abs_logit          tft     +0.00021    0.576    0.576      +0.001  0.000
exp4_llama_abs_scratchpad     allc    +0.00271    0.640    0.677      +0.036  0.024
exp4_llama_abs_scratchpad     tft     -0.00174    0.571    0.540      -0.031  0.018
exp4_llama_sem_logit          allc    +0.00402    0.048    0.110      +0.062  0.000
exp4_llama_sem_logit          tft     +0.00680    0.046    0.149      +0.102  0.000
exp4_llama_sem_scratchpad     allc    +0.01143    0.452    0.630      +0.178  0.001
exp4_llama_sem_scratchpad     tft     +0.01485    0.467    0.695      +0.227  0.001
exp4_mistral_abs_logit        allc    +0.02900    0.591    0.995      +0.404  1.000  VOID
exp4_mistral_abs_logit        tft     +0.02386    0.417    0.787      +0.370  1.000  VOID
exp4_mistral_abs_scratchpad   allc    +0.00758    0.643    0.752      +0.109  0.811  VOID
exp4_mistral_abs_scratchpad   tft     +0.00299    0.501    0.551      +0.050  0.774  VOID
exp4_mistral_sem_logit        allc    -0.00002    0.000    0.000      -0.000  0.000
exp4_mistral_sem_logit        tft     +0.00001    0.000    0.000      +0.000  0.000
exp4_mistral_sem_scratchpad   allc    +0.01406    0.267    0.477      +0.209  0.120  VOID
exp4_mistral_sem_scratchpad   tft     +0.00743    0.259    0.369      +0.110  0.124  VOID
exp4_qwen_abs_logit           allc    +0.03031    0.389    0.893      +0.504  0.000
exp4_qwen_abs_logit           tft     +0.02863    0.455    0.942      +0.486  0.000
exp4_qwen_abs_scratchpad      allc    +0.00497    0.789    0.873      +0.084  0.196  VOID
exp4_qwen_abs_scratchpad      tft     +0.00695    0.831    0.942      +0.111  0.196  VOID
exp4_qwen_sem_logit           allc    +0.01025    0.004    0.165      +0.161  0.000
exp4_qwen_sem_logit           tft     +0.01307    0.005    0.209      +0.204  0.000
exp4_qwen_sem_scratchpad      allc    +0.02053    0.255    0.562      +0.307  0.081
exp4_qwen_sem_scratchpad      tft     +0.03563    0.292    0.825      +0.534  0.072
exp5_llama_sem_minimal        allc    +0.01076    0.607    0.766      +0.159  0.001
exp5_llama_sem_minimal        tft     +0.01064    0.583    0.747      +0.164  0.000
exp5_mistral_sem_minimal      allc    +0.01271    0.313    0.493      +0.180  0.108  VOID
exp5_mistral_sem_minimal      tft     +0.00967    0.309    0.456      +0.146  0.115  VOID
exp5_qwen_sem_minimal         allc    +0.01296    0.521    0.720      +0.199  0.123  VOID
exp5_qwen_sem_minimal         tft     +0.01939    0.548    0.847      +0.299  0.143  VOID
sweep                         allc    +0.00505    0.039    0.116      +0.078  0.000
sweep                         tft     +0.00701    0.043    0.149      +0.106  0.000
```

**Reading it.** A positive slope with `last−first` well above zero means the model shortens its horizon as the game ends — the mechanism the experiment needs. Flat or negative in every valid cell means the models never engage it, and the honest response is to report that as a capability finding rather than to build a fourth opponent around a mechanism that is not there.


A caveat worth stating in the writeup: a rising profile is also consistent with defection simply being sticky — one defection provokes TFT, which provokes more. Against ALLC there is no such feedback, so **the ALLC column is the cleaner evidence of end-game planning**; if the rise appears only against TFT it is more likely retaliation spirals than horizon reasoning.
