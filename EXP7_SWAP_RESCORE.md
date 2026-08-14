# Label-swap probe re-scoring

The swap condition inverts the action words. The scorer compared the model's answer against **unswapped** ground truth, so every non-turn-0 `opponent_last` answer was marked wrong and CPR collapsed to 0.200 - exactly the turn-0-only rate.

**Step 1 is the evidence. Step 2 is the correction. Read them in order** - if the contingency table is not a clean inversion, the correction is not justified.

No database is modified.


==============================================================================
## `exp7_llama_swap_logit`
==============================================================================

### 1. `opponent_last`: what was true vs what the model said

_No rescoring applied. This table is the evidence._

```
     want vs got     Cooperate        Defect          none
       Cooperate         4,050        22,038             0
          Defect        11,393         2,492            27
            none             0             0        10,000
```

agrees: 16,542/50,000 (0.331) · exactly inverted: 33,431/50,000 (0.669)


The off-diagonal dominates: the model answered in the swapped label space and was graded against unswapped truth. Rescoring is justified.


### 2. CPR before and after

_All-or-nothing over three probes. Pre-registered gate 0.85._

```
   arm    opp        n   CPR as run   CPR rescored    gate
     1   allc    5,000        0.200          0.200    fail
     1    tft    5,000        0.200          0.200    fail
     3   allc    5,000        0.200          1.000    PASS
     3    tft    5,000        0.200          1.000    PASS
    3b   allc    5,000        0.200          0.200    fail
    3b    tft    5,000        0.200          0.200    fail
    3m   allc    5,000        0.847          0.353    fail
    3m    tft    5,000        0.795          0.405    fail
    3s   allc    5,000        0.000          0.000    fail
    3s    tft    5,000        0.000          0.000    fail
```

==============================================================================
## `exp7_qwen_swap_logit`
==============================================================================

### 1. `opponent_last`: what was true vs what the model said

_No rescoring applied. This table is the evidence._

```
     want vs got     Cooperate        Defect          none
       Cooperate           621        26,329             0
          Defect        12,502           548             0
            none             0             0        10,000
```

agrees: 11,169/50,000 (0.223) · exactly inverted: 38,831/50,000 (0.777)


The off-diagonal dominates: the model answered in the swapped label space and was graded against unswapped truth. Rescoring is justified.


### 2. CPR before and after

_All-or-nothing over three probes. Pre-registered gate 0.85._

```
   arm    opp        n   CPR as run   CPR rescored    gate
     1   allc    5,000        0.200          0.200    fail
     1    tft    5,000        0.200          0.200    fail
     3   allc    5,000        0.200          1.000    PASS
     3    tft    5,000        0.200          1.000    PASS
    3b   allc    5,000        0.200          0.200    fail
    3b    tft    5,000        0.200          0.200    fail
    3m   allc    5,000        0.292          0.908    PASS
    3m    tft    5,000        0.299          0.894    PASS
    3s   allc    5,000        0.000          0.000    fail
    3s    tft    5,000        0.000          0.000    fail
```

==============================================================================
## What to do with this
==============================================================================

If arm 3 rescores to >= 0.85, the swap groups satisfy the pre-registered manipulation check and their behavioural results become usable — three of exp3's nine groups.

Record it in `EXPERIMENTS.md` as a **scorer** defect found after the fact, alongside the zero-padding and density defects, not as a silent correction. The behavioural data never changed; only its manipulation check was graded in the wrong label space.

One consequence to state plainly: `exp3_qwen_swap` was one of only two SUPPORTED sign-flip verdicts in the whole project, and it was voided *because* of this failed check. If the rescore passes, that verdict returns and has to be reported — including the fact that its ATE_true of +0.456 vs ALLC is an order of magnitude beyond anything in the semantic data.
