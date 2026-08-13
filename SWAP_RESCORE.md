# Label-swap probe re-scoring

The swap condition inverts the action words. The scorer compared the model's answer against **unswapped** ground truth, so every non-turn-0 `opponent_last` answer was marked wrong and CPR collapsed to 0.200 - exactly the turn-0-only rate.

**Step 1 is the evidence. Step 2 is the correction. Read them in order** - if the contingency table is not a clean inversion, the correction is not justified.

No database is modified.


==============================================================================
## `exp2_llama_labelswap`
==============================================================================

### 1. `opponent_last`: what was true vs what the model said

_No rescoring applied. This table is the evidence._

```
     want vs got     Cooperate        Defect          none
       Cooperate            40        17,229             0
          Defect         7,834           460            37
            none             0             0         6,400
```

agrees: 6,900/32,000 (0.216) · exactly inverted: 25,063/32,000 (0.783)


The off-diagonal dominates: the model answered in the swapped label space and was graded against unswapped truth. Rescoring is justified.


### 2. CPR before and after

_All-or-nothing over three probes. Pre-registered gate 0.85._

```
   arm    opp        n   CPR as run   CPR rescored    gate
     1   allc    8,000        0.200          0.200    fail
     1    tft    8,000        0.200          0.200    fail
    3b   allc    8,000        0.200          0.200    fail
    3b    tft    8,000        0.200          0.200    fail
```

==============================================================================
## `exp2_qwen_labelswap`
==============================================================================

### 1. `opponent_last`: what was true vs what the model said

_No rescoring applied. This table is the evidence._

```
     want vs got     Cooperate        Defect          none
       Cooperate           235        19,077             0
          Defect         6,154           134             0
            none             0             0         6,400
```

agrees: 6,769/32,000 (0.212) · exactly inverted: 25,231/32,000 (0.788)


The off-diagonal dominates: the model answered in the swapped label space and was graded against unswapped truth. Rescoring is justified.


### 2. CPR before and after

_All-or-nothing over three probes. Pre-registered gate 0.85._

```
   arm    opp        n   CPR as run   CPR rescored    gate
     1   allc    8,000        0.200          0.200    fail
     1    tft    8,000        0.200          0.200    fail
    3b   allc    8,000        0.200          0.200    fail
    3b    tft    8,000        0.200          0.200    fail
```

==============================================================================
## `exp3_llama_swap`
==============================================================================

### 1. `opponent_last`: what was true vs what the model said

_No rescoring applied. This table is the evidence._

```
     want vs got     Cooperate        Defect          none
       Cooperate         1,145        51,997             0
          Defect        23,677         3,133            48
            none         4,000             0        16,000
```

agrees: 20,278/100,000 (0.203) · exactly inverted: 75,674/100,000 (0.757)


The off-diagonal dominates: the model answered in the swapped label space and was graded against unswapped truth. Rescoring is justified.


### 2. CPR before and after

_All-or-nothing over three probes. Pre-registered gate 0.85._

```
   arm    opp        n   CPR as run   CPR rescored    gate
     1   allc   10,000        0.200          0.200    fail
     1    tft   10,000        0.200          0.200    fail
     3   allc   10,000        0.200          1.000    PASS
     3    tft   10,000        0.200          1.000    PASS
    3b   allc   10,000        0.200          0.200    fail
    3b    tft   10,000        0.200          0.200    fail
    3c   allc   10,000        0.200          0.200    fail
    3c    tft   10,000        0.208          0.207    fail
    3d   allc   10,000        0.000          0.000    fail
    3d    tft   10,000        0.000          0.000    fail
```

==============================================================================
## `exp3_mistral_swap`
==============================================================================

### 1. `opponent_last`: what was true vs what the model said

_No rescoring applied. This table is the evidence._

```
     want vs got     Cooperate        Defect          NoneNone (since no round has been played yet)
       Cooperate            18        40,364             0             0
          Defect        38,699           919             0             0
            none             0             0         8,000        12,000
```

agrees: 8,937/100,000 (0.089) · exactly inverted: 79,063/100,000 (0.791)


The off-diagonal dominates: the model answered in the swapped label space and was graded against unswapped truth. Rescoring is justified.


### 2. CPR before and after

_All-or-nothing over three probes. Pre-registered gate 0.85._

```
   arm    opp        n   CPR as run   CPR rescored    gate
     1   allc   10,000        0.000          0.000    fail
     1    tft   10,000        0.000          0.000    fail
     3   allc   10,000        0.000          0.800    fail
     3    tft   10,000        0.000          0.800    fail
    3b   allc   10,000        0.000          0.000    fail
    3b    tft   10,000        0.000          0.000    fail
    3c   allc   10,000        0.000          0.000    fail
    3c    tft   10,000        0.000          0.000    fail
    3d   allc   10,000        0.000          0.000    fail
    3d    tft   10,000        0.000          0.002    fail
```

==============================================================================
## `exp3_qwen_swap`
==============================================================================

### 1. `opponent_last`: what was true vs what the model said

_No rescoring applied. This table is the evidence._

```
     want vs got     Cooperate        Defect          none
       Cooperate           624        52,361             0
          Defect        21,124         5,891             0
            none             0             0        20,000
```

agrees: 26,515/100,000 (0.265) · exactly inverted: 73,485/100,000 (0.735)


The off-diagonal dominates: the model answered in the swapped label space and was graded against unswapped truth. Rescoring is justified.


### 2. CPR before and after

_All-or-nothing over three probes. Pre-registered gate 0.85._

```
   arm    opp        n   CPR as run   CPR rescored    gate
     1   allc   10,000        0.200          0.200    fail
     1    tft   10,000        0.200          0.200    fail
     3   allc   10,000        0.200          1.000    PASS
     3    tft   10,000        0.200          1.000    PASS
    3b   allc   10,000        0.200          0.200    fail
    3b    tft   10,000        0.200          0.200    fail
    3c   allc   10,000        0.200          0.200    fail
    3c    tft   10,000        0.201          0.206    fail
    3d   allc   10,000        0.200          0.200    fail
    3d    tft   10,000        0.200          0.200    fail
```

==============================================================================
## What to do with this
==============================================================================

If arm 3 rescores to >= 0.85, the swap groups satisfy the pre-registered manipulation check and their behavioural results become usable — three of exp3's nine groups.

Record it in `EXPERIMENTS.md` as a **scorer** defect found after the fact, alongside the zero-padding and density defects, not as a silent correction. The behavioural data never changed; only its manipulation check was graded in the wrong label space.

One consequence to state plainly: `exp3_qwen_swap` was one of only two SUPPORTED sign-flip verdicts in the whole project, and it was voided *because* of this failed check. If the rescore passes, that verdict returns and has to be reported — including the fact that its ATE_true of +0.456 vs ALLC is an order of magnitude beyond anything in the semantic data.
