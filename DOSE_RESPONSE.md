# Arm 3c: does defection scale with the size of the lie?

`donor_agent_score` is the number the block displayed. The true score is reconstructed as the cumulative `agent_payoff` of the same episode up to (not including) the current turn.

`lie = donor_agent_score - true_score`. Rows with `donor_degenerate = 1` are excluded: there the donor is the recipient's own state and the lie is zero by construction.


==============================================================================
## `exp3_llama_abs`
==============================================================================

reconstruction check: true score is 0 at turn 0 in every episode — OK


**vs allc** — n=38,000 falsified turns, off-task 0.000

*by absolute size of the lie*

```
     |lie|        n   P(defect)
       1-3   18,316      0.7260
       4-8   17,190      0.7274
      9-15    2,264      0.6855
     16-25      230      0.6391
```

*by signed lie — a block claiming a HIGHER score than the truth may mean something different from one claiming lower*

```
   direction        n   P(defect)
       lower   19,564      0.7399
      higher   18,436      0.7065
```

**vs tft** — n=38,000 falsified turns, off-task 0.000

*by absolute size of the lie*

```
     |lie|        n   P(defect)
         0    2,099      0.5855
       1-3   19,180      0.5869
       4-8   12,318      0.5792
      9-15    3,981      0.5501
     16-25      420      0.5095
       26+        2      0.5000
```

*by signed lie — a block claiming a HIGHER score than the truth may mean something different from one claiming lower*

```
   direction        n   P(defect)
       lower   17,931      0.5352
       equal    2,099      0.5855
      higher   17,970      0.6232
```

==============================================================================
## `exp3_llama_sem`
==============================================================================

reconstruction check: true score is 0 at turn 0 in every episode — OK


**vs allc** — n=38,000 falsified turns, off-task 0.000

*by absolute size of the lie*

```
     |lie|        n   P(defect)
       1-3   25,704      0.0855
       4-8   10,727      0.1236
      9-15    1,170      0.2000
     16-25      372      0.2984
       26+       27      0.3704
```

*by signed lie — a block claiming a HIGHER score than the truth may mean something different from one claiming lower*

```
   direction        n   P(defect)
       lower   12,745      0.1468
      higher   25,255      0.0795
```

**vs tft** — n=38,000 falsified turns, off-task 0.000

*by absolute size of the lie*

```
     |lie|        n   P(defect)
         0    2,020      0.1530
       1-3   30,312      0.1125
       4-8    4,633      0.1915
      9-15      871      0.3203
     16-25      163      0.4172
       26+        1      0.0000
```

*by signed lie — a block claiming a HIGHER score than the truth may mean something different from one claiming lower*

```
   direction        n   P(defect)
       lower   18,059      0.1160
       equal    2,020      0.1530
      higher   17,921      0.1422
```

==============================================================================
## `exp3_llama_swap`
==============================================================================

reconstruction check: true score is 0 at turn 0 in every episode — OK


**vs allc** — n=38,000 falsified turns, off-task 0.000

*by absolute size of the lie*

```
     |lie|        n   P(defect)
       1-3   14,047      0.4783
       4-8   15,918      0.2838
      9-15    5,911      0.2341
     16-25    2,057      0.2591
       26+       67      0.3731
```

*by signed lie — a block claiming a HIGHER score than the truth may mean something different from one claiming lower*

```
   direction        n   P(defect)
       lower   20,976      0.4629
      higher   17,024      0.2037
```

**vs tft** — n=38,000 falsified turns, off-task 0.000

*by absolute size of the lie*

```
     |lie|        n   P(defect)
         0      686      0.7522
       1-3   18,454      0.8048
       4-8    9,295      0.7480
      9-15    5,704      0.6462
     16-25    3,214      0.5439
       26+      647      0.5502
```

*by signed lie — a block claiming a HIGHER score than the truth may mean something different from one claiming lower*

```
   direction        n   P(defect)
       lower   18,239      0.6286
       equal      686      0.7522
      higher   19,075      0.8456
```

==============================================================================
## `exp3_mistral_abs`
==============================================================================

reconstruction check: true score is 0 at turn 0 in every episode — OK


**vs allc** — n=36,000 falsified turns, off-task 1.000  **VOID**

*by absolute size of the lie*

```
     |lie|        n   P(defect)
       1-3   24,085      0.8908
       4-8   11,776      0.9462
      9-15      139      0.9928
```

*by signed lie — a block claiming a HIGHER score than the truth may mean something different from one claiming lower*

```
   direction        n   P(defect)
       lower   18,740      0.9015
      higher   17,260      0.9178
```

**vs tft** — n=36,000 falsified turns, off-task 0.999  **VOID**

*by absolute size of the lie*

```
     |lie|        n   P(defect)
         0    2,164      0.6419
       1-3   20,198      0.6223
       4-8   10,963      0.7057
      9-15    2,534      0.7790
     16-25      141      0.8014
```

*by signed lie — a block claiming a HIGHER score than the truth may mean something different from one claiming lower*

```
   direction        n   P(defect)
       lower   17,158      0.6133
       equal    2,164      0.6419
      higher   16,678      0.7117
```

==============================================================================
## `exp3_mistral_sem`
==============================================================================

reconstruction check: true score is 0 at turn 0 in every episode — OK


**vs allc** — n=34,000 falsified turns, off-task 0.000

*by absolute size of the lie*

```
     |lie|        n   P(defect)
       1-3   34,000      0.0000
```

*by signed lie — a block claiming a HIGHER score than the truth may mean something different from one claiming lower*

```
   direction        n   P(defect)
       lower       80      0.0000
      higher   33,920      0.0000
```

**vs tft** — n=34,000 falsified turns, off-task 0.000

*by absolute size of the lie*

```
     |lie|        n   P(defect)
       1-3   34,000      0.0001
```

*by signed lie — a block claiming a HIGHER score than the truth may mean something different from one claiming lower*

```
   direction        n   P(defect)
       lower   31,688      0.0001
      higher    2,312      0.0000
```

==============================================================================
## `exp3_mistral_swap`
==============================================================================

reconstruction check: true score is 0 at turn 0 in every episode — OK


**vs allc** — n=36,000 falsified turns, off-task 0.000

*by absolute size of the lie*

```
     |lie|        n   P(defect)
       1-3   31,462      0.9092
       4-8    4,530      0.8974
      9-15        8      0.7500
```

*by signed lie — a block claiming a HIGHER score than the truth may mean something different from one claiming lower*

```
   direction        n   P(defect)
       lower   21,580      0.8923
      higher   14,420      0.9307
```

**vs tft** — n=36,000 falsified turns, off-task 0.000

*by absolute size of the lie*

```
     |lie|        n   P(defect)
       1-3   27,369      0.9982
       4-8    8,172      0.9993
      9-15      459      0.9978
```

*by signed lie — a block claiming a HIGHER score than the truth may mean something different from one claiming lower*

```
   direction        n   P(defect)
       lower    4,150      0.9906
      higher   31,850      0.9994
```

==============================================================================
## `exp3_qwen_abs`
==============================================================================

reconstruction check: true score is 0 at turn 0 in every episode — OK


**vs allc** — n=38,000 falsified turns, off-task 0.000

*by absolute size of the lie*

```
     |lie|        n   P(defect)
       1-3   13,673      0.8489
       4-8   23,962      0.7289
      9-15      364      0.9121
     16-25        1      1.0000
```

*by signed lie — a block claiming a HIGHER score than the truth may mean something different from one claiming lower*

```
   direction        n   P(defect)
       lower   16,271      0.8157
      higher   21,729      0.7425
```

**vs tft** — n=38,000 falsified turns, off-task 0.000

*by absolute size of the lie*

```
     |lie|        n   P(defect)
         0    2,542      0.4622
       1-3   12,392      0.7042
       4-8   11,963      0.8284
      9-15    9,857      0.8575
     16-25    1,226      0.8075
       26+       20      0.8000
```

*by signed lie — a block claiming a HIGHER score than the truth may mean something different from one claiming lower*

```
   direction        n   P(defect)
       lower   17,185      0.7696
       equal    2,542      0.4622
      higher   18,273      0.8138
```

==============================================================================
## `exp3_qwen_sem`
==============================================================================

reconstruction check: true score is 0 at turn 0 in every episode — OK


**vs allc** — n=36,000 falsified turns, off-task 0.000

*by absolute size of the lie*

```
     |lie|        n   P(defect)
       1-3   19,018      0.0430
       4-8   15,559      0.0531
      9-15    1,176      0.1080
     16-25      230      0.1783
       26+       17      0.3529
```

*by signed lie — a block claiming a HIGHER score than the truth may mean something different from one claiming lower*

```
   direction        n   P(defect)
       lower    3,016      0.3249
      higher   32,984      0.0254
```

**vs tft** — n=36,000 falsified turns, off-task 0.000

*by absolute size of the lie*

```
     |lie|        n   P(defect)
         0    4,900      0.2751
       1-3   23,042      0.3179
       4-8    7,178      0.4306
      9-15      866      0.5346
     16-25       14      0.5714
```

*by signed lie — a block claiming a HIGHER score than the truth may mean something different from one claiming lower*

```
   direction        n   P(defect)
       lower   13,838      0.4181
       equal    4,900      0.2751
      higher   17,262      0.2956
```

==============================================================================
## `exp3_qwen_swap`
==============================================================================

reconstruction check: true score is 0 at turn 0 in every episode — OK


**vs allc** — n=36,000 falsified turns, off-task 0.000

*by absolute size of the lie*

```
     |lie|        n   P(defect)
       1-3   16,258      0.6778
       4-8   15,830      0.7359
      9-15    3,720      0.7086
     16-25      192      0.6667
```

*by signed lie — a block claiming a HIGHER score than the truth may mean something different from one claiming lower*

```
   direction        n   P(defect)
       lower   19,363      0.7578
      higher   16,637      0.6467
```

**vs tft** — n=38,000 falsified turns, off-task 0.000

*by absolute size of the lie*

```
     |lie|        n   P(defect)
         0      308      0.8312
       1-3   18,485      0.8370
       4-8    7,415      0.8456
      9-15    7,326      0.8867
     16-25    4,214      0.8671
       26+      252      0.6667
```

*by signed lie — a block claiming a HIGHER score than the truth may mean something different from one claiming lower*

```
   direction        n   P(defect)
       lower   17,304      0.7783
       equal      308      0.8312
      higher   20,388      0.9119
```

==============================================================================
## How to read this
==============================================================================

**Rising P(defect) with |lie|** — the model responds to the block's content, graded by how wrong it is. That upgrades the qwen-vs-TFT result from one anomalous cell to a dose-response relationship, and it is the strongest positive evidence of content use the study can produce.

**Flat across bins** — the model registers that a number-shaped field exists and not which number it holds. That is the same conclusion `analysis/08_decomposition_ci.py` reaches from the other direction, and two independent routes to it is worth more than either alone.

**Asymmetry between higher and lower** — a falsely high score may read as 'I am winning, press the advantage' and a falsely low one as 'I am behind'. If the signed table splits and the absolute one does not, the effect is directional and |lie| is the wrong summary.


These are raw rates, not contrasts, and turns within an episode are not independent. Treat any pattern here as descriptive until it is re-estimated at the episode level.
