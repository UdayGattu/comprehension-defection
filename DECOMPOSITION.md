# Content vs schema, and the interaction

Episode-level bootstrap, 10,000 resamples, seed 20260811. The episode is the unit; arms and opponents are resampled independently because they are separate runs.

```
content     = P(D|3)  - P(D|3c)   same template, TRUE vs FALSE numbers
schema      = P(D|3c) - P(D|3b)   number-shaped fields present at all
ATE_true    = P(D|3)  - P(D|3b)   = content + schema
interaction = ATE_true(allc) - ATE_true(tft)
```

Block-reading is ~1.00 in both arm 3 and arm 3c, so the content contrast holds attention constant and varies only truth.


==============================================================================
## 1. Decomposition (needs arm 3c: exp2 and exp3 only)
==============================================================================

```
database                  opp  contrast        est              95% CI        p  
------------------------------------------------------------------------------------
exp2_llama                allc content     -0.0110  [-0.0174,-0.0044]   0.0018 ** 
exp2_llama                allc schema      -0.0007  [-0.0086,+0.0073]   0.8718    
exp2_llama                allc ATE_true    -0.0115  [-0.0194,-0.0039]   0.0034 ** 
exp2_llama                tft  content     -0.0176  [-0.0266,-0.0087]   0.0001 ***
exp2_llama                tft  schema      +0.0126  [+0.0028,+0.0226]   0.0156 *  
exp2_llama                tft  ATE_true    -0.0051  [-0.0153,+0.0048]   0.3108    
exp2_qwen                 allc content     +0.0014  [-0.0036,+0.0063]   0.5852    
exp2_qwen                 allc schema      -0.0135  [-0.0205,-0.0068]   0.0001 ***
exp2_qwen                 allc ATE_true    -0.0122  [-0.0187,-0.0058]   0.0004 ***
exp2_qwen                 tft  content     -0.2407  [-0.2476,-0.2335]   0.0001 ***
exp2_qwen                 tft  schema      +0.2378  [+0.2294,+0.2460]   0.0001 ***
exp2_qwen                 tft  ATE_true    -0.0029  [-0.0102,+0.0045]   0.4514    
exp3_llama_abs            allc content     -0.0003  [-0.0081,+0.0071]   0.9404    
exp3_llama_abs            allc schema      +0.0030  [-0.0057,+0.0121]   0.5102    
exp3_llama_abs            allc ATE_true    +0.0028  [-0.0059,+0.0115]   0.5274    
exp3_llama_abs            tft  content     +0.0208  [+0.0124,+0.0291]   0.0001 ***
exp3_llama_abs            tft  schema      -0.1935  [-0.2033,-0.1839]   0.0001 ***
exp3_llama_abs            tft  ATE_true    -0.1728  [-0.1817,-0.1640]   0.0001 ***
exp3_llama_sem            allc content     -0.0123  [-0.0188,-0.0057]   0.0004 ***
exp3_llama_sem            allc schema      -0.0013  [-0.0090,+0.0064]   0.7450    
exp3_llama_sem            allc ATE_true    -0.0135  [-0.0210,-0.0059]   0.0001 ***
exp3_llama_sem            tft  content     -0.0243  [-0.0321,-0.0164]   0.0001 ***
exp3_llama_sem            tft  schema      +0.0036  [-0.0058,+0.0131]   0.4590    
exp3_llama_sem            tft  ATE_true    -0.0206  [-0.0299,-0.0115]   0.0001 ***
exp3_llama_swap           allc content     +0.0220  [+0.0096,+0.0347]   0.0002 ***
exp3_llama_swap           allc schema      +0.0449  [+0.0333,+0.0566]   0.0001 ***
exp3_llama_swap           allc ATE_true    +0.0669  [+0.0550,+0.0788]   0.0001 ***
exp3_llama_swap           tft  content     +0.0201  [+0.0031,+0.0371]   0.0206 *  
exp3_llama_swap           tft  schema      +0.0131  [-0.0046,+0.0309]   0.1432    
exp3_llama_swap           tft  ATE_true    +0.0331  [+0.0157,+0.0506]   0.0004 ***
exp3_mistral_sem          allc content     -0.0000  [-0.0002,+0.0001]   0.5990    
exp3_mistral_sem          allc schema      -0.0002  [-0.0005,-0.0000]   0.0450 *  
exp3_mistral_sem          allc ATE_true    -0.0003  [-0.0005,-0.0001]   0.0064 ** 
exp3_mistral_sem          tft  content     -0.0000  [-0.0002,+0.0002]   0.8684    
exp3_mistral_sem          tft  schema      +0.0001  [-0.0000,+0.0003]   0.1066    
exp3_mistral_sem          tft  ATE_true    +0.0001  [-0.0000,+0.0003]   0.2032    
exp3_mistral_swap         allc content     -0.0002  [-0.0030,+0.0026]   0.9050    
exp3_mistral_swap         allc schema      +0.1351  [+0.1311,+0.1391]   0.0001 ***
exp3_mistral_swap         allc ATE_true    +0.1349  [+0.1308,+0.1390]   0.0001 ***
exp3_mistral_swap         tft  content     -0.0010  [-0.0018,-0.0002]   0.0230 *  
exp3_mistral_swap         tft  schema      +0.0120  [+0.0108,+0.0132]   0.0001 ***
exp3_mistral_swap         tft  ATE_true    +0.0110  [+0.0098,+0.0122]   0.0001 ***
exp3_qwen_abs             allc content     +0.0169  [+0.0127,+0.0212]   0.0001 ***
exp3_qwen_abs             allc schema      +0.7281  [+0.7221,+0.7338]   0.0001 ***
exp3_qwen_abs             allc ATE_true    +0.7451  [+0.7387,+0.7511]   0.0001 ***
exp3_qwen_abs             tft  content     +0.0202  [+0.0125,+0.0280]   0.0001 ***
exp3_qwen_abs             tft  schema      +0.7175  [+0.7088,+0.7261]   0.0001 ***
exp3_qwen_abs             tft  ATE_true    +0.7375  [+0.7295,+0.7452]   0.0001 ***
exp3_qwen_sem             allc content     +0.0056  [+0.0010,+0.0102]   0.0160 *  
exp3_qwen_sem             allc schema      -0.0021  [-0.0081,+0.0036]   0.4828    
exp3_qwen_sem             allc ATE_true    +0.0035  [-0.0020,+0.0090]   0.2062    
exp3_qwen_sem             tft  content     -0.2375  [-0.2444,-0.2308]   0.0001 ***
exp3_qwen_sem             tft  schema      +0.2535  [+0.2463,+0.2607]   0.0001 ***
exp3_qwen_sem             tft  ATE_true    +0.0160  [+0.0095,+0.0223]   0.0001 ***
exp3_qwen_swap            allc content     +0.0017  [-0.0070,+0.0103]   0.7054    
exp3_qwen_swap            allc schema      +0.4543  [+0.4469,+0.4617]   0.0001 ***
exp3_qwen_swap            allc ATE_true    +0.4559  [+0.4487,+0.4631]   0.0001 ***
exp3_qwen_swap            tft  content     +0.0395  [+0.0287,+0.0503]   0.0001 ***
exp3_qwen_swap            tft  schema      -0.1054  [-0.1158,-0.0955]   0.0001 ***
exp3_qwen_swap            tft  ATE_true    -0.0659  [-0.0754,-0.0563]   0.0001 ***
```

Across 20 valid cells: mean |content| = **0.0347**, mean |schema| = **0.1466** (4.2x).

A cell where |schema| exceeds |content| and both CIs exclude zero is direct evidence that ATE_true is dominated by the block's SHAPE rather than by the truth of its contents.


==============================================================================
## 2. The interaction — the actual pre-registered prediction
==============================================================================

The prediction is opponent-conditional: defection **down** vs TFT and **up** vs ALLC. That is a single quantity.

`interaction = ATE_true(allc) - ATE_true(tft)`, predicted **positive**.

Two same-signed effects are a main effect of defection, not the predicted conditional pattern, however large they are.

```
database                        ATE(allc)   ATE(tft)  interaction              95% CI        p  
--------------------------------------------------------------------------------------------------
exp2_llama                        -0.0116    -0.0051      -0.0065  [-0.0193,+0.0064]   0.3204    
exp2_qwen                         -0.0121    -0.0028      -0.0093  [-0.0191,+0.0004]   0.0620    
exp3_llama_abs                    +0.0028    -0.1728      +0.1756  [+0.1634,+0.1882]   0.0001 ***
exp3_llama_sem                    -0.0135    -0.0207      +0.0071  [-0.0047,+0.0188]   0.2386    
exp3_llama_swap                   +0.0670    +0.0332      +0.0339  [+0.0126,+0.0549]   0.0020 ** 
exp3_mistral_abs                  +0.0078    -0.2266      +0.2345  [+0.2284,+0.2406]   0.0001 ***  VOID
exp3_mistral_sem                  -0.0003    +0.0001      -0.0004  [-0.0006,-0.0001]   0.0018 ** 
exp3_mistral_swap                 +0.1349    +0.0110      +0.1239  [+0.1196,+0.1283]   0.0001 ***
exp3_qwen_abs                     +0.7451    +0.7376      +0.0076  [-0.0023,+0.0176]   0.1274    
exp3_qwen_sem                     +0.0035    +0.0160      -0.0125  [-0.0209,-0.0041]   0.0028 ** 
exp3_qwen_swap                    +0.4559    -0.0659      +0.5219  [+0.5101,+0.5337]   0.0001 ***
exp4_llama_abs_logit              +0.0003    -0.1847      +0.1850  [+0.1676,+0.2026]   0.0001 ***
exp4_llama_abs_scratchpad         +0.1179    +0.0110      +0.1069  [+0.0888,+0.1247]   0.0001 ***
exp4_llama_sem_logit              -0.0145    -0.0229      +0.0084  [-0.0089,+0.0259]   0.3338    
exp4_llama_sem_scratchpad         -0.0241    -0.0541      +0.0301  [+0.0062,+0.0538]   0.0124 *  
exp4_mistral_abs_logit            +0.0041    -0.2191      +0.2232  [+0.2145,+0.2319]   0.0001 ***  VOID
exp4_mistral_abs_scratchpad       +0.1364    +0.0375      +0.0989  [+0.0861,+0.1118]   0.0001 ***  VOID
exp4_mistral_sem_logit            -0.0001    -0.0002      +0.0001  [-0.0004,+0.0005]   0.8506    
exp4_mistral_sem_scratchpad       +0.0674    +0.0315      +0.0358  [+0.0141,+0.0574]   0.0004 ***
exp4_qwen_abs_logit               +0.7408    +0.7373      +0.0037  [-0.0126,+0.0199]   0.6528    
exp4_qwen_abs_scratchpad          -0.0062    -0.0268      +0.0207  [+0.0064,+0.0350]   0.0042 ** 
exp4_qwen_sem_logit               +0.0090    +0.0203      -0.0112  [-0.0221,-0.0004]   0.0426 *  
exp4_qwen_sem_scratchpad          -0.2136    -0.1015      -0.1121  [-0.1356,-0.0883]   0.0001 ***
exp5_llama_sem_minimal            +0.0739    +0.0237      +0.0501  [+0.0311,+0.0686]   0.0001 ***
exp5_mistral_sem_minimal          +0.0294    +0.0360      -0.0064  [-0.0199,+0.0069]   0.3496    
exp5_qwen_sem_minimal             +0.0438    +0.0171      +0.0265  [-0.0011,+0.0542]   0.0630    
sweep                             +0.0420    +0.0525      -0.0106  [-0.0203,-0.0010]   0.0302 *  
```

Read the sign, not the magnitude. A significantly **positive** interaction is the only result that supports the registration. A significantly negative one contradicts it. Zero is a null.
