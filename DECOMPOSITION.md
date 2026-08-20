# Veracity vs schema, and the interaction

Episode-level bootstrap, 10,000 resamples, seed 20260811. The episode is the unit; arms and opponents are resampled independently because they are separate runs.

Turn filter: `excl_t0`. Turn 0 is dropped from every arm, so both sides of each contrast run on the same 19 turns.

```
veracity    = P(D|3)  - P(D|3c)   same template, TRUE vs FALSE numbers
schema      = P(D|3c) - P(D|3b)   number-shaped fields present at all
ATE_true    = P(D|3)  - P(D|3b)   = veracity + schema
interaction = ATE_true(allc) - ATE_true(tft)
```

Block-reading is ~1.00 in both arm 3 and arm 3c, so the veracity contrast holds attention constant and varies only truth.


==============================================================================
## 1. Decomposition (needs arm 3c: exp2, exp3 and exp6)
==============================================================================

```
database                  opp  contrast        est              95% CI        p      dose  
----------------------------------------------------------------------------------------------
exp2_llama                allc veracity    -0.0113  [-0.0182,-0.0045]   0.0018       n/r ** 
exp2_llama                allc schema      -0.0000  [-0.0083,+0.0083]   0.9966              
exp2_llama                allc ate_true    -0.0113  [-0.0196,-0.0032]   0.0072           ** 
exp2_llama                tft  veracity    -0.0184  [-0.0279,-0.0091]   0.0001       n/r ***
exp2_llama                tft  schema      +0.0139  [+0.0037,+0.0243]   0.0110           *  
exp2_llama                tft  ate_true    -0.0047  [-0.0153,+0.0058]   0.3794              
exp2_qwen                 allc veracity    +0.0015  [-0.0038,+0.0066]   0.5852       n/r    
exp2_qwen                 allc schema      -0.0142  [-0.0215,-0.0071]   0.0001           ***
exp2_qwen                 allc ate_true    -0.0128  [-0.0197,-0.0061]   0.0004           ***
exp2_qwen                 tft  veracity    -0.2534  [-0.2607,-0.2458]   0.0001       n/r ***
exp2_qwen                 tft  schema      +0.2503  [+0.2415,+0.2589]   0.0001           ***
exp2_qwen                 tft  ate_true    -0.0031  [-0.0108,+0.0047]   0.4484              
exp3_llama_abs            allc veracity    +0.0005  [-0.0078,+0.0085]   0.8896   0.00605    
exp3_llama_abs            allc schema      -0.0025  [-0.0119,+0.0071]   0.5998              
exp3_llama_abs            allc ate_true    -0.0019  [-0.0111,+0.0072]   0.6838              
exp3_llama_abs            tft  veracity    +0.0204  [+0.0116,+0.0289]   0.0001   0.01687 ***
exp3_llama_abs            tft  schema      -0.2057  [-0.2157,-0.1960]   0.0001           ***
exp3_llama_abs            tft  ate_true    -0.1855  [-0.1945,-0.1764]   0.0001           ***
exp3_llama_sem            allc veracity    -0.0131  [-0.0199,-0.0062]   0.0004   0.01050 ***
exp3_llama_sem            allc schema      -0.0000  [-0.0081,+0.0080]   0.9954              
exp3_llama_sem            allc ate_true    -0.0130  [-0.0209,-0.0051]   0.0008           ***
exp3_llama_sem            tft  veracity    -0.0255  [-0.0337,-0.0172]   0.0001   0.00582 ***
exp3_llama_sem            tft  schema      +0.0044  [-0.0055,+0.0142]   0.3838              
exp3_llama_sem            tft  ate_true    -0.0211  [-0.0307,-0.0116]   0.0001           ***
exp3_llama_swap           allc veracity    +0.0231  [+0.0101,+0.0365]   0.0002   0.05589 ***
exp3_llama_swap           allc schema      +0.0467  [+0.0345,+0.0590]   0.0001           ***
exp3_llama_swap           allc ate_true    +0.0699  [+0.0574,+0.0825]   0.0001           ***
exp3_llama_swap           tft  veracity    +0.0212  [+0.0034,+0.0392]   0.0196   0.11576 *  
exp3_llama_swap           tft  schema      +0.0132  [-0.0055,+0.0318]   0.1606              
exp3_llama_swap           tft  ate_true    +0.0343  [+0.0162,+0.0528]   0.0004           ***
exp3_mistral_sem          allc veracity    -0.0001  [-0.0002,+0.0001]   0.6020   0.00000    
exp3_mistral_sem          allc schema      -0.0002  [-0.0005,-0.0000]   0.0478           *  
exp3_mistral_sem          allc ate_true    -0.0003  [-0.0005,-0.0001]   0.0066           ** 
exp3_mistral_sem          tft  veracity    -0.0000  [-0.0002,+0.0002]   0.8888   0.00000    
exp3_mistral_sem          tft  schema      +0.0001  [-0.0000,+0.0003]   0.1070              
exp3_mistral_sem          tft  ate_true    +0.0001  [-0.0000,+0.0003]   0.2042              
exp3_mistral_swap         allc veracity    -0.0002  [-0.0031,+0.0028]   0.9050   0.00000    
exp3_mistral_swap         allc schema      +0.1422  [+0.1380,+0.1464]   0.0001           ***
exp3_mistral_swap         allc ate_true    +0.1420  [+0.1377,+0.1463]   0.0001           ***
exp3_mistral_swap         tft  veracity    -0.0010  [-0.0019,-0.0002]   0.0226   0.00000 *  
exp3_mistral_swap         tft  schema      +0.0126  [+0.0113,+0.0139]   0.0001           ***
exp3_mistral_swap         tft  ate_true    +0.0116  [+0.0103,+0.0128]   0.0001           ***
exp3_qwen_abs             allc veracity    +0.0185  [+0.0143,+0.0228]   0.0001   0.00003 ***
exp3_qwen_abs             allc schema      +0.7468  [+0.7405,+0.7527]   0.0001           ***
exp3_qwen_abs             allc ate_true    +0.7654  [+0.7590,+0.7714]   0.0001           ***
exp3_qwen_abs             tft  veracity    +0.0219  [+0.0148,+0.0291]   0.0001   0.05197 ***
exp3_qwen_abs             tft  schema      +0.7360  [+0.7273,+0.7446]   0.0001           ***
exp3_qwen_abs             tft  ate_true    +0.7578  [+0.7499,+0.7653]   0.0001           ***
exp3_qwen_sem             allc veracity    +0.0059  [+0.0011,+0.0108]   0.0160   0.00686 *  
exp3_qwen_sem             allc schema      -0.0022  [-0.0085,+0.0038]   0.4840              
exp3_qwen_sem             allc ate_true    +0.0037  [-0.0021,+0.0095]   0.2056              
exp3_qwen_sem             tft  veracity    -0.2500  [-0.2572,-0.2430]   0.0001   0.00061 ***
exp3_qwen_sem             tft  schema      +0.2669  [+0.2593,+0.2744]   0.0001           ***
exp3_qwen_sem             tft  ate_true    +0.0169  [+0.0100,+0.0235]   0.0001           ***
exp3_qwen_swap            allc veracity    +0.0018  [-0.0073,+0.0109]   0.6982   0.00533    
exp3_qwen_swap            allc schema      +0.4782  [+0.4704,+0.4860]   0.0001           ***
exp3_qwen_swap            allc ate_true    +0.4800  [+0.4723,+0.4876]   0.0001           ***
exp3_qwen_swap            tft  veracity    +0.0415  [+0.0302,+0.0529]   0.0001   0.14463 ***
exp3_qwen_swap            tft  schema      -0.1109  [-0.1218,-0.1005]   0.0001           ***
exp3_qwen_swap            tft  ate_true    -0.0693  [-0.0793,-0.0592]   0.0001           ***
exp6_llama_sem_logit      allc veracity    -0.0140  [-0.0234,-0.0047]   0.0036   0.00574 ** 
exp6_llama_sem_logit      allc schema      +0.0022  [-0.0089,+0.0129]   0.6886              
exp6_llama_sem_logit      allc ate_true    -0.0119  [-0.0236,-0.0007]   0.0372           *  
exp6_llama_sem_logit      tft  veracity    -0.0337  [-0.0464,-0.0206]   0.0001   0.00526 ***
exp6_llama_sem_logit      tft  schema      +0.0138  [-0.0001,+0.0275]   0.0522              
exp6_llama_sem_logit      tft  ate_true    -0.0200  [-0.0337,-0.0065]   0.0036           ** 
exp6_mistral_sem_logit    allc veracity    +0.0000  [-0.0002,+0.0002]   1.0000   0.00000    
exp6_mistral_sem_logit    allc schema      -0.0003  [-0.0006,+0.0000]   0.0670              
exp6_mistral_sem_logit    allc ate_true    -0.0003  [-0.0005,+0.0000]   0.0666              
exp6_mistral_sem_logit    tft  veracity    -0.0003  [-0.0006,+0.0001]   0.1630   0.00000    
exp6_mistral_sem_logit    tft  schema      -0.0003  [-0.0008,+0.0003]   0.3588              
exp6_mistral_sem_logit    tft  ate_true    -0.0005  [-0.0010,-0.0001]   0.0198           *  
exp6_qwen_sem_logit       allc veracity    +0.0128  [+0.0066,+0.0189]   0.0001   0.00156 ***
exp6_qwen_sem_logit       allc schema      +0.0054  [-0.0022,+0.0128]   0.1606              
exp6_qwen_sem_logit       allc ate_true    +0.0183  [+0.0111,+0.0256]   0.0001           ***
exp6_qwen_sem_logit       tft  veracity    -0.2017  [-0.2108,-0.1923]   0.0001   0.00058 ***
exp6_qwen_sem_logit       tft  schema      +0.2374  [+0.2281,+0.2466]   0.0001           ***
exp6_qwen_sem_logit       tft  ate_true    +0.0357  [+0.0275,+0.0438]   0.0001           ***
```

Across 26 valid cells: mean |veracity| = **0.0382**, mean |schema| = **0.1272** (3.3x).

Restricted to the retaliator, where the donor can corrupt the move field: mean |veracity| = **0.0684**, mean |schema| = **0.1435** (2.1x) over 13 cells.

Neither ratio orders the two channels. Against the unconditional cooperator the dose on the move field is zero by construction, so the veracity term there is measured at partial dose against a schema term at full dose. The `dose` column is P(|donor - true| >= 15) over non-degenerate arm-3c rows; `n/r` marks a database with no donor score column.


==============================================================================
## 2. The interaction — the actual pre-registered prediction
==============================================================================

The prediction is opponent-conditional: defection **down** vs TFT and **up** vs ALLC. That is a single quantity.

`interaction = ATE_true(allc) - ATE_true(tft)`, predicted **positive**.

Two same-signed effects are a main effect of defection, not the predicted conditional pattern, however large they are.

```
database                        ATE(allc)   ATE(tft)  interaction              95% CI        p  
--------------------------------------------------------------------------------------------------
exp2_llama                        -0.0113    -0.0046      -0.0067  [-0.0201,+0.0068]   0.3276    
exp2_qwen                         -0.0128    -0.0030      -0.0097  [-0.0201,+0.0005]   0.0626    
exp3_llama_abs                    -0.0019    -0.1854      +0.1835  [+0.1707,+0.1963]   0.0001 ***
exp3_llama_sem                    -0.0130    -0.0211      +0.0081  [-0.0043,+0.0202]   0.2002    
exp3_llama_swap                   +0.0699    +0.0344      +0.0356  [+0.0132,+0.0576]   0.0020 ** 
exp3_mistral_abs                  +0.0082    -0.2385      +0.2468  [+0.2404,+0.2533]   0.0001 ***  VOID
exp3_mistral_sem                  -0.0003    +0.0001      -0.0004  [-0.0007,-0.0001]   0.0016 ** 
exp3_mistral_swap                 +0.1420    +0.0116      +0.1304  [+0.1259,+0.1350]   0.0001 ***
exp3_qwen_abs                     +0.7654    +0.7578      +0.0076  [-0.0024,+0.0176]   0.1248    
exp3_qwen_sem                     +0.0037    +0.0169      -0.0132  [-0.0220,-0.0044]   0.0026 ** 
exp3_qwen_swap                    +0.4799    -0.0694      +0.5493  [+0.5369,+0.5618]   0.0001 ***
exp4_llama_abs_logit              -0.0022    -0.1986      +0.1963  [+0.1781,+0.2145]   0.0001 ***
exp4_llama_abs_scratchpad         +0.1238    +0.0125      +0.1112  [+0.0931,+0.1292]   0.0001 ***
exp4_llama_sem_logit              -0.0140    -0.0229      +0.0090  [-0.0091,+0.0273]   0.3242    
exp4_llama_sem_scratchpad         -0.0272    -0.0601      +0.0331  [+0.0088,+0.0570]   0.0072 ** 
exp4_mistral_abs_logit            +0.0043    -0.2306      +0.2349  [+0.2257,+0.2441]   0.0001 ***  VOID
exp4_mistral_abs_scratchpad       +0.1471    +0.0438      +0.1032  [+0.0887,+0.1175]   0.0001 ***  VOID
exp4_mistral_sem_logit            -0.0002    -0.0002      +0.0001  [-0.0004,+0.0005]   0.8586    
exp4_mistral_sem_scratchpad       +0.0697    +0.0317      +0.0379  [+0.0155,+0.0605]   0.0002 ***
exp4_qwen_abs_logit               +0.7570    +0.7537      +0.0034  [-0.0127,+0.0197]   0.6724    
exp4_qwen_abs_scratchpad          -0.0036    -0.0266      +0.0230  [+0.0090,+0.0371]   0.0012 ** 
exp4_qwen_sem_logit               +0.0095    +0.0214      -0.0118  [-0.0233,-0.0003]   0.0424 *  
exp4_qwen_sem_scratchpad          -0.2211    -0.1039      -0.1172  [-0.1406,-0.0936]   0.0001 ***
exp5_llama_sem_minimal            +0.0731    +0.0215      +0.0515  [+0.0320,+0.0705]   0.0001 ***
exp5_mistral_sem_minimal          +0.0285    +0.0357      -0.0070  [-0.0211,+0.0070]   0.3252    
exp5_qwen_sem_minimal             +0.0488    +0.0198      +0.0289  [+0.0016,+0.0563]   0.0360 *  
exp6_llama_sem_logit              -0.0118    -0.0199      +0.0080  [-0.0098,+0.0261]   0.3814    
exp6_llama_sem_scratchpad         +0.0620    +0.0124      +0.0496  [+0.0314,+0.0678]   0.0001 ***
exp6_mistral_sem_logit            -0.0003    -0.0005      +0.0003  [-0.0003,+0.0008]   0.3574    
exp6_mistral_sem_scratchpad       +0.0234    +0.0319      -0.0085  [-0.0224,+0.0053]   0.2288    
exp6_qwen_sem_logit               +0.0183    +0.0357      -0.0174  [-0.0286,-0.0064]   0.0024 ** 
exp6_qwen_sem_scratchpad          +0.0790    +0.0504      +0.0287  [-0.0009,+0.0577]   0.0562    
exp7_llama_abs_logit              -0.0004    -0.1970      +0.1968  [+0.1786,+0.2144]   0.0001 ***
exp7_llama_absnohist_logit        +0.3257    +0.2101      +0.1156  [+0.1027,+0.1283]   0.0001 ***
exp7_llama_nohist_logit           -0.0397    -0.0369      -0.0028  [-0.0115,+0.0063]   0.5462    
exp7_llama_swap_logit             +0.0582    +0.0256      +0.0326  [+0.0009,+0.0646]   0.0454 *  
exp7_mistral_nohist_logit         +0.0128    +0.0147      -0.0020  [-0.0048,+0.0009]   0.1790    
exp7_qwen_abs_logit               +0.7727    +0.7629      +0.0096  [-0.0020,+0.0215]   0.1092    
exp7_qwen_absnohist_logit         +0.8510    +0.3933      +0.4577  [+0.4507,+0.4645]   0.0001 ***
exp7_qwen_nohist_logit            -0.0013    -0.0017      +0.0004  [-0.0008,+0.0015]   0.5486    
exp7_qwen_swap_logit              +0.4879    -0.0834      +0.5713  [+0.5534,+0.5891]   0.0001 ***
exp8_llama_anchor_logit           -0.0122    -0.0135      +0.0015  [-0.0156,+0.0188]   0.8694    
exp8_llama_origpermp2_logit       -0.1304    -0.1420      +0.0116  [-0.0078,+0.0314]   0.2466    
exp8_llama_rewordp2_logit         -0.1729    -0.1945      +0.0214  [+0.0018,+0.0413]   0.0342 *  
exp8_llama_rewordpermp1_logit     -0.0716    -0.0992      +0.0276  [+0.0095,+0.0463]   0.0034 ** 
exp8_mistral_anchor_logit         -0.0002    -0.0001      -0.0001  [-0.0004,+0.0003]   0.8266    
exp8_mistral_origpermp2_logit     -0.0001    +0.0006      -0.0007  [-0.0029,+0.0015]   0.5080    
exp8_mistral_rewordp2_logit       -0.0018    -0.0013      -0.0006  [-0.0018,+0.0007]   0.3888    
exp8_mistral_rewordpermp1_logit    +0.0010    +0.0018      -0.0008  [-0.0021,+0.0003]   0.1772    
exp8_qwen_anchor_logit            +0.0229    +0.0293      -0.0065  [-0.0177,+0.0048]   0.2574    
exp8_qwen_origp2_logit            +0.0067    +0.0009      +0.0058  [+0.0009,+0.0109]   0.0214 *  
exp8_qwen_origpermp1_logit        -0.0194    -0.0128      -0.0066  [-0.0168,+0.0037]   0.2066    
exp8_qwen_origpermp2_logit        -0.0029    -0.0002      -0.0028  [-0.0068,+0.0011]   0.1544    
exp8_qwen_rewordp1_logit          +0.0538    +0.0695      -0.0156  [-0.0235,-0.0078]   0.0001 ***
exp8_qwen_rewordp2_logit          -0.0000    -0.0018      +0.0019  [-0.0023,+0.0061]   0.3846    
exp8_qwen_rewordpermp1_logit      +0.0155    +0.0173      -0.0018  [-0.0078,+0.0041]   0.5564    
exp8_qwen_rewordpermp2_logit      -0.0009    -0.0021      +0.0011  [-0.0031,+0.0053]   0.6276    
sweep                             +0.0435    +0.0550      -0.0117  [-0.0218,-0.0017]   0.0210 *  
```

Read the sign, not the magnitude. A significantly **positive** interaction is the only result that supports the registration. A significantly negative one contradicts it. Zero is a null.
