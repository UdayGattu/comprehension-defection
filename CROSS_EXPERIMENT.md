# Cross-experiment tables

Built by `analysis/07_cross_experiment.py` from `EVIDENCE_cells.csv`, which `06_evidence.py` produced from SQL. No number here was recalled or recomputed — each is a subtraction between two cells in that file.

Defect rates are **episode-level means** (`defect_ep`). A contrast is marked `VOID` when either end exceeds the off-task gate of 0.1; the value is still printed so an excluded result stays visible.

Bootstrap intervals are not recomputed here. The `ep_*.json` file carrying each row's interval is named beside every table.


## 0. How each database was classified

Every table below is built from this parse. If a row is wrong, the tables built from it are wrong — which is why it is printed first.

```
  database                     exp   model    framing  readout     CoT prompt  intervals                          
  ---------------------------  ----  -------  -------  ----------  ----------  -----------------------------------
  exp2_llama                   exp2  llama    sem      logit       n/a         ep_exp2_llama.json                 
  exp2_llama_labelswap         exp2  llama    swap     logit       n/a         none                               
  exp2_qwen                    exp2  qwen     sem      logit       n/a         ep_exp2_qwen.json                  
  exp2_qwen_labelswap          exp2  qwen     swap     logit       n/a         none                               
  exp3_llama_abs               exp3  llama    abs      logit       n/a         ep_exp3_llama_abs.json             
  exp3_llama_sem               exp3  llama    sem      logit       n/a         ep_exp3_llama_sem.json             
  exp3_llama_swap              exp3  llama    swap     logit       n/a         ep_exp3_llama_swap.json            
  exp3_mistral_abs             exp3  mistral  abs      logit       n/a         ep_exp3_mistral_abs.json           
  exp3_mistral_sem             exp3  mistral  sem      logit       n/a         ep_exp3_mistral_sem.json           
  exp3_mistral_swap            exp3  mistral  swap     logit       n/a         ep_exp3_mistral_swap.json          
  exp3_qwen_abs                exp3  qwen     abs      logit       n/a         ep_exp3_qwen_abs.json              
  exp3_qwen_sem                exp3  qwen     sem      logit       n/a         ep_exp3_qwen_sem.json              
  exp3_qwen_swap               exp3  qwen     swap     logit       n/a         ep_exp3_qwen_swap.json             
  exp4_llama_abs_logit         exp4  llama    abs      logit       n/a         ep_exp4_llama_abs_logit.json       
  exp4_llama_abs_scratchpad    exp4  llama    abs      scratchpad  guided      ep_exp4_llama_abs_scratchpad.json  
  exp4_llama_sem_logit         exp4  llama    sem      logit       n/a         ep_exp4_llama_sem_logit.json       
  exp4_llama_sem_scratchpad    exp4  llama    sem      scratchpad  guided      ep_exp4_llama_sem_scratchpad.json  
  exp4_mistral_abs_logit       exp4  mistral  abs      logit       n/a         ep_exp4_mistral_abs_logit.json     
  exp4_mistral_abs_scratchpad  exp4  mistral  abs      scratchpad  guided      ep_exp4_mistral_abs_scratchpad.json
  exp4_mistral_sem_logit       exp4  mistral  sem      logit       n/a         ep_exp4_mistral_sem_logit.json     
  exp4_mistral_sem_scratchpad  exp4  mistral  sem      scratchpad  guided      ep_exp4_mistral_sem_scratchpad.json
  exp4_qwen_abs_logit          exp4  qwen     abs      logit       n/a         ep_exp4_qwen_abs_logit.json        
  exp4_qwen_abs_scratchpad     exp4  qwen     abs      scratchpad  guided      ep_exp4_qwen_abs_scratchpad.json   
  exp4_qwen_sem_logit          exp4  qwen     sem      logit       n/a         ep_exp4_qwen_sem_logit.json        
  exp4_qwen_sem_scratchpad     exp4  qwen     sem      scratchpad  guided      ep_exp4_qwen_sem_scratchpad.json   
  exp5_llama_sem_minimal       exp5  llama    sem      scratchpad  minimal     ep_exp5_llama_sem_minimal.json     
  exp5_mistral_sem_minimal     exp5  mistral  sem      scratchpad  minimal     ep_exp5_mistral_sem_minimal.json   
  exp5_qwen_sem_minimal        exp5  qwen     sem      scratchpad  minimal     ep_exp5_qwen_sem_minimal.json      
  sweep                        exp1  llama    sem      logit       n/a         none                               

```

## 1. Readout ladder — semantic framing

`P(D)` per arm across three ways of eliciting the same decision from the same weights on the same game.

**The LOGIT vs CoT comparison is confounded within exp4**: its scratchpad instruction names the finite horizon. The `CoT minimal` column is the control — its instruction is *"Before choosing, think step by step."* and names nothing. Read the two CoT columns together or not at all.


### llama

```
  arm     opp   LOGIT (exp4)  CoT guided (exp4)  CoT minimal (exp5)
  ------  ----  ------------  -----------------  ------------------
  arm 1   allc  0.3236        0.5572             0.7307            
  arm 1   tft   0.3367        0.6500             0.7361            
  arm 3b  allc  0.1017        0.5787             0.6386            
  arm 3b  tft   0.1298        0.6661             0.6613            
  arm 3   allc  0.0872        0.5546             0.7125            
  arm 3   tft   0.1069        0.6120             0.6851            

```
intervals: `ep_exp4_llama_sem_logit.json`, `ep_exp4_llama_sem_scratchpad.json`, `ep_exp5_llama_sem_minimal.json`


### mistral

```
  arm     opp   LOGIT (exp4)  CoT guided (exp4)  CoT minimal (exp5)
  ------  ----  ------------  -----------------  ------------------
  arm 1   allc  0.0003        0.4396             0.4274            
  arm 1   tft   0.0002        0.4117             0.3807            
  arm 3b  allc  0.0003        0.3347             0.4062            
  arm 3b  tft   0.0003        0.2952             0.3771            
  arm 3   allc  0.0002        0.4021             0.4356            
  arm 3   tft   0.0002        0.3267             0.4131            

```
intervals: `ep_exp4_mistral_sem_logit.json`, `ep_exp4_mistral_sem_scratchpad.json`, `ep_exp5_mistral_sem_minimal.json`


### qwen

```
  arm     opp   LOGIT (exp4)  CoT guided (exp4)  CoT minimal (exp5)
  ------  ----  ------------  -----------------  ------------------
  arm 1   allc  0.0089        0.4036             0.6499            
  arm 1   tft   0.0088        0.4947             0.7572            
  arm 3b  allc  0.0420        0.5970             0.5974            
  arm 3b  tft   0.0450        0.6990             0.7347            
  arm 3   allc  0.0510        0.3835             0.6411            
  arm 3   tft   0.0653        0.5976             0.7518            

```
intervals: `ep_exp4_qwen_sem_logit.json`, `ep_exp4_qwen_sem_scratchpad.json`, `ep_exp5_qwen_sem_minimal.json`


## 2. Contrasts across the readout ladder

`perturbation = P(D|3b) - P(D|1)` — does inserting any block matter?

`ATE_true = P(D|3) - P(D|3b)` — holding the block constant, does its content matter?


### perturbation = P(D|3b) − P(D|1)

```
  model    opp   LOGIT    CoT guided  CoT minimal
  -------  ----  -------  ----------  -----------
  llama    allc  -0.2218  +0.0215     -0.0920    
  llama    tft   -0.2069  +0.0161     -0.0747    
  mistral  allc  +0.0000  -0.1049     -0.0212    
  mistral  tft   +0.0002  -0.1165     -0.0036    
  qwen     allc  +0.0331  +0.1934     -0.0525    
  qwen     tft   +0.0362  +0.2043     -0.0226    

```

### ATE_true = P(D|3) − P(D|3b)

```
  model    opp   LOGIT    CoT guided  CoT minimal
  -------  ----  -------  ----------  -----------
  llama    allc  -0.0145  -0.0242     +0.0738    
  llama    tft   -0.0228  -0.0541     +0.0237    
  mistral  allc  -0.0002  +0.0674     +0.0294    
  mistral  tft   -0.0002  +0.0315     +0.0360    
  qwen     allc  +0.0090  -0.2135     +0.0437    
  qwen     tft   +0.0203  -0.1015     +0.0171    

```

### ATE_naive = P(D|3) − P(D|1)

```
  model    opp   LOGIT    CoT guided  CoT minimal
  -------  ----  -------  ----------  -----------
  llama    allc  -0.2364  -0.0026     -0.0182    
  llama    tft   -0.2298  -0.0379     -0.0510    
  mistral  allc  -0.0002  -0.0375     +0.0081    
  mistral  tft   +0.0000  -0.0850     +0.0324    
  qwen     allc  +0.0421  -0.0202     -0.0088    
  qwen     tft   +0.0565  +0.1029     -0.0054    

```

## 3. Stack drift — exp3 vs exp4, identical condition

Same model, framing, readout, arms and N-per-cell; different vLLM, torch, transformers and driver versions. Any difference is the inference stack, not the treatment.

Run `06_evidence.py` section 3 for the exact version strings of each.


### perturbation

```
  model    framing  opp   exp3                    exp4 LOGIT              drift  
  -------  -------  ----  ----------------------  ----------------------  -------
  llama    sem      allc  -0.1806                 -0.2218                 -0.0412
  llama    sem      tft   -0.1921                 -0.2069                 -0.0148
  llama    abs      allc  +0.0067                 +0.0110                 +0.0043
  llama    abs      tft   +0.0292                 +0.0430                 +0.0138
  mistral  sem      allc  +0.0002                 +0.0000                 -0.0002
  mistral  sem      tft   +0.0000                 +0.0002                 +0.0002
  mistral  abs      allc  +0.0171 VOID(off 0.83)  +0.0208 VOID(off 0.83)  +0.0037
  mistral  abs      tft   +0.0624 VOID(off 0.98)  +0.0538 VOID(off 0.97)  -0.0086
  qwen     sem      allc  +0.0386                 +0.0331                 -0.0055
  qwen     sem      tft   +0.0381                 +0.0362                 -0.0019
  qwen     abs      allc  -0.0549                 -0.0378                 +0.0171
  qwen     abs      tft   -0.0647                 -0.0458                 +0.0188

```

### ATE_true

```
  model    framing  opp   exp3                    exp4 LOGIT              drift  
  -------  -------  ----  ----------------------  ----------------------  -------
  llama    sem      allc  -0.0135                 -0.0145                 -0.0010
  llama    sem      tft   -0.0207                 -0.0228                 -0.0022
  llama    abs      allc  +0.0028                 +0.0003                 -0.0024
  llama    abs      tft   -0.1728                 -0.1848                 -0.0120
  mistral  sem      allc  -0.0003                 -0.0002                 +0.0001
  mistral  sem      tft   +0.0001                 -0.0002                 -0.0003
  mistral  abs      allc  +0.0078 VOID(off 1.00)  +0.0041 VOID(off 1.00)  -0.0037
  mistral  abs      tft   -0.2266 VOID(off 1.00)  -0.2191 VOID(off 1.00)  +0.0075
  qwen     sem      allc  +0.0035                 +0.0090                 +0.0055
  qwen     sem      tft   +0.0160                 +0.0203                 +0.0043
  qwen     abs      allc  +0.7451                 +0.7409                 -0.0042
  qwen     abs      tft   +0.7375                 +0.7373                 -0.0002

```

## 4. Lexical falsification — framing, exp3

Identical blocks, identical positions, identical token parity. The only difference is whether the action labels carry meaning: `sem` = Cooperate/Defect, `swap` = the same words with their meanings inverted, `abs` = X/Y.

Pre-specified test: if the container effect is about the word "Cooperate", abstract labels should shrink it. If it is unchanged under X/Y, the lexical account is wrong.


### perturbation

```
  model    opp   semantic  swap     abstract              
  -------  ----  --------  -------  ----------------------
  llama    allc  -0.1806   +0.1223  +0.0067               
  llama    tft   -0.1921   +0.2422  +0.0292               
  mistral  allc  +0.0002   +0.0898  +0.0171 VOID(off 0.83)
  mistral  tft   +0.0000   -0.0064  +0.0624 VOID(off 0.98)
  qwen     allc  +0.0386   +0.1269  -0.0549               
  qwen     tft   +0.0381   +0.8928  -0.0647               

```

### ATE_true

```
  model    opp   semantic  swap     abstract              
  -------  ----  --------  -------  ----------------------
  llama    allc  -0.0135   +0.0670  +0.0028               
  llama    tft   -0.0207   +0.0332  -0.1728               
  mistral  allc  -0.0003   +0.1349  +0.0078 VOID(off 1.00)
  mistral  tft   +0.0001   +0.0110  -0.2266 VOID(off 1.00)
  qwen     allc  +0.0035   +0.4559  +0.7451               
  qwen     tft   +0.0160   -0.0659  +0.7375               

```

### baseline P(D|1)

```
  model    opp   semantic  swap    abstract   
  -------  ----  --------  ------  -----------
  llama    allc  0.2792    0.2117  0.7067     
  llama    tft   0.3126    0.4969  0.7423     
  mistral  allc  0.0001    0.6911  0.8215 VOID
  mistral  tft   0.0001    0.9923  0.7698 VOID
  qwen     allc  0.0090    0.1433  0.0808     
  qwen     tft   0.0143    0.0705  0.0975     

```

## 5. Manipulation check — CPR by arm

Pre-registered gate: CPR(arm 3) >= 0.85. exp1 failed it; every run after passed. CPR takes no partial credit — all three probes must be correct on a probed turn.

```
  database                     opp   CPR arm 1  CPR arm 3b  CPR arm 3  gate
  ---------------------------  ----  ---------  ----------  ---------  ----
  exp2_llama                   allc  0.200      0.200       1.000      PASS
  exp2_llama                   tft   0.200      0.200       1.000      PASS
  exp2_qwen                    allc  0.400      0.400       1.000      PASS
  exp2_qwen                    tft   0.400      0.400       1.000      PASS
  exp3_llama_abs               allc  0.249      0.200       1.000      PASS
  exp3_llama_abs               tft   0.245      0.200       1.000      PASS
  exp3_llama_sem               allc  0.200      0.200       1.000      PASS
  exp3_llama_sem               tft   0.200      0.200       1.000      PASS
  exp3_llama_swap              allc  0.200      0.200       0.200      FAIL
  exp3_llama_swap              tft   0.200      0.200       0.200      FAIL
  exp3_mistral_abs             allc  0.206      0.000       1.000      PASS
  exp3_mistral_abs             tft   0.239      0.001       0.996      PASS
  exp3_mistral_sem             allc  0.400      0.400       1.000      PASS
  exp3_mistral_sem             tft   0.400      0.400       1.000      PASS
  exp3_mistral_swap            allc  0.200      0.200       0.200      FAIL
  exp3_mistral_swap            tft   0.200      0.200       0.200      FAIL
  exp3_qwen_abs                allc  0.396      0.398       1.000      PASS
  exp3_qwen_abs                tft   0.395      0.397       1.000      PASS
  exp3_qwen_sem                allc  0.400      0.400       1.000      PASS
  exp3_qwen_sem                tft   0.400      0.400       1.000      PASS
  exp3_qwen_swap               allc  0.200      0.200       0.200      FAIL
  exp3_qwen_swap               tft   0.200      0.200       0.200      FAIL
  exp4_llama_abs_logit         allc  0.244      0.200       1.000      PASS
  exp4_llama_abs_logit         tft   0.246      0.200       0.999      PASS
  exp4_llama_abs_scratchpad    allc  0.200      0.200       1.000      PASS
  exp4_llama_abs_scratchpad    tft   0.200      0.200       1.000      PASS
  exp4_llama_sem_logit         allc  0.200      0.200       1.000      PASS
  exp4_llama_sem_logit         tft   0.200      0.200       1.000      PASS
  exp4_llama_sem_scratchpad    allc  0.200      0.200       1.000      PASS
  exp4_llama_sem_scratchpad    tft   0.200      0.200       1.000      PASS
  exp4_mistral_abs_logit       allc  0.205      0.000       1.000      PASS
  exp4_mistral_abs_logit       tft   0.246      0.000       0.996      PASS
  exp4_mistral_abs_scratchpad  allc  0.000      0.200       1.000      PASS
  exp4_mistral_abs_scratchpad  tft   0.197      0.200       1.000      PASS
  exp4_mistral_sem_logit       allc  0.400      0.400       1.000      PASS
  exp4_mistral_sem_logit       tft   0.400      0.400       1.000      PASS
  exp4_mistral_sem_scratchpad  allc  0.200      0.000       1.000      PASS
  exp4_mistral_sem_scratchpad  tft   0.000      0.000       1.000      PASS
  exp4_qwen_abs_logit          allc  0.397      0.396       1.000      PASS
  exp4_qwen_abs_logit          tft   0.397      0.395       1.000      PASS
  exp4_qwen_abs_scratchpad     allc  0.210      0.209       1.000      PASS
  exp4_qwen_abs_scratchpad     tft   0.211      0.211       1.000      PASS
  exp4_qwen_sem_logit          allc  0.400      0.400       1.000      PASS
  exp4_qwen_sem_logit          tft   0.400      0.400       1.000      PASS
  exp4_qwen_sem_scratchpad     allc  0.329      0.325       1.000      PASS
  exp4_qwen_sem_scratchpad     tft   0.327      0.349       1.000      PASS
  exp5_llama_sem_minimal       allc  0.200      0.200       1.000      PASS
  exp5_llama_sem_minimal       tft   0.200      0.200       1.000      PASS
  exp5_mistral_sem_minimal     allc  0.200      0.200       0.997      PASS
  exp5_mistral_sem_minimal     tft   0.200      0.200       0.998      PASS
  exp5_qwen_sem_minimal        allc  0.250      0.268       1.000      PASS
  exp5_qwen_sem_minimal        tft   0.385      0.381       1.000      PASS
  sweep                        allc  0.200      0.000       0.307      FAIL
  sweep                        tft   0.200      0.000       0.244      FAIL

```

## 6. Regret against solved optimal play

`episode_regret` is payoff lost against the optimal policy for that opponent, computed by `cdx/optimal.py` — not assumed. Collected on every run and, before this table, never reported.

More interpretable than a defection rate: *defects 58% of the time* is hard to weigh, *loses N points of the available total* is not.

```
  database                     arm  opp   mean regret  n     P(action=optimal)      
  ---------------------------  ---  ----  -----------  ----  -----------------  ----
  exp2_llama                   1    allc  27.73        1600  0.307                  
  exp2_llama                   1    tft   10.32        1600  0.677                  
  exp2_llama                   3b   allc  36.35        1600  0.091                  
  exp2_llama                   3b   tft   4.37         1600  0.858                  
  exp2_llama                   3    allc  36.81        1600  0.080                  
  exp2_llama                   3    tft   4.22         1600  0.861                  
  exp2_llama_labelswap         1    allc  31.41        1600  0.215                  
  exp2_llama_labelswap         1    tft   18.03        1600  0.473                  
  exp2_llama_labelswap         3b   allc  26.27        1600  0.343                  
  exp2_llama_labelswap         3b   tft   27.01        1600  0.272                  
  exp2_qwen                    1    allc  39.65        1600  0.009                  
  exp2_qwen                    1    tft   2.17         1600  0.945                  
  exp2_qwen                    3b   allc  37.54        1600  0.061                  
  exp2_qwen                    3b   tft   2.28         1600  0.941                  
  exp2_qwen                    3    allc  38.03        1600  0.049                  
  exp2_qwen                    3    tft   2.56         1600  0.920                  
  exp2_qwen_labelswap          1    allc  34.68        1600  0.133                  
  exp2_qwen_labelswap          1    tft   3.33         1600  0.887                  
  exp2_qwen_labelswap          3b   allc  29.00        1600  0.275                  
  exp2_qwen_labelswap          3b   tft   36.84        1600  0.073                  
  exp3_llama_abs               1    allc  11.73        2000  0.707                  
  exp3_llama_abs               1    tft   26.35        2000  0.285                  
  exp3_llama_abs               3b   allc  11.46        2000  0.713                  
  exp3_llama_abs               3b   tft   27.65        2000  0.256                  
  exp3_llama_abs               3    allc  11.35        2000  0.716                  
  exp3_llama_abs               3    tft   19.33        2000  0.410                  
  exp3_llama_sem               1    allc  28.83        2000  0.279                  
  exp3_llama_sem               1    tft   10.02        2000  0.685                  
  exp3_llama_sem               3b   allc  36.06        2000  0.099                  
  exp3_llama_sem               3b   tft   4.59         2000  0.850                  
  exp3_llama_sem               3    allc  36.60        2000  0.085                  
  exp3_llama_sem               3    tft   4.04         2000  0.866                  
  exp3_llama_swap              1    allc  31.53        2000  0.212                  
  exp3_llama_swap              1    tft   17.36        2000  0.491                  
  exp3_llama_swap              3b   allc  26.64        2000  0.334                  
  exp3_llama_swap              3b   tft   26.70        2000  0.278                  
  exp3_llama_swap              3    allc  23.96        2000  0.401                  
  exp3_llama_swap              3    tft   27.92        2000  0.250                  
  exp3_mistral_abs             1    allc  7.14         2000  0.822              VOID
  exp3_mistral_abs             1    tft   27.41        2000  0.279              VOID
  exp3_mistral_abs             3b   allc  6.46         2000  0.839              VOID
  exp3_mistral_abs             3b   tft   30.89        2000  0.218              VOID
  exp3_mistral_abs             3    allc  6.14         2000  0.846              VOID
  exp3_mistral_abs             3    tft   18.62        2000  0.431              VOID
  exp3_mistral_sem             1    allc  39.99        2000  0.000                  
  exp3_mistral_sem             1    tft   2.00         2000  0.950                  
  exp3_mistral_sem             3b   allc  39.99        2000  0.000                  
  exp3_mistral_sem             3b   tft   2.00         2000  0.950                  
  exp3_mistral_sem             3    allc  40.00        2000  0.000                  
  exp3_mistral_sem             3    tft   2.00         2000  0.950                  
  exp3_mistral_swap            1    allc  12.36        2000  0.691                  
  exp3_mistral_swap            1    tft   37.54        2000  0.058                  
  exp3_mistral_swap            3b   allc  8.77         2000  0.781                  
  exp3_mistral_swap            3b   tft   37.16        2000  0.064                  
  exp3_mistral_swap            3    allc  3.37         2000  0.916                  
  exp3_mistral_swap            3    tft   37.81        2000  0.053                  
  exp3_qwen_abs                1    allc  36.77        2000  0.081                  
  exp3_qwen_abs                1    tft   4.63         2000  0.883                  
  exp3_qwen_abs                3b   allc  38.96        2000  0.026                  
  exp3_qwen_abs                3b   tft   2.87         2000  0.928                  
  exp3_qwen_abs                3    allc  9.16         2000  0.771                  
  exp3_qwen_abs                3    tft   28.06        2000  0.277                  
  exp3_qwen_sem                1    allc  39.64        2000  0.009                  
  exp3_qwen_sem                1    tft   2.29         2000  0.941                  
  exp3_qwen_sem                3b   allc  38.10        2000  0.048                  
  exp3_qwen_sem                3b   tft   2.34         2000  0.940                  
  exp3_qwen_sem                3    allc  37.95        2000  0.051                  
  exp3_qwen_sem                3    tft   2.79         2000  0.914                  
  exp3_qwen_swap               1    allc  34.27        2000  0.143                  
  exp3_qwen_swap               1    tft   3.42         2000  0.884                  
  exp3_qwen_swap               3b   allc  29.19        2000  0.270                  
  exp3_qwen_swap               3b   tft   36.62        2000  0.078                  
  exp3_qwen_swap               3    allc  10.96        2000  0.726                  
  exp3_qwen_swap               3    tft   33.07        2000  0.150                  
  exp4_llama_abs_logit         1    allc  11.95        1000  0.701                  
  exp4_llama_abs_logit         1    tft   25.75        1000  0.297                  
  exp4_llama_abs_logit         3b   allc  11.51        1000  0.712                  
  exp4_llama_abs_logit         3b   tft   27.82        1000  0.252                  
  exp4_llama_abs_logit         3    allc  11.49        1000  0.713                  
  exp4_llama_abs_logit         3    tft   18.98        1000  0.417                  
  exp4_llama_abs_scratchpad    1    allc  19.02        1000  0.524                  
  exp4_llama_abs_scratchpad    1    tft   16.61        1000  0.489                  
  exp4_llama_abs_scratchpad    3b   allc  18.10        1000  0.548                  
  exp4_llama_abs_scratchpad    3b   tft   18.57        1000  0.444                  
  exp4_llama_abs_scratchpad    3    allc  13.38        1000  0.665                  
  exp4_llama_abs_scratchpad    3    tft   18.37        1000  0.441                  
  exp4_llama_sem_logit         1    allc  27.06        1000  0.324                  
  exp4_llama_sem_logit         1    tft   10.65        1000  0.666                  
  exp4_llama_sem_logit         3b   allc  35.93        1000  0.102                  
  exp4_llama_sem_logit         3b   tft   4.89         1000  0.842                  
  exp4_llama_sem_logit         3    allc  36.51        1000  0.087                  
  exp4_llama_sem_logit         3    tft   4.26         1000  0.859                  
  exp4_llama_sem_scratchpad    1    allc  17.71        1000  0.557                  
  exp4_llama_sem_scratchpad    1    tft   21.99        1000  0.376                  
  exp4_llama_sem_scratchpad    3b   allc  16.85        1000  0.579                  
  exp4_llama_sem_scratchpad    3b   tft   22.81        1000  0.365                  
  exp4_llama_sem_scratchpad    3    allc  17.82        1000  0.555                  
  exp4_llama_sem_scratchpad    3    tft   20.21        1000  0.408                  
  exp4_mistral_abs_logit       1    allc  7.06         1000  0.824              VOID
  exp4_mistral_abs_logit       1    tft   27.97        1000  0.268              VOID
  exp4_mistral_abs_logit       3b   allc  6.23         1000  0.844              VOID
  exp4_mistral_abs_logit       3b   tft   31.02        1000  0.216              VOID
  exp4_mistral_abs_logit       3    allc  6.06         1000  0.848              VOID
  exp4_mistral_abs_logit       3    tft   19.05        1000  0.421              VOID
  exp4_mistral_abs_scratchpad  1    allc  19.35        1000  0.516              VOID
  exp4_mistral_abs_scratchpad  1    tft   13.16        1000  0.567              VOID
  exp4_mistral_abs_scratchpad  3b   allc  17.52        1000  0.562              VOID
  exp4_mistral_abs_scratchpad  3b   tft   14.88        1000  0.524              VOID
  exp4_mistral_abs_scratchpad  3    allc  12.07        1000  0.698              VOID
  exp4_mistral_abs_scratchpad  3    tft   16.27        1000  0.482              VOID
  exp4_mistral_sem_logit       1    allc  39.99        1000  0.000                  
  exp4_mistral_sem_logit       1    tft   2.00         1000  0.950                  
  exp4_mistral_sem_logit       3b   allc  39.99        1000  0.000                  
  exp4_mistral_sem_logit       3b   tft   2.01         1000  0.950                  
  exp4_mistral_sem_logit       3    allc  39.99        1000  0.000                  
  exp4_mistral_sem_logit       3    tft   2.00         1000  0.950                  
  exp4_mistral_sem_scratchpad  1    allc  22.41        1000  0.440                  
  exp4_mistral_sem_scratchpad  1    tft   12.91        1000  0.589                  
  exp4_mistral_sem_scratchpad  3b   allc  26.61        1000  0.335                  
  exp4_mistral_sem_scratchpad  3b   tft   9.51         1000  0.703                  
  exp4_mistral_sem_scratchpad  3    allc  23.92        1000  0.402                  
  exp4_mistral_sem_scratchpad  3    tft   10.25        1000  0.661                  
  exp4_qwen_abs_logit          1    allc  37.19        1000  0.070                  
  exp4_qwen_abs_logit          1    tft   4.46         1000  0.887                  
  exp4_qwen_abs_logit          3b   allc  38.70        1000  0.033                  
  exp4_qwen_abs_logit          3b   tft   3.23         1000  0.919                  
  exp4_qwen_abs_logit          3    allc  9.07         1000  0.773                  
  exp4_qwen_abs_logit          3    tft   28.51        1000  0.265                  
  exp4_qwen_abs_scratchpad     1    allc  10.63        1000  0.734              VOID
  exp4_qwen_abs_scratchpad     1    tft   34.85        1000  0.112              VOID
  exp4_qwen_abs_scratchpad     3b   allc  6.42         1000  0.839                  
  exp4_qwen_abs_scratchpad     3b   tft   35.37        1000  0.103                  
  exp4_qwen_abs_scratchpad     3    allc  6.67         1000  0.833                  
  exp4_qwen_abs_scratchpad     3    tft   34.20        1000  0.127                  
  exp4_qwen_sem_logit          1    allc  39.64        1000  0.009                  
  exp4_qwen_sem_logit          1    tft   2.13         1000  0.946                  
  exp4_qwen_sem_logit          3b   allc  38.32        1000  0.042                  
  exp4_qwen_sem_logit          3b   tft   2.14         1000  0.945                  
  exp4_qwen_sem_logit          3    allc  37.96        1000  0.051                  
  exp4_qwen_sem_logit          3    tft   2.64         1000  0.918                  
  exp4_qwen_sem_scratchpad     1    allc  23.86        1000  0.404                  
  exp4_qwen_sem_scratchpad     1    tft   17.56        1000  0.530                  
  exp4_qwen_sem_scratchpad     3b   allc  16.12        1000  0.597                  
  exp4_qwen_sem_scratchpad     3b   tft   24.79        1000  0.338                  
  exp4_qwen_sem_scratchpad     3    allc  24.66        1000  0.383                  
  exp4_qwen_sem_scratchpad     3    tft   20.52        1000  0.441                  
  exp5_llama_sem_minimal       1    allc  10.77        1000  0.731                  
  exp5_llama_sem_minimal       1    tft   25.27        1000  0.291                  
  exp5_llama_sem_minimal       3b   allc  14.46        1000  0.639                  
  exp5_llama_sem_minimal       3b   tft   21.96        1000  0.367                  
  exp5_llama_sem_minimal       3    allc  11.50        1000  0.712                  
  exp5_llama_sem_minimal       3    tft   22.87        1000  0.341                  
  exp5_mistral_sem_minimal     1    allc  22.90        1000  0.427                  
  exp5_mistral_sem_minimal     1    tft   11.44        1000  0.615                  
  exp5_mistral_sem_minimal     3b   allc  23.75        1000  0.406                  
  exp5_mistral_sem_minimal     3b   tft   11.37        1000  0.621                  
  exp5_mistral_sem_minimal     3    allc  22.58        1000  0.436                  
  exp5_mistral_sem_minimal     3    tft   11.99        1000  0.589                  
  exp5_qwen_sem_minimal        1    allc  14.00        1000  0.650                  
  exp5_qwen_sem_minimal        1    tft   27.35        1000  0.280                  
  exp5_qwen_sem_minimal        3b   allc  16.11        1000  0.597                  
  exp5_qwen_sem_minimal        3b   tft   26.07        1000  0.304                  
  exp5_qwen_sem_minimal        3    allc  14.36        1000  0.641                  
  exp5_qwen_sem_minimal        3    tft   26.85        1000  0.286                  
  sweep                        1    allc  27.79        1600  0.305                  
  sweep                        1    tft   9.73         1600  0.692                  
  sweep                        3b   allc  38.47        1600  0.038                  
  sweep                        3b   tft   2.82         1600  0.913                  
  sweep                        3    allc  36.79        1600  0.080                  
  sweep                        3    tft   4.02         1600  0.869                  

```

## 7. Token parity across arms

The study's central methodological claim is that treatment and placebo are token-matched. This checks it per database from `scaffold_tokens`, which records the real length of every injected block.

```
  database                     arms with a block  token range  verdict
  ---------------------------  -----------------  -----------  -------
  exp2_llama                   3,3b,3c,3d         34..34       MATCHED
  exp2_llama_labelswap         3b                 34..34       MATCHED
  exp2_qwen                    3,3b,3c,3d         39..39       MATCHED
  exp2_qwen_labelswap          3b                 39..39       MATCHED
  exp3_llama_abs               3,3b,3c,3d         34..34       MATCHED
  exp3_llama_sem               3,3b,3c,3d         34..34       MATCHED
  exp3_llama_swap              3,3b,3c,3d         34..34       MATCHED
  exp3_mistral_abs             3,3b,3c,3d         45..45       MATCHED
  exp3_mistral_sem             3,3b,3c,3d         45..45       MATCHED
  exp3_mistral_swap            3,3b,3c,3d         45..45       MATCHED
  exp3_qwen_abs                3,3b,3c,3d         39..39       MATCHED
  exp3_qwen_sem                3,3b,3c,3d         39..39       MATCHED
  exp3_qwen_swap               3,3b,3c,3d         39..39       MATCHED
  exp4_llama_abs_logit         3,3b               34..34       MATCHED
  exp4_llama_abs_scratchpad    3,3b               34..34       MATCHED
  exp4_llama_sem_logit         3,3b               34..34       MATCHED
  exp4_llama_sem_scratchpad    3,3b               34..34       MATCHED
  exp4_mistral_abs_logit       3,3b               45..45       MATCHED
  exp4_mistral_abs_scratchpad  3,3b               45..45       MATCHED
  exp4_mistral_sem_logit       3,3b               45..45       MATCHED
  exp4_mistral_sem_scratchpad  3,3b               45..45       MATCHED
  exp4_qwen_abs_logit          3,3b               39..39       MATCHED
  exp4_qwen_abs_scratchpad     3,3b               39..39       MATCHED
  exp4_qwen_sem_logit          3,3b               39..39       MATCHED
  exp4_qwen_sem_scratchpad     3,3b               39..39       MATCHED
  exp5_llama_sem_minimal       3,3b               34..34       MATCHED
  exp5_mistral_sem_minimal     3,3b               45..45       MATCHED
  exp5_qwen_sem_minimal        3,3b               39..39       MATCHED
  sweep                        3,3b               31..32       SPREAD 

```

A `SPREAD` means the block was not a constant token length. exp1 used character padding (`Your score: 003`), which fixes character width but not token count under BPE. Later runs enforce parity on token IDs.
