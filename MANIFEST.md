# Archive manifest

Generated 2026-08-16 by counting `SELECT COUNT(*) FROM turns`
and `FROM episodes` over every committed archive. Not derived from the
experiment drivers, and not copied from any prose figure in this repository.

**81 archives, 1.59 GB compressed, 12,700,960 turns, 635,048 episodes.**

## Totals by experiment

| group | archives | turns | episodes |
|---|---:|---:|---:|
| `cotsmoke` | 12 | 5,760 | 288 |
| `exp2` | 4 | 896,000 | 44,800 |
| `exp3` | 9 | 3,600,000 | 180,000 |
| `exp4` | 12 | 1,440,000 | 72,000 |
| `exp5` | 3 | 360,000 | 18,000 |
| `exp6` | 6 | 1,200,000 | 60,000 |
| `exp7` | 9 | 1,800,000 | 90,000 |
| `exp8` | 16 | 3,200,000 | 160,000 |
| `smoke` | 9 | 7,200 | 360 |
| `sweep` | 1 | 192,000 | 9,600 |
| **all** | **81** | **12,700,960** | **635,048** |

## Per-file

| file | bytes | turns | episodes | sha256 |
|---|---:|---:|---:|---|
| `cotsmoke_llama_abs_logit.sqlite.gz` | 72,202 | 480 | 24 | `32b8a25bb046f8032f5d2220b95cab56f643b0f64fc7ee37c9ae4ea52baa6042` |
| `cotsmoke_llama_abs_scratchpad.sqlite.gz` | 170,484 | 480 | 24 | `d2bc74ac3ed24f9ef118d07ab4776a3e83063a53c9db5e9a88d26a7296b5d001` |
| `cotsmoke_llama_sem_logit.sqlite.gz` | 69,441 | 480 | 24 | `9159a76ee00d00febee966d704ae488b1d0a890f47099660e9238b0578816cdd` |
| `cotsmoke_llama_sem_scratchpad.sqlite.gz` | 170,474 | 480 | 24 | `d85db0ac9272f674ab62766cb02a63c2b1e331ad8889b05d51fa3ac1d1bff742` |
| `cotsmoke_mistral_abs_logit.sqlite.gz` | 74,754 | 480 | 24 | `4efec0cfaf65d8110acd6e805e332d40e4771231ecb9b9390cb2ef74fb5e90c4` |
| `cotsmoke_mistral_abs_scratchpad.sqlite.gz` | 161,177 | 480 | 24 | `1b977163faba432825420cdf352c58cef25abc7da59dbe0d4d6ba18dfb0d14be` |
| `cotsmoke_mistral_sem_logit.sqlite.gz` | 66,260 | 480 | 24 | `d277dc04ad225cf83e08a9f187568c6dbc6003e3a7e0387417e059ce44e83758` |
| `cotsmoke_mistral_sem_scratchpad.sqlite.gz` | 169,121 | 480 | 24 | `bf875188873b1071ea71207e2457ffcd1877b4e79934336cae0db987aeb3ccd8` |
| `cotsmoke_qwen_abs_logit.sqlite.gz` | 55,774 | 480 | 24 | `f15ecaf974bfb3ec9198468cd1e31860b724979c9dcb850f578412305d511904` |
| `cotsmoke_qwen_abs_scratchpad.sqlite.gz` | 151,229 | 480 | 24 | `76f2b9dbe35244f1a639f93147ba803840a7a03ad6e63bc8dda9b054c8960114` |
| `cotsmoke_qwen_sem_logit.sqlite.gz` | 60,393 | 480 | 24 | `a40f2a1e356362b485a289f2d2420a44c8ae155e026ad1c34092408b98bc4d8f` |
| `cotsmoke_qwen_sem_scratchpad.sqlite.gz` | 153,446 | 480 | 24 | `34de43a5796bfcd53191bd98d38d3397f4045a447ea1313d162de82311df0c35` |
| `exp2_llama.sqlite.gz` | 34,819,236 | 320,000 | 16,000 | `47a15f7b8f4e1f0735718ca0c4a4f7536dc1525e526a8904e0af2b16de013c6d` |
| `exp2_llama_labelswap.sqlite.gz` | 15,003,580 | 128,000 | 6,400 | `389f13c9685cfedab56a3228fdf42435f0cbe5ca9895cfc4550f17e755067ca4` |
| `exp2_qwen.sqlite.gz` | 27,241,037 | 320,000 | 16,000 | `73bfa36a34626fef6a1a538e3e7729ac177f6df41a8aab73c89b41699b6b79e6` |
| `exp2_qwen_labelswap.sqlite.gz` | 11,902,531 | 128,000 | 6,400 | `3e6fa88dbbbdfdef63fcb5ce71d2ca4104fccb7f7fa57512d79bbc3aaacefee0` |
| `exp3_llama_abs.sqlite.gz` | 50,119,976 | 400,000 | 20,000 | `0d5f102253cbcb16a68e859a4bf183729880234aecb517b76b55725039f6d1dd` |
| `exp3_llama_sem.sqlite.gz` | 44,144,467 | 400,000 | 20,000 | `ac1da0dbd0735c7591e0ea5db7dd811770fb52700908d2545bdaa068e04755b8` |
| `exp3_llama_swap.sqlite.gz` | 46,943,492 | 400,000 | 20,000 | `7e0c9b3739d5ffe09a0c08a7e37271f8149d8501684ebbb1cc2629157ac3f80c` |
| `exp3_mistral_abs.sqlite.gz` | 48,125,806 | 400,000 | 20,000 | `0cad530f8a7df8c291efa749893734509a1b9ed7170b0c35331cf682202ca9f0` |
| `exp3_mistral_sem.sqlite.gz` | 33,628,045 | 400,000 | 20,000 | `22d3dd4e6802dbed74bf92d29d06380e5c651c86ad24f468c26b3ba0f5a2566a` |
| `exp3_mistral_swap.sqlite.gz` | 45,378,567 | 400,000 | 20,000 | `2e8fc1f24db9f211cb8afc99a418b49d646c9abd2d74707598cb757a1e9cb921` |
| `exp3_qwen_abs.sqlite.gz` | 31,681,530 | 400,000 | 20,000 | `ce3e66657f836a64ff50d4e5dfd471a5f31b6ded2b90fc9bb000c9c4e04dc0dc` |
| `exp3_qwen_sem.sqlite.gz` | 35,173,961 | 400,000 | 20,000 | `321ecf26c8ec3447fb9486f1ab779c1c479406f1dd6fc79168f8166fa6e2501b` |
| `exp3_qwen_swap.sqlite.gz` | 39,695,439 | 400,000 | 20,000 | `da5ab2ba5adb4c4e78fb62972ff3e9a7fecab9366d08fc9e40104c9fda46eb9c` |
| `exp4_llama_abs_logit.sqlite.gz` | 14,930,579 | 120,000 | 6,000 | `690e184b690e92467a45e86d3ae12a458c8e263f90459c3c797eb21628e6f1ef` |
| `exp4_llama_abs_scratchpad.sqlite.gz` | 37,905,565 | 120,000 | 6,000 | `6b6d1450344c92cf6d3b16fd82b90c712e22464a27e4ab041b24fde06d4cc262` |
| `exp4_llama_sem_logit.sqlite.gz` | 13,124,246 | 120,000 | 6,000 | `487f54abbd4514f661a191e44d04361e20e1c8b7730085ff0cbbcac4e831bead` |
| `exp4_llama_sem_scratchpad.sqlite.gz` | 38,125,465 | 120,000 | 6,000 | `b0f248af8806d54ef640276c69263b89ab9a55a2240ee61a6646a465c14f5a36` |
| `exp4_mistral_abs_logit.sqlite.gz` | 14,709,019 | 120,000 | 6,000 | `4a000bb50cd08ca879fc51c25f1538563d4c3236fb225f74ec0daf3f4e201929` |
| `exp4_mistral_abs_scratchpad.sqlite.gz` | 35,656,882 | 120,000 | 6,000 | `9585efaf61c5f010c344c27a9787027f1fbb8946cde644f337e54811a9a1ea4b` |
| `exp4_mistral_sem_logit.sqlite.gz` | 10,300,211 | 120,000 | 6,000 | `f9540f762bde60495c0fd9d0575e37c964322196c2b2acf30ffd552acb4ae148` |
| `exp4_mistral_sem_scratchpad.sqlite.gz` | 37,428,676 | 120,000 | 6,000 | `6ee6d351aa5fb1dbef82d8f7f53d1999ceae5dd865b390350b1e933460f14630` |
| `exp4_qwen_abs_logit.sqlite.gz` | 9,911,720 | 120,000 | 6,000 | `84a8081e22d5d70a3d05f523153e73748d0260cb244c282917e366fe23ba1c9f` |
| `exp4_qwen_abs_scratchpad.sqlite.gz` | 32,560,944 | 120,000 | 6,000 | `055508421f4e972cc0e15684c835113b343e83d177c9dfa437926d213ca64cd4` |
| `exp4_qwen_sem_logit.sqlite.gz` | 10,127,560 | 120,000 | 6,000 | `55f3bd02ac6d127730fc065b9971b96a8d891ec12c394f8dec7dd2a04257b262` |
| `exp4_qwen_sem_scratchpad.sqlite.gz` | 33,604,815 | 120,000 | 6,000 | `d1dcf28d1fdb86fa3e48885b1321828df3ad425d38f2cbc1a027bdb49c38cc38` |
| `exp5_llama_sem_minimal.sqlite.gz` | 36,061,358 | 120,000 | 6,000 | `88037672070d33d2b1db2afd603866057d0cfd9b34d9615665c138149bbe71f9` |
| `exp5_mistral_sem_minimal.sqlite.gz` | 37,072,872 | 120,000 | 6,000 | `3b1da7fd34ad2665919a83f0b9d96b1d4e2555a30f63c213bd1efbc5d39cc612` |
| `exp5_qwen_sem_minimal.sqlite.gz` | 36,073,673 | 120,000 | 6,000 | `f719cbb6982d26b9bbca052f6670f1dcc183d2981901d23dca7fdce3ac3b3a62` |
| `exp6_llama_sem_logit.sqlite.gz` | 27,054,842 | 240,000 | 12,000 | `41a1e278b110ced3694d6b768f2d6033bb85d802c81eaf4c5c323f5474f0212d` |
| `exp6_llama_sem_scratchpad.sqlite.gz` | 48,801,683 | 160,000 | 8,000 | `43cb1c45bc7808046205b0db42bf7a6320ff95e1d767612761e71608744322db` |
| `exp6_mistral_sem_logit.sqlite.gz` | 22,577,387 | 240,000 | 12,000 | `94303512433a9bfb1925b2f76db1b9d4eede135225fb7b1769169e1738720ab6` |
| `exp6_mistral_sem_scratchpad.sqlite.gz` | 49,962,671 | 160,000 | 8,000 | `22c321cea15c9f51198edb5ba2b7d3ab44f6178c6a032c7350387f9c4e0923cf` |
| `exp6_qwen_sem_logit.sqlite.gz` | 22,810,260 | 240,000 | 12,000 | `9f6e577671d865fc44e989ef5f09a971f1fcf154f74bdc5ef95f79dd62ad7c36` |
| `exp6_qwen_sem_scratchpad.sqlite.gz` | 48,305,102 | 160,000 | 8,000 | `9cb9f99d6365ec5f4d67c8e526bf47523b7f0488f4134c3e85ebf3895053be8f` |
| `exp7_llama_abs_logit.sqlite.gz` | 25,227,396 | 200,000 | 10,000 | `b7cc8928352222b552bbf540ec1708ce22f6011e5ce9cf0f58a44f464fc409c7` |
| `exp7_llama_absnohist_logit.sqlite.gz` | 17,675,896 | 200,000 | 10,000 | `15fbf9ff8d68e137f332c6ffcb6a74e341283e7c0a0374096ac84b33db3fef71` |
| `exp7_llama_nohist_logit.sqlite.gz` | 16,409,342 | 200,000 | 10,000 | `3ecff2fcfe7ae2165d1a63a8f7cffad6e8d91acf689faefd2fecb76b5da73b68` |
| `exp7_llama_swap_logit.sqlite.gz` | 23,646,455 | 200,000 | 10,000 | `0b2e0af11d8cffe32da411d8198e681373f3ef1e0be718b425033695673c38b5` |
| `exp7_mistral_nohist_logit.sqlite.gz` | 17,166,046 | 200,000 | 10,000 | `87b6b55cf82c5664a0f92ea1f87b2466bfaa294e4032b14c47b5c43af66eb9b3` |
| `exp7_qwen_abs_logit.sqlite.gz` | 17,990,872 | 200,000 | 10,000 | `3516ddf3a2905c08d6f03ed54321551d8dc4921ea7fe56af081ca7636f6a8aec` |
| `exp7_qwen_absnohist_logit.sqlite.gz` | 13,105,406 | 200,000 | 10,000 | `ff729af66ca4e6b9470d1a57a7c7f51eb7b070de660fdfa55e56c33b29720ac5` |
| `exp7_qwen_nohist_logit.sqlite.gz` | 13,394,659 | 200,000 | 10,000 | `8ce7ee9f5209d0ea61ad0b73a919a53f9282559278342eec769d91f0e88485dd` |
| `exp7_qwen_swap_logit.sqlite.gz` | 20,509,403 | 200,000 | 10,000 | `0ab33a73337712669ee9799773e825ade61c0e76eb8ba2be1f5aa851443f9dfa` |
| `exp8_llama_anchor_logit.sqlite.gz` | 22,379,529 | 200,000 | 10,000 | `9447d41ae5b22f55dcb8ced22c9ffd1230781ae571c7ca52d4c80d2a453a1e5a` |
| `exp8_llama_origpermp2_logit.sqlite.gz` | 23,284,800 | 200,000 | 10,000 | `abc5fa778c7261a6ca203a1ab758e6bac9acc138f9f15f5ef59834d695783f00` |
| `exp8_llama_rewordp2_logit.sqlite.gz` | 22,761,572 | 200,000 | 10,000 | `0d0560b5531c71b16ea1a8a54085f90a182aa2e86e10d90ca45e195267c248de` |
| `exp8_llama_rewordpermp1_logit.sqlite.gz` | 22,852,444 | 200,000 | 10,000 | `83760e1c28b4fe82ba17703bdfd7bee9fe6f7b3df61894dcc8b6b54177107b9c` |
| `exp8_mistral_anchor_logit.sqlite.gz` | 18,740,874 | 200,000 | 10,000 | `f2cefc3b0f87eff3fc9dbfb653803022351dc3935a478febcb289ac17d02a1c3` |
| `exp8_mistral_origpermp2_logit.sqlite.gz` | 20,868,154 | 200,000 | 10,000 | `1683856c15fd545973ffdc1d8b4c4ac6bddcddf14a97ad58ca25d137ca9e4168` |
| `exp8_mistral_rewordp2_logit.sqlite.gz` | 20,508,411 | 200,000 | 10,000 | `261c08fdcaf52ea797f9fff6670eda7c930eca5160a418ff3cf02bb96ad5b4ff` |
| `exp8_mistral_rewordpermp1_logit.sqlite.gz` | 20,529,941 | 200,000 | 10,000 | `c03768c41b549c72b41b387e3b87935e109382d15fcb17ab116baf089e6a5c4c` |
| `exp8_qwen_anchor_logit.sqlite.gz` | 18,561,868 | 200,000 | 10,000 | `02a9b45b909151dcde85f43c55f49beb69f3f4931054d10273d6fe4a880be421` |
| `exp8_qwen_origp2_logit.sqlite.gz` | 18,295,865 | 200,000 | 10,000 | `0598252e97de83f16f15218ec459ecefd768841743681875e89cb4b1b9d1d550` |
| `exp8_qwen_origpermp1_logit.sqlite.gz` | 18,306,683 | 200,000 | 10,000 | `92170662f6aa1f304e982c08a07d98fac783f5dc3b678600f173391405d0e807` |
| `exp8_qwen_origpermp2_logit.sqlite.gz` | 18,645,711 | 200,000 | 10,000 | `437b6cfe93183feb17655e1683863d94c5168e848e6447776c5d7990f3e566cd` |
| `exp8_qwen_rewordp1_logit.sqlite.gz` | 18,391,338 | 200,000 | 10,000 | `f08e9eb72dfe56cb5991c10a8e77c2ed68dc97285fa196e8dd058c9b9d717699` |
| `exp8_qwen_rewordp2_logit.sqlite.gz` | 18,200,717 | 200,000 | 10,000 | `213ac365e61dc7ec389b7af567c483f0e17275e7f9187e4a833a3c2e3ce76633` |
| `exp8_qwen_rewordpermp1_logit.sqlite.gz` | 18,578,995 | 200,000 | 10,000 | `754a8308176f5daf3465a3a61dfba033708ed1ad6918d1b70129627051800a55` |
| `exp8_qwen_rewordpermp2_logit.sqlite.gz` | 18,843,937 | 200,000 | 10,000 | `82d33cec8b897377c024c7771f643d436fde8427ca91b778c44c0e8c0ebf8f7e` |
| `smoke_llama_abs.sqlite.gz` | 119,893 | 800 | 40 | `fd534a213a297402a972e042c210d68b1d06be2a8801c5ba962df31c5212711e` |
| `smoke_llama_sem.sqlite.gz` | 113,011 | 800 | 40 | `ea8938871b89914bb6ea34d79724b5a855bdbf888a13f948d6167bc9125c8189` |
| `smoke_llama_swap.sqlite.gz` | 116,638 | 800 | 40 | `788c6558b03c6228722c783d8a9d5de840b8dae31e3866ced9c984d474b34d0e` |
| `smoke_mistral_abs.sqlite.gz` | 121,392 | 800 | 40 | `c9855affd6db04b1181e4221cb0516852ffe7d22d4fa39f18669205100f0ab7c` |
| `smoke_mistral_sem.sqlite.gz` | 108,028 | 800 | 40 | `3bfac44ca4443051f9c84056daa1cf04a03365bed1f9fddcea1431896bbdb104` |
| `smoke_mistral_swap.sqlite.gz` | 122,677 | 800 | 40 | `b4ae19726511bebe5027c6a10eaff539a70c378e76df733a31b677054279503f` |
| `smoke_qwen_abs.sqlite.gz` | 91,315 | 800 | 40 | `38b03544a48f20a070a14e866918705c4ce2a23b0569622175829a8700ff672c` |
| `smoke_qwen_sem.sqlite.gz` | 99,532 | 800 | 40 | `2a6a4e0186215e01f110b217700d3d2d796afa38d3df63ba6b1ff6d868556465` |
| `smoke_qwen_swap.sqlite.gz` | 104,967 | 800 | 40 | `ebb866fa5c7bb8003a9160fe3cda8f8c34dea016888cb055ec1b0d585b3507a8` |
| `sweep.sqlite.gz` | 19,577,362 | 192,000 | 9,600 | `31d957520791da8c7335f1732ff9011b1398c503b0bada17c6ff59cb6ebc55b0` |

Verify with:

```sh
sha256sum -c <(awk -F'|' '/^\| `.*sqlite.gz`/{gsub(/[` ]/,"",$2);gsub(/[` ]/,"",$6);print $6"  "$2}' MANIFEST.md)
```
