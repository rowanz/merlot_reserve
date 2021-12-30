# VCR finetuning

Use `prep_data.sh` after downloading the data.

Finetune the model with

```bash
export lr=1e-5
export ne=5

python qa_qar_joint_finetune.py ../../pretrain/configs/base.yaml ${path_to_ckpt} -lr=${lr} -ne=${ne} -output_grid_h=18 -output_grid_w=32
```

Then submit it to the leaderboard with something like

```bash
ipython -i submit_to_leaderboard.py -- ../../pretrain/configs/base.yaml path_to_ckpt_33240 -output_grid_h=18 -output_grid_w=32
```