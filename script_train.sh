DATA=tox21
TDATA=tox21
setting=pre_par
NS=10
NQ=16
pretrain=1
gpu=0
seed=0


nohup python -u main.py --epochs 1000 --eval_steps 10 --pretrained $pretrain \
--setting $setting --n-shot-train $NS  --n-shot-test $NS --n-query $NQ --dataset $DATA --test-dataset $TDATA --seed $seed --cuda $gpu\
> nohup_${DATA}${TDATA}-${setting}_s${NS}q${NQ} 2>&1 &
