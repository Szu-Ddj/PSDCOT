export CUDA_VISIBLE_DEVICES=7

data_targets=("hc-dt")
lrs=("1e-3")
bert_lrs=("1e-5")
model_names=("MPLN")
seeds=("0" "1" "2" "3")
dropouts=("0.2")
lambdaas=(0.1)
hops=(3)
lids=(0)
for lr in ${lrs[*]}
do
    for bert_lr in ${bert_lrs[*]}
    do
        for model_name in ${model_names[*]}
        do
            for data_target in ${data_targets[*]}
            do
                for seed in ${seeds[*]}
                do
                    for dropout in ${dropouts[*]}
                    do
                        for lid in ${lids[*]}
                        do
                            for hop in ${hops[*]}
                            do
                                for lambdaa in ${lambdaas[*]}
                                do

                                            python3 ../codes/train.py \
                                            --lr $lr \
                                            --model_name $model_name \
                                            --data_target $data_target \
                                            --seed $seed \
                                            --dropout $dropout \
                                            --hop $hop \
                                            --lambdaa $lambdaa \
                                            --lid $lid \

                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

# nohup bash script/PStance_bernie_run.sh > logs/_PStance_bernie.out &
