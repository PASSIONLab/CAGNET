declare -a procarr=(1 2 4)
# declare -a bsarr=(128 256 512)
declare -a bsarr=(512)
declare -a snarr=(10 20 25)
# declare -a mbarr=(64 128 256)
declare -a mbarr=(64 128)
declare -a dataarr=("reddit" "ogbn-products")

for bs in "${bsarr[@]}"
do
    for sn in "${snarr[@]}"
    do
        for mb in "${mbarr[@]}"
        do
            for data in "${dataarr[@]}"
            do
                for proc in "${procarr[@]}"
                do
                    echo $bs $sn $mb $data $proc
                    srun -l -n $proc --ntasks-per-node 4 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 python gcn_15d.py --dataset $data --batch-size $bs --samp-num $sn --n-bulkmb $mb --replication 1 --sample-method sage --timing --baseline &> cagnet_outputs/$data/cagnet_$data\_bs$bs\_sn$sn\_mb$mb\_p$proc\_timing.out
                    srun -l -n $proc --ntasks-per-node 4 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 python gcn_15d.py --dataset $data --batch-size $bs --samp-num $sn --n-bulkmb $mb --replication 1 --sample-method sage --baseline &> cagnet_outputs/$data/cagnet_$data\_bs$bs\_sn$sn\_mb$mb\_p$proc\.out
                done

                srun -l -n 4 --ntasks-per-node 4 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 python gcn_15d.py --dataset $data --batch-size $bs --samp-num $sn --n-bulkmb $mb --replication 2 --sample-method sage --timing --baseline &> cagnet_outputs/$data/cagnet_$data\_bs$bs\_sn$sn\_mb$mb\_p4c2_timing.out
                srun -l -n 4 --ntasks-per-node 4 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 python gcn_15d.py --dataset $data --batch-size $bs --samp-num $sn --n-bulkmb $mb --replication 2 --sample-method sage --baseline &> cagnet_outputs/$data/cagnet_$data\_bs$bs\_sn$sn\_mb$mb\_p4c2.out
            done
        done
    done
done
