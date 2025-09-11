python tools/inference/torch_inf_multi_gpu_batch_optimized.py \
    -c configs/dfine/custom/dfine_hgnetv2_l_custom.yml \
    -r /DATA/jhlee/D-FINE/output/dfine_hgnetv2_l_pseudo_aug_9k_new/best_stg1.pth \
    -i /DATA/jhlee/synthetic_visdrone/VisDrone2019-DET-test-dev/images \
    -o /DATA/jhlee/synthetic_visdrone_test_pseudo_v2 \
    --batch-size 16 \
    --num-gpus 4