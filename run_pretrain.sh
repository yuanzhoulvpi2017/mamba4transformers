# deepspeed --include localhost:0 
    # --deepspeed ds_zero2_no_offload.json \
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc_per_node=1 run_pretrain.py \
    --data_dir data/data4pretrain \
    --cache_local_dir cache_data2 \
    --tokenizer_path internlm_tokenizer \
    --bf16 true \
    --fp16 false \
    --output_dir modeloutput \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate 4e-4 \
    --logging_steps 10 \
    --max_seq_length 1024