python3 train.py \
  --output_dir ./rsc/output/ \
  --log_dir ./fine_tune_runs/ \
  --bert_model_config bert-base-cased \
  --model_type baseline \
  --train_file ./rsc/features/kp20k.feature.train.256.32.hdf5 \
  --valid_file ./rsc/features/kp20k.feature.valid.256.32.hdf5 \
  --valid_per_step 1000 \
  --train_batch_size 64 \
  --learning_rate 5e-5 \
  --num_train_epochs 4.0 \
  --seed 42
