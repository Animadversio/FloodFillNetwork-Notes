python compute_partitions.py \
    --input_volume third_party/LGN_DATA/groundtruth.h5:stack \
    --output_volume third_party/LGN_DATA/af.h5:af \
    --thresholds 0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
    --lom_radius 24,24,24 \
    --min_size 10000

python compute_partitions.py \
    --input_volume third_party/LGN_DATA/groundtruth_zyx.h5:stack \
    --output_volume third_party/LGN_DATA/af_zyx.h5:af \
    --thresholds 0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
    --lom_radius 24,24,24 \
    --min_size 10000


 python build_coordinates.py \
     --partition_volumes LGN_hr:third_party/LGN_DATA/af_zyx.h5:af \
     --coordinate_output third_party/LGN_DATA/tf_record_file \
     --margin 24,24,24

resampling and shuffling takes 30 mins

 python train.py \
    --train_coords third_party/LGN_DATA/tf_record_file \
    --data_volumes LGN_hr:third_party/LGN_DATA/grayscale_maps_zyx.h5:raw \
    --label_volumes LGN_hr:third_party/LGN_DATA/groundtruth_zyx.h5:stack \
    --model_name convstack_3d.ConvStack3DFFNModel \
    --model_args "{\"depth\": 12, \"fov_size\": [33, 33, 33], \"deltas\": [8, 8, 8]}" \
    --image_mean 128 \
    --image_stddev 33 \
    --train_dir /tmp/big_model1 

 python train.py \
    --train_coords third_party/LGN_DATA/tf_record_file \
    --data_volumes LGN_hr:third_party/LGN_DATA/grayscale_maps_zyx.h5:raw \
    --label_volumes LGN_hr:third_party/LGN_DATA/groundtruth_zyx.h5:stack \
    --train_dir /tmp/big_model3 \
    --model_name convstack_3d.ConvStack3DFFNModel \
    --model_args "{\"depth\": 12, \"fov_size\": [49, 49, 49], \"deltas\": [0, 0, 0]}" \
    --image_mean 128 \
    --image_stddev 33

python run_inference.py \
    --inference_request="$(cat configs/inference_training_LGN.pbtxt)" \
    --bounding_box 'start { x:0 y:0 z:0 } size { x:1000 y:1000 z:175 }'

inference_training_LGN.pbtxt

image {
  hdf5: "third_party/LGN_DATA/grayscale_maps_zyx.h5:raw"
}
image_mean: 128
image_stddev: 33
checkpoint_interval: 1800
seed_policy: "PolicyPeaks"
model_checkpoint_path: "/tmp/model.ckpt-5848"
model_name: "convstack_3d.ConvStack3DFFNModel"
model_args: "{\"depth\": 12, \"fov_size\": [33, 33, 33], \"deltas\": [8, 8, 8]}"
segmentation_output_dir: "results/LGN/testing"
inference_options {
  init_activation: 0.95
  pad_value: 0.05
  move_threshold: 0.9
  min_boundary_dist { x: 1 y: 1 z: 1}
  segment_threshold: 0.6
  min_segment_size: 1000
}


20720,14308  25416,17668

python compute_partitions.py \
--input_volume third_party/LGN_DATA/groundtruth_LR.h5:stack \
​--output_volume third_party/LGN_DATA/af_LR.h5:af \
--thresholds 0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
--lom_radius 36,24,11 \
--min_size 10000

% z, y, x tuple! in build coordinates
​    
python build_coordinates.py \
--partition_volumes LGN_LR:third_party/LGN_DATA/af_LR.h5:af \
--coordinate_output third_party/LGN_DATA/tf_record_file_LR \
--margin 11,24,36



python train.py \
--train_coords third_party/LGN_DATA/tf_record_file_LR \
--data_volumes LGN_LR:third_party/LGN_DATA/grayscale_maps_LR.h5:raw \
--label_volumes LGN_LR:third_party/LGN_DATA/groundtruth_LR.h5:stack \
--train_dir /tmp/LR_model \
--model_name convstack_3d.ConvStack3DFFNModel \
--model_args "{\"depth\": 9, \"fov_size\": [55, 37, 17], \"deltas\": [9,6,3]}" \
--image_mean 128 \
--image_stddev 33


python compute_partitions.py --input_volume third_party/LGN_DATA/groundtruth_LR.h5:stack ​--output_volume third_party/LGN_DATA/af_LR.h5:af --thresholds 0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 --lom_radius 36,24,11 --min_size 10000

​    
python build_coordinates.py --partition_volumes LGN_LR:third_party/LGN_DATA/af_LR.h5:af --coordinate_output third_party/LGN_DATA/tf_record_file_LR --margin 35,24,11


python train.py --train_coords third_party/LGN_DATA/tf_record_file_LR --data_volumes LGN_LR:third_party/LGN_DATA/grayscale_maps_LR.h5:raw --label_volumes LGN_LR:third_party/LGN_DATA/groundtruth_LR.h5:stack --train_dir /tmp/LR_model --model_name convstack_3d.ConvStack3DFFNModel --model_args "{\"depth\": 9, \"fov_size\": [55, 37, 17], \"deltas\": [9,6,3]}" --image_mean 128 --image_stddev 33


New parallelized code 



python compute_partitions_parallel.py \
    --input_volume third_party/LGN_DATA/groundtruth_LR.h5:stack \
    --output_volume third_party/LGN_DATA/af_LR2.h5:af \
    --thresholds 0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
    --lom_radius 24,24,24 \
    --min_size 10000


### New parallelized code 

```
python compute_partitions_parallel.py     
--input_volume third_party/LGN_DATA/groundtruth_LR.h5:stack     
--output_volume third_party/LGN_DATA/af_LR2.h5:af     
--thresholds 0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9     
--lom_radius 24,24,24     
--min_size 10000

Total Processing time 1021.864 s
I0115 13:27:22.047548 139741016774464 compute_partitions_parallel.py:374] Nonzero values: 115180415
```


```

python train.py \
--train_coords third_party/LGN_DATA/tf_record_file_LR \
--data_volumes LGN_LR:third_party/LGN_DATA/grayscale_maps_LR.h5:raw \
--label_volumes LGN_LR:third_party/LGN_DATA/groundtruth_LR.h5:stack \
--train_dir /tmp/LR_model \
--model_name convstack_3d.ConvStack3DFFNModel \
--model_args "{\"depth\": 9, \"fov_size\": [55, 37, 17], \"deltas\": [9,6,3]}" \
--image_mean 128 \
--image_stddev 33 \
--permutable_axes 0
```

Bash Output
> I0115 13:53:48.034140 140419504748352 tf_logging.py:115] Saving checkpoints for 0 into /tmp/LR_model/model.ckpt.
I0115 13:55:56.953015 140419504748352 train.py:699] Saving summaries.
I0115 13:58:03.573235 140419504748352 train.py:699] Saving summaries.
INFO:tensorflow:Saving checkpoints for 41 into /tmp/LR_model/model.ckpt.
I0115 13:58:54.746328 140419504748352 tf_logging.py:115] Saving checkpoints for 41 into /tmp/LR_model/model.ckpt.
I0115 14:00:07.680537 140419504748352 train.py:699] Saving summaries.
I0115 14:02:12.235027 140419504748352 train.py:699] Saving summaries.
INFO:tensorflow:Saving checkpoints for 82 into /tmp/LR_model/model.ckpt.
I0115 14:03:57.264208 140419504748352 tf_logging.py:115] Saving checkpoints for 82 into /tmp/LR_model/model.ckpt.
I0115 14:04:12.281582 140419504748352 train.py:699] Saving summaries.
INFO:tensorflow:global_step/sec: 0.134778
I0115 14:06:10.195768 140419504748352 tf_logging.py:115] global_step/sec: 0.134778
......
I0115 21:58:49.042545 140419504748352 tf_logging.py:115] Saving checkpoints for 3968 into /tmp/LR_model/model.ckpt.
I0115 21:59:54.704546 140419504748352 train.py:699] Saving summaries.
I0115 22:01:58.458268 140419504748352 train.py:699] Saving summaries.
INFO:tensorflow:global_step/sec: 0.137051
I0115 22:02:42.492995 140419504748352 tf_logging.py:115] global_step/sec: 0.137051
INFO:tensorflow:Saving checkpoints for 4010 into /tmp/LR_model/model.ckpt.
I0115 22:03:55.568126 140419504748352 tf_logging.py:115] Saving checkpoints for 4010 into /tmp/LR_model/model.ckpt.
I0115 22:04:03.141271 140419504748352 train.py:699] Saving summaries.


python run_inference.py \
  --inference_request="$(cat configs/inference_training_LGN_LR.pbtxt)" \
  --bounding_box 'start { x:0 y:0 z:0 } size { x:1000 y:1000 z:175 }'

>python run_inference.py   --inference_request="$(cat configs/inference_training_LGN_LR.pbtxt)"   --bounding_box 'start { x:0 y:0 z:0 } size { x:1000 y:1000 z:175 }'
2019-01-15 22:52:42.926844: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
I0115 22:52:42.933228 139856426125120 inference.py:891] Available TF devices: [_DeviceAttributes(/job:localhost/replica:0/task:0/device:CPU:0, CPU, 268435456, 321787398782472866), _DeviceAttributes(/job:localhost/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 17179869184, 6405407232747171761)]
I0115 22:52:42.933463 139856426125120 import_util.py:44] Importing symbol ConvStack3DFFNModel from ffn.training.models.convstack_3d
I0115 22:52:43.669929 139856426125120 inference.py:822] Loading checkpoint.
I0115 22:52:43.671457 139856426125120 tf_logging.py:115] Restoring parameters from /tmp/LR_model/model.ckpt-3680
I0115 22:52:43.688838 139856426125120 inference.py:824] Checkpoint loaded.
I0115 22:52:43.689184 139850569213696 executor.py:177] Executor starting.
I0115 22:52:43.689719 139856426125120 inference.py:1005] Process subvolume: (0L, 0L, 0L)
I0115 22:52:43.690022 139856426125120 inference.py:1023] Requested bounds are (0L, 0L, 0L) + (175L, 1000L, 1000L)
I0115 22:52:43.690109 139856426125120 inference.py:1024] Destination bounds are (0L, 0L, 0L) + (175L, 1000L, 1000L)
I0115 22:52:43.690176 139856426125120 inference.py:1025] Fetch bounds are array([0, 0, 0]) + array([ 175, 1000, 1000])
I0115 22:52:43.818634 139856426125120 inference.py:1040] Fetched image of size (175, 1000, 1000) prior to transform
I0115 22:52:43.831613 139856426125120 inference.py:1050] Image data loaded, shape: (175, 1000, 1000).
I0115 22:52:44.352058 139856426125120 inference.py:299] Registered as client 0.
I0115 22:52:44.360819 139856426125120 seed.py:106] peaks: starting
I0115 22:52:44.393812 139850569213696 executor.py:198] client 0 starting
I0115 22:53:08.067902 139856426125120 seed.py:127] peaks: filtering done
I0115 22:53:55.842163 139856426125120 seed.py:129] peaks: edt done
I0115 22:54:06.999095 139856426125120 seed.py:145] peaks: found 258754 local maxima
......