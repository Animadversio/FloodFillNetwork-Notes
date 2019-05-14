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
```bash
python compute_partitions.py \
--input_volume third_party/LGN_DATA/groundtruth_LR.h5:stack \
​--output_volume third_party/LGN_DATA/af_LR.h5:af \
--thresholds 0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
--lom_radius 36,24,11 \
--min_size 10000

# z, y, x tuple! in build coordinates
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
```

python compute_partitions.py --input_volume third_party/LGN_DATA/groundtruth_LR.h5:stack ​--output_volume third_party/LGN_DATA/af_LR.h5:af --thresholds 0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 --lom_radius 36,24,11 --min_size 10000

​    
python build_coordinates.py --partition_volumes LGN_LR:third_party/LGN_DATA/af_LR.h5:af --coordinate_output third_party/LGN_DATA/tf_record_file_LR --margin 35,24,11


python train.py --train_coords third_party/LGN_DATA/tf_record_file_LR --data_volumes LGN_LR:third_party/LGN_DATA/grayscale_maps_LR.h5:raw --label_volumes LGN_LR:third_party/LGN_DATA/groundtruth_LR.h5:stack --train_dir /tmp/LR_model --model_name convstack_3d.ConvStack3DFFNModel --model_args "{\"depth\": 9, \"fov_size\": [55, 37, 17], \"deltas\": [9,6,3]}" --image_mean 128 --image_stddev 33


**New parallelized code**
```bash
python compute_partitions_parallel.py \
    --input_volume third_party/LGN_DATA/groundtruth_LR.h5:stack \
    --output_volume third_party/LGN_DATA/af_LR2.h5:af \
    --thresholds 0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
    --lom_radius 24,24,24 \
    --min_size 10000
```

### New parallelized code 

```bash
python compute_partitions_parallel.py     
--input_volume third_party/LGN_DATA/groundtruth_LR.h5:stack     
--output_volume third_party/LGN_DATA/af_LR2.h5:af     
--thresholds 0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9     
--lom_radius 24,24,24     
--min_size 10000
```


Total Processing time 1021.864 s
> I0115 13:27:22.047548 139741016774464 compute_partitions_parallel.py:374] Nonzero values: 115180415

```bash
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

```bash
python run_inference.py \
  --inference_request="$(cat configs/inference_training_LGN_LR.pbtxt)" \
  --bounding_box 'start { x:0 y:0 z:0 } size { x:1000 y:1000 z:175 }'
```
**Log Output**

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
I0116 11:02:51.475986 139856426125120 inference.py:554] [cl 0] Starting segmentation at (95, 833, 35) (zyx)
I0116 11:02:51.856883 139856426125120 inference.py:554] [cl 0] Failed: too small: 172
I0116 11:02:51.857259 139856426125120 inference.py:554] [cl 0] Starting segmentation at (95, 833, 874) (zyx)
I0116 11:02:52.240228 139856426125120 inference.py:554] [cl 0] Failed: too small: 55
I0116 11:02:52.240566 139856426125120 inference.py:554] [cl 0] Starting segmentation at (95, 834, 540) (zyx)
I0116 11:02:52.620095 139856426125120 inference.py:554] [cl 0] Failed: too small: 85
......

Approximately 0.4 s per seed ! estimated time 28.75 h for 258754 seeds ! 


```bash
python compute_partitions_parallel.py \
    --input_volume third_party/LGN_DATA/groundtruth_LR.h5:stack \
    --output_volume third_party/LGN_DATA/af_LR_WF.h5:af \
    --thresholds 0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
    --lom_radius 53,35,16 \
    --min_size 10000
    
python build_coordinates.py \
     --partition_volumes LGN_LR:third_party/LGN_DATA/af_LR_WF.h5:af \
     --coordinate_output third_party/LGN_DATA/tf_record_file_LR_WF \
     --margin 16,35,53

python train.py \
--train_coords third_party/LGN_DATA/tf_record_file_LR_WF \
--data_volumes LGN_LR:third_party/LGN_DATA/grayscale_maps_LR.h5:raw \
--label_volumes LGN_LR:third_party/LGN_DATA/groundtruth_LR.h5:stack \
--train_dir /tmp/LR_model_WF \
--model_name convstack_3d.ConvStack3DFFNModel \
--model_args "{\"depth\": 9, \"fov_size\": [77, 51, 23], \"deltas\": [15,10,5]}" \
--image_mean 128 \
--image_stddev 33 \
--permutable_axes 0
```

> I0117 20:13:38.941535 140457088677696 compute_partitions_parallel.py:337] Labels to process: 707
......
Label 772 processing finished. Time 1080.489 s
Total Processing time 1080.730 s
I0117 20:31:40.274130 140457088677696 compute_partitions_parallel.py:375] Nonzero values: 121202245
.....
I0117 20:38:31.459806 140381154473792 build_coordinates.py:88] Saving coordinates.
I0118 00:31:46.862736 140385005000512 import_util.py:44] Importing symbol ConvStack3DFFNModel from ffn.training.models.convstack_3d
I0118 00:31:46.867686 140385005000512 train.py:721] Random seed: 1547793106
.....
I0118 19:35:46.724340 140385005000512 train.py:699] Saving summaries.
INFO:tensorflow:Saving checkpoints for 3663 into /tmp/LR_model_WF/model.ckpt.
I0118 19:36:05.467489 140385005000512 tf_logging.py:115] Saving checkpoints for 3663 into /tmp/LR_model_WF/model.ckpt.
## Retraining with Cluster 
Run!
Finished !! 8 hours for the bigger model! 


## Inference with Cluster 
Run!!

Local Run inference! 
```bash
source ~/virtenvs/test1/bin/activate
export Request='image {
 hdf5: "/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/grayscale_maps_LR.h5:raw"
}
image_mean: 128
image_stddev: 33
checkpoint_interval: 1200
seed_policy: "PolicyPeaks"
model_checkpoint_path: "models/LR_model_WF/model.ckpt-10000"
model_name: "convstack_3d.ConvStack3DFFNModel"
model_args: "{\"depth\": 9, \"fov_size\": [77, 51, 23], \"deltas\": [15,10,5]}"
segmentation_output_dir: "results/LGN/testing_LR_WF"
inference_options {
  init_activation: 0.95
  pad_value: 0.05
  move_threshold: 0.95
  min_boundary_dist { x: 3 y: 3 z: 1}
  segment_threshold: 0.5
  min_segment_size: 1000
}'

python run_inference.py \
  --inference_request="$Request" \
  --bounding_box 'start { x:0 y:0 z:0 } size { x:1180 y:1058 z:175 }' 
```
```
python run_inference.py \
  --inference_request="$(cat configs/inference_training_LGN_LR_WF.pbtxt))" \
  --bounding_box 'start { x:0 y:0 z:0 } size { x:1180 y:1058 z:175 }'
```
Sun 15:00 PM
Mon 12:52 PM
## Inspect the inference result

Too many merger site !! not sure if consensus can rescue this 

## Resegmentation Points finding 

Parallelize, 19:05 start 24 processses
Pairs to process 6658. 
Resegmentation start 
21∶27∶49 PM Finish the file writing ! 

Save the deeper model in LR_model2 depth = 12 
scp -r binxu.wang@login.chpc.wustl.edu:/scratch/binxu.wang/ffn-Data/models/LR_model2/ .

## Longtime training of model
### Train normal field of view model 

```bash
python3 train.py \
--train_coords /home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/tf_record_file_LR \
--data_volumes LGN_LR:/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/grayscale_maps_LR.h5:raw \
--label_volumes LGN_LR:/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/groundtruth_LR.h5:stack \
--train_dir /tmp/LR_model \
--model_name convstack_3d.ConvStack3DFFNModel \
--model_args "{\"depth\": 9, \"fov_size\": [55, 37, 17], \"deltas\": [9,6,3]}" \
--max_steps 30000000 \
--summary_rate_secs 300 \
--image_mean 136 \
--image_stddev 55 \
--permutable_axes 0
```
no `--fov_policy 'max_pred_moves'` 


### Train maximum pred move smaller field of view model 
```bash
python train.py \
--train_coords /home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/tf_record_file_LR \
--data_volumes LGN_LR:/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/grayscale_maps_LR.h5:raw \
--label_volumes LGN_LR:/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/groundtruth_LR.h5:stack \
--train_dir /tmp/LR_model_test2 \
--model_name convstack_3d.ConvStack3DFFNModel \
--model_args "{\"depth\": 9, \"fov_size\": [37, 25, 15], \"deltas\": [8,6,2]}" \
--fov_policy 'max_pred_moves' \
--max_steps 1000000 \
--summary_rate_secs 300 \
--image_mean 136 \
--image_stddev 55 \
--permutable_axes 0
```
margin 16,35,53 (WF)
margin 11,24,36
4,12,16
7,12,20
15, 25, 41
12, 23, 37

After long term training, model.ckpt-259308 still cannnot recognize membrane correctly. But note that this model do inference really quick! Leakage / merging happens so frequently. 



## Local inference Long term training 
```bash
source ~/virtenvs/test1/bin/activate
export Request='image {
 hdf5: "/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/grayscale_maps_LR.h5:raw"
}
image_mean: 136
image_stddev: 55
checkpoint_interval: 1200
seed_policy: "PolicyPeaks"
model_checkpoint_path: "models/LR_model_Longtime/model.ckpt-230168"
model_name: "convstack_3d.ConvStack3DFFNModel"
model_args: "{\"depth\": 9, \"fov_size\": [55, 37, 17], \"deltas\": [9,6,3]}"
segmentation_output_dir: "results/LGN/testing_LR_Longtime2"
inference_options {
  init_activation: 0.95
  pad_value: 0.05
  move_threshold: 0.85
  min_boundary_dist { x: 5 y: 5 z: 1}
  segment_threshold: 0.6
  min_segment_size: 1000
}'
cd Downloads/ffn-master
python run_inference.py \
  --inference_request="$Request" \
  --bounding_box 'start { x:0 y:0 z:0 } size { x:1180 y:1058 z:175 }' 
```

> 16:42 start stuck????
I0123 16:43:01.390228 140031168128832 inference.py:554] [cl 0] Created supervoxel:31  seed(zyx):(8, 61, 655)  size:2837  iters:1
I0123 16:43:01.390513 140031168128832 inference.py:554] [cl 0] Starting segmentation at (8, 61, 717) (zyx)
I0123 16:43:04.068252 140031168128832 inference.py:554] [cl 0] Failed: weak seed
I0123 16:43:04.068701 140031168128832 inference.py:554] [cl 0] Starting segmentation at (8, 62, 888) (zyx)
I0123 16:43:04.479909 140031168128832 inference.py:554] [cl 0] Created supervoxel:32  seed(zyx):(8, 62, 888)  size:7025  iters:1
I0123 16:43:04.480253 140031168128832 inference.py:554] [cl 0] Starting segmentation at (8, 63, 1001) (zyx)


there are usual stucks ! in the code but why??

> I0123 17:01:41.457992 140289761634112 inference.py:554] [cl 0] Starting segmentation at (8, 53, 1134) (zyx)
I0123 17:01:42.008541 140289761634112 inference.py:554] [cl 0] Created supervoxel:29  seed(zyx):(8, 53, 1134)  size:3626  iters:1
I0123 17:01:42.008889 140289761634112 inference.py:554] [cl 0] Starting segmentation at (8, 56, 134) (zyx)
I0123 17:01:42.412203 140289761634112 inference.py:554] [cl 0] Created supervoxel:30  seed(zyx):(8, 56, 134)  size:1171  iters:1
I0123 17:01:42.412626 140289761634112 inference.py:554] [cl 0] Starting segmentation at (8, 57, 77) (zyx)
I0123 17:01:42.816596 140289761634112 inference.py:554] [cl 0] Failed: too small: 105
I0123 17:01:42.816968 140289761634112 inference.py:554] [cl 0] Starting segmentation at (8, 59, 115) (zyx)
I0123 17:01:43.232366 140289761634112 inference.py:554] [cl 0] Failed: too small: 72
I0123 17:01:43.232676 140289761634112 inference.py:554] [cl 0] Starting segmentation at (8, 59, 798) (zyx)


This time, Really large and movement based segment! 

> I0123 17:21:37.205122 139877120067392 inference.py:554] [cl 0] Created supervoxel:1  seed(zyx):(8, 18, 61)  size:9817  iters:1
I0123 17:21:37.205513 139877120067392 inference.py:554] [cl 0] Starting segmentation at (8, 18, 113) (zyx)
I0123 17:21:37.613900 139877120067392 inference.py:554] [cl 0] Created supervoxel:2  seed(zyx):(8, 18, 113)  size:14471  iters:1
I0123 17:21:37.614259 139877120067392 inference.py:554] [cl 0] Starting segmentation at (8, 18, 201) (zyx)
I0123 17:21:38.031404 139877120067392 inference.py:554] [cl 0] Created supervoxel:3  seed(zyx):(8, 18, 201)  size:15173  iters:1
I0123 17:21:38.031712 139877120067392 inference.py:554] [cl 0] Starting segmentation at (8, 18, 244) (zyx)
/home/morganlab/Downloads/ffn-master/ffn/inference/inference.py:627: RuntimeWarning: invalid value encountered in greater_equal
mask = self.seed[sel] >= self.options.segment_threshold
I0123 17:22:37.690314 139877120067392 inference.py:554] [cl 0] Created supervoxel:4  seed(zyx):(8, 18, 244)  size:77788  iters:242
I0123 17:22:37.690902 139877120067392 inference.py:554] [cl 0] Starting segmentation at (8, 19, 444) (zyx)
I0123 17:22:38.250994 139877120067392 inference.py:554] [cl 0] Created supervoxel:5  seed(zyx):(8, 19, 444)  size:4176  iters:1
I0123 17:22:38.251336 139877120067392 inference.py:554] [cl 0] Starting segmentation at (8, 19, 533) (zyx)
I0123 17:22:38.669008 139877120067392 inference.py:554] [cl 0] Created supervoxel:6  seed(zyx):(8, 19, 533)  size:8303  iters:1
I0123 17:22:38.669363 139877120067392 inference.py:554] [cl 0] Starting segmentation at (8, 19, 793) (zyx)
I0123 17:22:39.087285 139877120067392 inference.py:554] [cl 0] Created supervoxel:7  seed(zyx):(8, 19, 793)  size:21272  iters:1
I0123 17:22:39.087636 139877120067392 inference.py:554] [cl 0] Starting segmentation at (8, 19, 975) (zyx)


Things get stuck really because the segment_at is using too many iterations and just cannot stop!! 
It will generate giant segments instead of small ones! Which is good


```bash
source ~/virtenvs/test1/bin/activate
export Request='image {
 hdf5: "/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/grayscale_maps_LR.h5:raw"
}
image_mean: 136
image_stddev: 55
checkpoint_interval: 1200
seed_policy: "PolicyPeaks"
model_checkpoint_path: "models/LR_model_Longtime/model.ckpt-264380"
model_name: "convstack_3d.ConvStack3DFFNModel"
model_args: "{\"depth\": 9, \"fov_size\": [55, 37, 17], \"deltas\": [9,6,3]}"
segmentation_output_dir: "results/LGN/testing_LR_Longtime_new"
inference_options {
  init_activation: 0.95
  pad_value: 0.05
  move_threshold: 0.9
  min_boundary_dist { x: 5 y: 5 z: 1}
  segment_threshold: 0.6
  min_segment_size: 1000
  disco_seed_threshold: 0.15
}'
cd Downloads/ffn-master
python run_inference.py \
  --inference_request="$Request" \
  --bounding_box 'start { x:0 y:0 z:0 } size { x:1180 y:1058 z:175 }' 
```



## Test the segmentation from given seeds ! 

```bash
source  ~/virtenvs/test1/bin/activate
cd  PycharmProjects/FloodFillNetwork-Notes/
python3 run_inference_script.py
```

### Normal long term model: 
```python
config = '''image {
 hdf5: "/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/grayscale_maps_LR.h5:raw"
}
image_mean: 136
image_stddev: 55
checkpoint_interval: 1200
seed_policy: "PolicyPeaks"
model_checkpoint_path: "/home/morganlab/Downloads/ffn-master/models/LR_model_Longtime/model.ckpt-264380"
model_name: "convstack_3d.ConvStack3DFFNModel"
model_args: "{\\"depth\\": 9, \\"fov_size\\": [55, 37, 17], \\"deltas\\": [9,6,3]}"
segmentation_output_dir: "/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR_Longtime_NF_point"
inference_options {
  init_activation: 0.95
  pad_value: 0.05
  move_threshold: 0.90
  min_boundary_dist { x: 5 y: 5 z: 1}
  segment_threshold: 0.6
  min_segment_size: 1000
  disco_seed_threshold: 0.30
'''
```
```
I0124 23:37:51.036206 140017940272960 inference.py:557] [cl 0] Created supervoxel:1  seed(zyx):(92, 764, 306)  size:128344 (raw size 128344)  iters:87
I0124 23:37:59.974752 140017940272960 inference.py:557] [cl 0] Starting segmentation at (92, 90, 308) (zyx)
/home/morganlab/PycharmProjects/FloodFillNetwork-Notes/ffn/inference/inference.py:549: RuntimeWarning: invalid value encountered in greater_equal

I0125 06:51:41.916663 140017940272960 inference.py:557] [cl 0] Iteration #: 659500, Segmented Voxels: 106480047
I0125 06:51:45.667349 140017940272960 inference.py:557] [cl 0] Iteration #: 659600, Segmented Voxels: 106489737
I0125 06:51:49.462082 140017940272960 inference.py:557] [cl 0] Iteration #: 659700, Segmented Voxels: 106492544
Overlapping segments (label, size):  (1, 106477)
I0125 06:51:54.449415 140017940272960 inference.py:557] [cl 0] Created supervoxel:2  seed(zyx):(92, 90, 308)  size:104388760 (raw size 104495237)  iters:659744
I0125 06:52:08.851992 140017940272960 inference.py:557] [cl 0] Starting segmentation at (43, 936, 808) (zyx)
Overlapping segments (label, size):  (2, 1)
I0125 06:52:09.958729 140017940272960 inference.py:557] [cl 0] Created supervoxel:3  seed(zyx):(43, 936, 808)  size:0 (raw size 1)  iters:0
I0125 06:52:20.152999 140017940272960 inference.py:557] [cl 0] Starting segmentation at (43, 356, 72) (zyx)
Overlapping segments (label, size):  (2, 529)
I0125 06:52:22.510858 140017940272960 inference.py:557] [cl 0] Created supervoxel:4  seed(zyx):(43, 356, 72)  size:69017 (raw size 69546)  iters:24
I0125 06:52:31.978192 140017940272960 inference.py:557] [cl 0] Starting segmentation at (45, 84, 200) (zyx)
Overlapping segments (label, size):  (2, 1)
I0125 06:52:33.412392 140017940272960 inference.py:557] [cl 0] Created supervoxel:5  seed(zyx):(45, 84, 200)  size:0 (raw size 1)  iters:0
I0125 06:52:40.994081 140017940272960 inference.py:557] [cl 0] Starting segmentation at (45, 124, 666) (zyx)
Overlapping segments (label, size):  (2, 1)
I0125 06:52:42.456773 140017940272960 inference.py:557] [cl 0] Created supervoxel:6  seed(zyx):(45, 124, 666)  size:0 (raw size 1)  iters:0
I0125 06:52:49.944672 140017940272960 inference.py:557] [cl 0] Starting segmentation at (45, 350, 60) (zyx)
Overlapping segments (label, size):  (4, 1)
I0125 06:52:51.413688 140017940272960 inference.py:557] [cl 0] Created supervoxel:7  seed(zyx):(45, 350, 60)  size:0 (raw size 1)  iters:0
```

For one seed to segment out 60% of the volume. takes 659700 iterations and 7 hours! 


### Narrow field trained to move model
```python
config = '''image {
 hdf5: "/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/grayscale_maps_LR.h5:raw"
}
image_mean: 136
image_stddev: 55
checkpoint_interval: 1200
seed_policy: "PolicyPeaks"
model_checkpoint_path: "/home/morganlab/Downloads/ffn-master/models/LR_model_Longtime_Mov/model.ckpt-259308"
model_name: "convstack_3d.ConvStack3DFFNModel"
model_args: "{\\"depth\\": 9, \\"fov_size\\": [37, 25, 15], \\"deltas\\": [8,6,2]}"
segmentation_output_dir: "/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR_Longtime_NF_point"
inference_options {
  init_activation: 0.95
  pad_value: 0.05
  move_threshold: 0.90
  min_boundary_dist { x: 5 y: 5 z: 1}
  segment_threshold: 0.6
  min_segment_size: 1000
  disco_seed_threshold: 0.30
}'''

seed_list = [(612, 1528, 92), (616, 180, 92), (1616, 1872, 43), (144, 712, 43), (400, 168, 45), (1332, 248, 45), (120, 700,45)]  # in xyz order
```


After long term training, model.ckpt-259308 still cannnot recognize membrane correctly. But note that this model do inference really quick! Leakage / merging happens so frequently. 


## First Really successful model segmentation! 
```python
config = '''image {
 hdf5: "/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/grayscale_maps_LR.h5:raw"
}
image_mean: 136
image_stddev: 55
checkpoint_interval: 1200
seed_policy: "PolicyPeaks"
model_checkpoint_path: "/home/morganlab/Downloads/ffn-master/models/LR_model_Longtime/model.ckpt-415908"
model_name: "convstack_3d.ConvStack3DFFNModel"
model_args: "{\\"depth\\": 9, \\"fov_size\\": [55, 37, 17], \\"deltas\\": [9,6,3]}"
segmentation_output_dir: "/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR_Longtime_point"
inference_options {
  init_activation: 0.95
  pad_value: 0.05
  move_threshold: 0.90
  min_boundary_dist { x: 5 y: 5 z: 1}
  segment_threshold: 0.6
  min_segment_size: 1000
  disco_seed_threshold: 0.005
}'''
```

See `Inference_log_success.md`



Try to do full segmentation! 
```bash
# source ~/virtenvs/test1/bin/activate
export Request='image {
 hdf5: "/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/grayscale_maps_LR.h5:raw"
}
image_mean: 136
image_stddev: 55
checkpoint_interval: 1200
seed_policy: "PolicyPeaks"
model_checkpoint_path: "/home/morganlab/Downloads/ffn-master/models/LR_model_Longtime/model.ckpt-415908"
model_name: "convstack_3d.ConvStack3DFFNModel"
model_args: "{\"depth\": 9, \"fov_size\": [55, 37, 17], \"deltas\": [9,6,3]}"
segmentation_output_dir: "/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR_Longtime_success"
inference_options {
  init_activation: 0.95
  pad_value: 0.05
  move_threshold: 0.90
  min_boundary_dist { x: 5 y: 5 z: 1}
  segment_threshold: 0.6
  min_segment_size: 5000
  disco_seed_threshold: 0.005
}'
cd PycharmProjects/FloodFillNetwork-Notes
python3 run_inference.py \
  --inference_request="$Request" \
  --bounding_box 'start { x:0 y:0 z:0 } size { x:1180 y:1058 z:175 }'
```
 > inference_log.log

Start processing at 18:18 and do 325097 seeds! 

morganlab@ml-linux:~/PycharmProjects/FloodFillNetwork-Notes$ python3 run_inference.py \
>   --inference_request="$Request" \
>   --bounding_box 'start { x:0 y:0 z:0 } size { x:1180 y:1058 z:175 }' > inference_log.log


> ......
I0128 03:39:00.259603 139723875120960 inference.py:560] [cl 0] Starting segmentation at (166, 1039, 831) (zyx)
I0128 03:39:00.411037 139723875120960 inference.py:560] [cl 0] Failed: too small: 441
I0128 03:39:00.506173 139723875120960 inference.py:560] [cl 0] Segmentation done.
I0128 03:39:00.506249 139723875120960 inference.py:306] Deregistering client 0
I0128 03:39:00.506444 139715135133440 executor.py:200] client 0 terminating
I0128 03:39:14.343287 139723875120960 executor.py:169] Requesting executor shutdown.
I0128 03:39:14.343586 139715135133440 executor.py:191] Executor shut down requested.
I0128 03:39:14.392710 139723875120960 executor.py:172] Executor shutdown complete.


~ 10 hour to segment all ! Quite a few really nice branches! 

## Try the NF and WF network segment from points 

### Narrow field High movement model-953997
```python
config = '''image {
 hdf5: "/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/grayscale_maps_LR.h5:raw"
}
image_mean: 136
image_stddev: 55
checkpoint_interval: 1200
seed_policy: "PolicyPeaks"
model_checkpoint_path: "/home/morganlab/Downloads/ffn-master/models/LR_model_Longtime_Mov/model.ckpt-953997"
model_name: "convstack_3d.ConvStack3DFFNModel"
model_args: "{\\"depth\\": 9, \\"fov_size\\": [37, 25, 15], \\"deltas\\": [8,6,2]}"
segmentation_output_dir: "/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR_Longtime_Mov_point"
inference_options {
  init_activation: 0.95
  pad_value: 0.05
  move_threshold: 0.90
  min_boundary_dist { x: 5 y: 5 z: 1}
  segment_threshold: 0.6
  min_segment_size: 10000
  disco_seed_threshold: 0.005
}'''
seed_list = [(1080, 860, 72), (1616, 1872, 43), (612, 1528, 92), (616, 180, 92),  (144, 712, 43), (400, 168, 45), (1332, 248, 45), (120, 700,45)]  # in xyz order
```

**Output**: 
> 2019-01-28 16:18:47.005045: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-01-28 16:18:47.135531: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1392] Found device 0 with properties: 
name: Quadro M4000 major: 5 minor: 2 memoryClockRate(GHz): 0.7725
pciBusID: 0000:03:00.0
totalMemory: 7.91GiB freeMemory: 5.36GiB
2019-01-28 16:18:47.135564: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1471] Adding visible gpu devices: 0
2019-01-28 16:18:47.412199: I tensorflow/core/common_runtime/gpu/gpu_device.cc:952] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-01-28 16:18:47.412240: I tensorflow/core/common_runtime/gpu/gpu_device.cc:958]      0 
2019-01-28 16:18:47.412245: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 0:   N 
2019-01-28 16:18:47.412429: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1084] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 5125 MB memory) -> physical GPU (device: 0, name: Quadro M4000, pci bus id: 0000:03:00.0, compute capability: 5.2)
INFO:tensorflow:Restoring parameters from /home/morganlab/Downloads/ffn-master/models/LR_model_Longtime_Mov/model.ckpt-953997

16:18~16:50 segment out 8 points 


See `Inference_log_NF_Mov1.md`


Start to do full segmentation with NF model 
```bash
# source ~/virtenvs/test1/bin/activate
export Request='image {
 hdf5: "/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/grayscale_maps_LR.h5:raw"
}
image_mean: 136
image_stddev: 55
checkpoint_interval: 1200
seed_policy: "PolicyPeaks"
model_checkpoint_path: "/home/morganlab/Downloads/ffn-master/models/LR_model_Longtime_Mov/model.ckpt-993165"
model_name: "convstack_3d.ConvStack3DFFNModel"
model_args: "{\"depth\": 9, \"fov_size\": [37, 25, 15], \"deltas\": [8,6,2]}"
segmentation_output_dir: "/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR_Longtime_Mov_full"
inference_options {
  init_activation: 0.95
  pad_value: 0.05
  move_threshold: 0.90
  min_boundary_dist { x: 5 y: 5 z: 1}
  segment_threshold: 0.6
  min_segment_size: 10000
  disco_seed_threshold: 0.005
}'
cd PycharmProjects/FloodFillNetwork-Notes
python3 run_inference.py \
  --inference_request="$Request" \
  --bounding_box 'start { x:0 y:0 z:0 } size { x:1180 y:1058 z:175 }'
```

See `Inference_log_NF_Mov_full.md`


### Wide field less trained model 
```python
config = '''image {
 hdf5: "/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/grayscale_maps_LR.h5:raw"
}
image_mean: 136
image_stddev: 55
checkpoint_interval: 1200
seed_policy: "PolicyPeaks"
model_checkpoint_path: "/home/morganlab/Downloads/ffn-master/models/LR_model_WF_Longtime/model.ckpt-214465" 
model_name: "convstack_3d.ConvStack3DFFNModel"
model_args: "{\\"depth\\": 9, \\"fov_size\\": [77, 51, 23], \\"deltas\\": [15,10,5]}"
segmentation_output_dir: "/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR_WF_Longtime_point"
inference_options {
  init_activation: 0.95
  pad_value: 0.05
  move_threshold: 0.90
  min_boundary_dist { x: 5 y: 5 z: 1}
  segment_threshold: 0.6
  min_segment_size: 10000
  disco_seed_threshold: 0.005
}'''
seed_list = [(1080, 860, 72), (1616, 1872, 43), (612, 1528, 92), (616, 180, 92),  (144, 712, 43), (400, 168, 45), (1332, 248, 45), (120, 700,45)]  # in xyz order
```

**Wide field model is Still not good enough! needs more training** 

> 2019-01-28 17:25:12.830664: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1471] Adding visible gpu devices: 0
2019-01-28 17:25:12.830752: I tensorflow/core/common_runtime/gpu/gpu_device.cc:952] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-01-28 17:25:12.830763: I tensorflow/core/common_runtime/gpu/gpu_device.cc:958]      0 
2019-01-28 17:25:12.830770: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 0:   N 
2019-01-28 17:25:12.831043: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1084] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 5125 MB memory) -> physical GPU (device: 0, name: Quadro M4000, pci bus id: 0000:03:00.0, compute capability: 5.2)
......
INFO:root:Available TF devices: [_DeviceAttributes(/job:localhost/replica:0/task:0/device:CPU:0, CPU, 268435456), _DeviceAttributes(/job:localhost/replica:0/task:0/device:GPU:0, GPU, 5374869504)]
INFO:root:Importing symbol ConvStack3DFFNModel from ffn.training.models.convstack_3d
INFO:root:Loading checkpoint.
INFO:tensorflow:Restoring parameters from /home/morganlab/Downloads/ffn-master/models/LR_model_WF_Longtime/model.ckpt-214465
INFO:root:Checkpoint loaded.
INFO:root:Executor starting.
INFO:root:Process subvolume: (0, 0, 0)
INFO:root:Requested bounds are (0, 0, 0) + (175, 1058, 1180)
INFO:root:Destination bounds are (0, 0, 0) + (175, 1058, 1180)
INFO:root:Fetch bounds are array([0, 0, 0]) + array([ 175, 1058, 1180])
INFO:root:Fetched image of size (175, 1058, 1180) prior to transform
INFO:root:Image data loaded, shape: (175, 1058, 1180).
INFO:root:Registered as client 0.
INFO:root:client 0 starting
INFO:root:Deregistering client 0
INFO:root:[cl 0] Starting segmentation at (72, 430, 540) (zyx)
INFO:root:[cl 0] Created supervoxel:1  seed(zyx):(72, 430, 540)  size:50891 (raw size 50891)  iters:1
INFO:root:[cl 0] Starting segmentation at (43, 936, 808) (zyx)
INFO:root:[cl 0] Created supervoxel:2  seed(zyx):(43, 936, 808)  size:195904 (raw size 195904)  iters:73
INFO:root:[cl 0] Starting segmentation at (92, 764, 306) (zyx)
INFO:root:[cl 0] Created supervoxel:3  seed(zyx):(92, 764, 306)  size:141280 (raw size 141280)  iters:18
INFO:root:[cl 0] Starting segmentation at (92, 90, 308) (zyx)
INFO:root:[cl 0] Created supervoxel:4  seed(zyx):(92, 90, 308)  size:263258 (raw size 263258)  iters:64
INFO:root:[cl 0] Starting segmentation at (43, 356, 72) (zyx)
INFO:root:[cl 0] Created supervoxel:5  seed(zyx):(43, 356, 72)  size:87692 (raw size 87692)  iters:12
INFO:root:[cl 0] Starting segmentation at (45, 84, 200) (zyx)
INFO:root:[cl 0] Iteration #: 100, Segmented Voxels: 245495
INFO:root:[cl 0] Iteration #: 200, Segmented Voxels: 260421
INFO:root:[cl 0] Iteration #: 300, Segmented Voxels: 265640
INFO:root:[cl 0] Iteration #: 400, Segmented Voxels: 275290
INFO:root:[cl 0] Iteration #: 500, Segmented Voxels: 268594
INFO:root:[cl 0] Iteration #: 600, Segmented Voxels: 269611
INFO:root:[cl 0] Created supervoxel:6  seed(zyx):(45, 84, 200)  size:259455 (raw size 259455)  iters:667
INFO:root:[cl 0] Starting segmentation at (45, 124, 666) (zyx)
INFO:root:[cl 0] Iteration #: 100, Segmented Voxels: 127803
INFO:root:[cl 0] Iteration #: 200, Segmented Voxels: 113867
INFO:root:[cl 0] Iteration #: 300, Segmented Voxels: 109722
INFO:root:[cl 0] Created supervoxel:7  seed(zyx):(45, 124, 666)  size:104128 (raw size 104128)  iters:325
INFO:root:[cl 0] Starting segmentation at (45, 350, 60) (zyx)
INFO:root:[cl 0] Created supervoxel:8  seed(zyx):(45, 350, 60)  size:0 (raw size 1)  iters:0




## Normal field model on new segment!

```python
config = '''image {
 hdf5: "/home/morganlab/Documents/Sample1_branch109/grayscale_branch.h5:raw"
}
image_mean: 140
image_stddev: 43
checkpoint_interval: 1200
seed_policy: "PolicyPeaks"
model_checkpoint_path: "/home/morganlab/Downloads/ffn-master/models/LR_model_Longtime/model.ckpt-687329"
model_name: "convstack_3d.ConvStack3DFFNModel"
model_args: "{\\"depth\\": 9, \\"fov_size\\": [55, 37, 17], \\"deltas\\": [9,6,3]}"
segmentation_output_dir: "/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR_Longtime_point2"
inference_options {
  init_activation: 0.95
  pad_value: 0.05
  move_threshold: 0.90
  min_boundary_dist { x: 5 y: 5 z: 1}
  segment_threshold: 0.6
  min_segment_size: 1000
  disco_seed_threshold: 0.005
}'''
seed_list = [(707, 353, 385), (371, 478, 384), (259, 391, 386), (88,264,333), (830, 894, 333), (502,462,278), (878, 256, 278)]
downsample_factor =1 # Mip level 0, df=1;  Mip level 1 df =2
canvas_bbox = [(0, 0, 0), (441, 1024, 1024)]
```

First 3 seeds stop at hundreds of steps, but the 4th seed just give 892600 steps and it's too much 
2019-01-31 11:39:19,898 [inference.py: log_info(): 559] [cl 0] Iteration #: 892600, Segmented Voxels: 195351134

And the segmented points are not conforming to any boundaries as well. 

### Train Speed Statistics
#### On Cluster
For small size model `model_args "{\"depth\": 9, \"fov_size\": [37, 25, 15], \"deltas\": [8,6,2]}" \` with extra move:

* 151 steps/min, 9069 steps/hr for  nodes=2:ppn=8:gpus=4


For normal size model `model_args "{\"depth\": 9, \"fov_size\": [55, 37, 17], \"deltas\": [9,6,3]}"`: 

* 62 stpes/min, 3730 steps/hr  for 2 nodes: 8ppn: 4 GPUs
* 56 steps/min, 3360 steps/hr  for 1 nodes: 4ppn: 4 GPUs


For large FoV model `model_args: "{\"depth\": 9, \"fov_size\": [77, 51, 23], \"deltas\": [15,10,5]}"`

* 19.8 steps/min, 1188 steps/hr  for  nodes=2:ppn=8:gpus=4,

#### Local Training
> 2019-01-29 15:02:37.531824: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1392] Found device 0 with properties: 
name: Quadro M4000 major: 5 minor: 2 memoryClockRate(GHz): 0.7725
pciBusID: 0000:03:00.0
totalMemory: 7.91GiB freeMemory: 4.58GiB
2019-01-29 15:02:37.531861: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1471] Adding visible gpu devices: 0
2019-01-29 15:02:37.783730: I tensorflow/core/common_runtime/gpu/gpu_device.cc:952] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-01-29 15:02:37.783777: I tensorflow/core/common_runtime/gpu/gpu_device.cc:958]      0 
2019-01-29 15:02:37.783783: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 0:   N 
2019-01-29 15:02:37.783975: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1084] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 4328 MB memory) -> physical GPU (device: 0, name: Quadro M4000, pci bus id: 0000:03:00.0, compute capability: 5.2)


Small model NF-model 5.4 steps/sec on local machine (19440 steps / hr) faster than Cluster ! 

Normal size model 2.2 steps/sec, 7920 steps/hr, faster than Cluster ! 

Seems with a single fast GPU, it's better than 4 slow GPU connected together. 


When doing inference register multiple client may increase speed !! 

## Check training process 
```bash
tensorboard --logdir=/home/morganlab/Downloads/ffn-mastemodels/LR_model_Longtime_Mov
tensorboard --logdir=/home/morganlab/Downloads/ffn-mastemodels/LR_model_Longtime
tensorboard --logdir=/home/morganlab/Downloads/ffn-mastemodels/LR_model_WF_Longtime
```

http://ml-linux:6006



## Application to other volumes

### Branched volume


After doing normalization and resizing, the segmentation of looks 
alright!

Normal field of view model: 

* The model is able to see membranes and myelination. 
* But movement is still not so well. 

Narrow field of view high movement model

* still segment out a mess and just cannot stop! .


```bash
# source ~/virtenvs/test1/bin/activate
export Request='image {
 hdf5: "/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/grayscale_maps_LR.h5:raw"
}
image_mean: 136
image_stddev: 55
checkpoint_interval: 1200
seed_policy: "PolicyPeaks"
model_checkpoint_path: "/home/morganlab/Downloads/ffn-master/models/LR_model_Longtime/model.ckpt-746473"
model_name: "convstack_3d.ConvStack3DFFNModel"
model_args: "{\"depth\": 9, \"fov_size\": [55, 37, 17], \"deltas\": [9,6,3]}"
segmentation_output_dir: "/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR_Longtime_success2"
inference_options {
  init_activation: 0.95
  pad_value: 0.05
  move_threshold: 0.90
  min_boundary_dist { x: 5 y: 5 z: 1}
  segment_threshold: 0.6
  min_segment_size: 5000
  disco_seed_threshold: 0.005
}'
cd PycharmProjects/FloodFillNetwork-Notes
python3 run_inference.py \
  --inference_request="$Request" \
  --bounding_box 'start { x:0 y:0 z:0 } size { x:1180 y:1058 z:175 }'
```

## P11 Volume

x (14296, 16343)
y (12056, 14103)
z (0, 151)
 4,4,40

 mip =1 
 1024, 1024, 151 volume

 (14364, 12932,43)
 Train small scale model! 

 (7148, 6028)
```bash
cd /home/morganlab/Downloads/ffn-master/
python3  train.py \
--train_coords third_party/LGN_DATA/tf_record_file_LR_WF \
--data_volumes LGN_LR:third_party/LGN_DATA/grayscale_maps_LR.h5:raw \
--label_volumes LGN_LR:third_party/LGN_DATA/groundtruth_LR.h5:stack \
--train_dir /tmp/models/LR_model_Longtime_Mov \
--model_name convstack_3d.ConvStack3DFFNModel \
--model_args "{\"depth\": 12, \"fov_size\": [33, 33, 17], \"deltas\": [8, 8, 4]}" \
--max_steps 30000000 \
--summary_rate_secs 300 \
--image_mean 136 \
--image_stddev 55 \
```


## Visualization 
```bash
module load singularity-2.4.2
export HDF5_USE_FILE_LOCKING=FALSE
module load singularity-2.4.2
singularity exec -B /scratch:/scratch --nv /export/tensorflow-1.7.0/test/ubuntu_tf_gpu python3 ~/FloodFillNetwork-Notes/proc_segmentation_script.py \
  --seg_dir /scratch/binxu.wang/results/LGN/testing_exp7 \
  --seg_export_dir  /scratch/binxu.wang/results/LGN/testing_exp7/Autoseg_exp7 \
  --imageh5_dir /scratch/binxu.wang/ffn-Data/LGN_DATA/grayscale_maps_LR.h5 \
  --render_dir /scratch/binxu.wang/results/LGN/testing_exp7/Autoseg_exp7 \
  --visualize True  



python3 proc_segmentation_script.py \
  --seg_dir /home/morganlab/Downloads/ffn-master/results/LGN/testing_exp8 \
  --seg_export_dir  /home/morganlab/Documents/Autoseg_result/Autoseg_exp8 \
  --imageh5_dir  "/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/grayscale_maps_LR(copy).h5" \
  --render_dir  /home/morganlab/Documents/Autoseg_result/Autoseg_exp8 \
  --visualize True  
```


```bash   
export Request='image {
 hdf5: "/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/grayscale_maps_LR(copy).h5:raw"
}
image_mean: 136
image_stddev: 55
checkpoint_interval: 1200
seed_policy: "PolicyPeaks"
model_checkpoint_path: "/home/morganlab/Downloads/ffn-master/models/LR_model_Longtime/model.ckpt-3205589"
model_name: "convstack_3d.ConvStack3DFFNModel"
model_args: "{\"depth\": 9, \"fov_size\": [55, 37, 17], \"deltas\": [9,6,3]}"
segmentation_output_dir: "/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR_Longtime_point3"
inference_options {
  init_activation: 0.95
  pad_value: 0.05
  move_threshold: 0.90
  min_boundary_dist { x: 5 y: 5 z: 1}
  segment_threshold: 0.6
  min_segment_size: 1000
  disco_seed_threshold: 0.005
}'
cd PycharmProjects/FloodFillNetwork-Notes/
python3 run_inference_from_seed.py \
  --inference_request "$Request" \
  --bounding_box 'start { x:0 y:0 z:0 } size { x:1180 y:1058 z:175 }' \
  --downsample_factor 2 \
  --corner "(0,0,0)" \
  --seed_list "[(1080, 860, 72), (1616, 1872, 43), (612, 1528, 92), (616, 180, 92),  (144, 712, 43), (400, 168, 45), (1332, 248, 45), (120, 700,45)] "\
  --logfile_name "inference_tmp.log"

```

```
```


### Tensorboard Visualization 
```bash
cd /home/morganlab/Downloads/ffn-master/models/
tensorboard --logdir Norm:LR_model_Longtime,Mov:LR_model_Longtime_Mov,SF:LR_model_Longtime_SF_Deep,WF:LR_model_WF_Longtime,SF_Cl:LR_model_Longtime_SF_Deep_cluster
```
```bash
cd /home/morganlab/Downloads/ffn-master/models/
tensorboard --logdir Norm:LR_model_Longtime,Mov:LR_model_Longtime_Mov,Deep_cluster:LR_model_Longtime_SF_Deep_cluster,WF:LR_model_WF_Longtime
```
Find the ROI cube, export segments, 
On Cluster get the images, gen h5 file
Do saturated or sparse segmentaion

New cube 
22660 : 21636, 
15388 : 14364, 
0
(10818, 7182, 0)

```bash
python3 generate_h5_file.py \
 --stack_n 152 --name_pattern "IxD_W002_invert2_2_export_s%03d.png" --path "/home/morganlab/Documents/ixP11LGN" --output_name "grayscale_ixP11_1.h5"

```
IxD_W002_invert2_2_export


## Consensus Segmentation
```python
config = """
segmentation1 {
    directory: "/home/morganlab/Documents/ixP11LGN/p11_1_exp2/"
    threshold: 0.6
    split_cc: 1
    min_size: 5000
}
segmentation2 {
    directory: "/home/morganlab/Documents/ixP11LGN/p11_1_exp3/"
    threshold: 0.6
    split_cc: 1
    min_size: 5000
}
segmentation_output_dir: "/home/morganlab/Documents/ixP11LGN/p11_1_consensus_2_3/"
type: CONSENSUS_SPLIT
split_min_size: 5000
"""
#%%
corner = (0,0,0)
consensus_req = consensus_pb2.ConsensusRequest()
_ = text_format.Parse(config, consensus_req)
cons_seg, origin = consensus.compute_consensus(corner, consensus_req)
#%%
seg_path = storage.segmentation_path(consensus_req.segmentation_output_dir, corner)
storage.save_subvolume(cons_seg, origin, seg_path)
```


```log
WARNING: Logging before flag parsing goes to stderr.
I0205 16:14:11.289180 140550458976064 consensus.py:78] consensus: mem[start] = 236 MiB
I0205 16:14:13.579145 140550458976064 storage.py:430] loading segmentation from: /home/morganlab/Documents/ixP11LGN/p11_1_exp2/0/0/seg-0_0_0.npz
I0205 16:14:13.579325 140550458976064 storage.py:433] thresholding at 0.600000
/home/morganlab/PycharmProjects/FloodFillNetwork-Notes/ffn/inference/storage.py:256: RuntimeWarning: invalid value encountered in less
  labels[prob < threshold] = 0
I0205 16:14:17.194858 140550458976064 storage.py:442] clean up with split_cc=True, min_size=5000
I0205 16:15:09.224370 140550458976064 consensus.py:82] consensus: v1 data loaded
I0205 16:15:11.271337 140550458976064 storage.py:430] loading segmentation from: /home/morganlab/Documents/ixP11LGN/p11_1_exp3/0/0/seg-0_0_0.npz
I0205 16:15:11.272299 140550458976064 storage.py:433] thresholding at 0.600000
I0205 16:15:14.597897 140550458976064 storage.py:442] clean up with split_cc=True, min_size=5000
I0205 16:15:59.285499 140550458976064 consensus.py:84] consensus: v2 data loaded
I0205 16:15:59.286098 140550458976064 consensus.py:87] consensus: mem[data loaded] = 2674 MiB
```

## Agglomeration

Resegmentation log, 6 min for 2 pairs! 
```log
I0206 18:32:01.959901 139806330058560 resegmentation.py:309] processing 0/2
I0206 18:32:01.961467 139806330058560 inference.py:1011] Process subvolume: array([ 72, 382, 472])
I0206 18:32:01.963613 139806330058560 inference.py:1029] Requested bounds are array([ 72, 382, 472]) + array([ 41, 401, 401])
I0206 18:32:01.964161 139806330058560 inference.py:1030] Destination bounds are array([ 72, 382, 472]) + array([ 41, 401, 401])
I0206 18:32:01.964603 139806330058560 inference.py:1031] Fetch bounds are array([ 72, 382, 472]) + array([ 41, 401, 401])
I0206 18:32:01.978317 139806330058560 inference.py:1046] Fetched image of size (41, 401, 401) prior to transform
I0206 18:32:01.978958 139806330058560 inference.py:1056] Image data loaded, shape: (41, 401, 401).
I0206 18:32:02.004559 139806330058560 inference.py:301] Registered as client 0.
I0206 18:32:02.004800 139797045683968 executor.py:198] client 0 starting
I0206 18:32:02.004930 139806330058560 inference.py:559] [cl 0] Loading initial segmentation from (zyx) array([ 72, 382, 472]):array([113, 783, 873])
I0206 18:32:02.586472 139806330058560 inference.py:559] [cl 0] Segmentation loaded, shape: (41, 401, 401). Canvas segmentation is (41, 401, 401)
I0206 18:32:02.586751 139806330058560 inference.py:559] [cl 0] Segmentation cropped to: (41, 401, 401)
I0206 18:32:02.620084 139806330058560 inference.py:559] [cl 0] Max restored ID is: 165.
I0206 18:32:02.654965 139806330058560 resegmentation.py:212] processing object 0
I0206 18:32:03.959067 139806330058560 inference.py:559] [cl 0] EDT computation done
I0206 18:32:03.965220 139806330058560 inference.py:559] [cl 0] .. starting segmentation at (xyz): 265 221 11
/home/morganlab/PycharmProjects/FloodFillNetwork-Notes/ffn/inference/inference.py:423: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  self.seed[[slice(s, e) for s, e in zip(start, end)]])  # Slice out a cube around pos with `_input_seed_size`
/home/morganlab/PycharmProjects/FloodFillNetwork-Notes/ffn/inference/inference.py:384: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  img = self.image[[slice(s, e) for s, e in zip(start, end)]]
/home/morganlab/PycharmProjects/FloodFillNetwork-Notes/ffn/inference/inference.py:456: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  old_seed = self.seed[sel]
/home/morganlab/PycharmProjects/FloodFillNetwork-Notes/ffn/inference/inference.py:460: RuntimeWarning: invalid value encountered in greater_equal
  np.sum((old_seed >= logit(0.8)) & (logits < th_max)))
/home/morganlab/PycharmProjects/FloodFillNetwork-Notes/ffn/inference/inference.py:474: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  self.seed[sel] = logits  # only place, that update `seed` segmentation and seg_prob is not updated
/home/morganlab/PycharmProjects/FloodFillNetwork-Notes/ffn/inference/movement.py:80: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  face_prob = prob_map[face_sel]  # restrict to one face of cube
/home/morganlab/PycharmProjects/FloodFillNetwork-Notes/ffn/inference/inference.py:551: RuntimeWarning: invalid value encountered in greater_equal
  np.sum(self.seed >= self.options.segment_threshold)))
I0206 18:32:08.585219 139806330058560 inference.py:559] [cl 0] Iteration #: 100, Segmented Voxels: 206276
I0206 18:32:12.290488 139806330058560 inference.py:559] [cl 0] Iteration #: 200, Segmented Voxels: 285557
I0206 18:32:16.090435 139806330058560 inference.py:559] [cl 0] Iteration #: 300, Segmented Voxels: 351474
I0206 18:32:20.224900 139806330058560 inference.py:559] [cl 0] Iteration #: 400, Segmented Voxels: 422152
I0206 18:32:24.477809 139806330058560 inference.py:559] [cl 0] Iteration #: 500, Segmented Voxels: 469347
I0206 18:32:28.268579 139806330058560 inference.py:559] [cl 0] Iteration #: 600, Segmented Voxels: 481011
I0206 18:32:31.916016 139806330058560 inference.py:559] [cl 0] Iteration #: 700, Segmented Voxels: 513203
I0206 18:32:35.605421 139806330058560 inference.py:559] [cl 0] Iteration #: 800, Segmented Voxels: 521899
I0206 18:32:39.267745 139806330058560 inference.py:559] [cl 0] Iteration #: 900, Segmented Voxels: 545950
I0206 18:32:42.913712 139806330058560 inference.py:559] [cl 0] Iteration #: 1000, Segmented Voxels: 570146
I0206 18:32:46.572585 139806330058560 inference.py:559] [cl 0] Iteration #: 1100, Segmented Voxels: 580942
I0206 18:32:50.209533 139806330058560 inference.py:559] [cl 0] Iteration #: 1200, Segmented Voxels: 606072
I0206 18:32:53.863352 139806330058560 inference.py:559] [cl 0] Iteration #: 1300, Segmented Voxels: 626096
I0206 18:32:57.535719 139806330058560 inference.py:559] [cl 0] Iteration #: 1400, Segmented Voxels: 657488
I0206 18:33:01.241122 139806330058560 inference.py:559] [cl 0] Iteration #: 1500, Segmented Voxels: 678360
I0206 18:33:04.937316 139806330058560 inference.py:559] [cl 0] Iteration #: 1600, Segmented Voxels: 698420
I0206 18:33:08.618413 139806330058560 inference.py:559] [cl 0] Iteration #: 1700, Segmented Voxels: 727582
I0206 18:33:12.320817 139806330058560 inference.py:559] [cl 0] Iteration #: 1800, Segmented Voxels: 749795
I0206 18:33:16.072740 139806330058560 inference.py:559] [cl 0] Iteration #: 1900, Segmented Voxels: 771904
I0206 18:33:19.754729 139806330058560 inference.py:559] [cl 0] Iteration #: 2000, Segmented Voxels: 844908
I0206 18:33:23.441378 139806330058560 inference.py:559] [cl 0] Iteration #: 2100, Segmented Voxels: 881565
I0206 18:33:27.122524 139806330058560 inference.py:559] [cl 0] Iteration #: 2200, Segmented Voxels: 925475
I0206 18:33:30.796905 139806330058560 inference.py:559] [cl 0] Iteration #: 2300, Segmented Voxels: 1023819
I0206 18:33:34.443420 139806330058560 inference.py:559] [cl 0] Iteration #: 2400, Segmented Voxels: 1091043
/home/morganlab/PycharmProjects/FloodFillNetwork-Notes/ffn/inference/resegmentation.py:256: RuntimeWarning: invalid value encountered in greater_equal
  segmented_voxels = np.nansum((crop_prob >= options.segment_threshold) &
I0206 18:33:36.594700 139806330058560 resegmentation.py:212] processing object 1
I0206 18:33:37.896921 139806330058560 inference.py:559] [cl 0] EDT computation done
I0206 18:33:37.903310 139806330058560 inference.py:559] [cl 0] .. starting segmentation at (xyz): 212 193 17
I0206 18:33:41.552881 139806330058560 inference.py:559] [cl 0] Iteration #: 100, Segmented Voxels: 195243
I0206 18:33:45.251270 139806330058560 inference.py:559] [cl 0] Iteration #: 200, Segmented Voxels: 260936
I0206 18:33:48.952684 139806330058560 inference.py:559] [cl 0] Iteration #: 300, Segmented Voxels: 318202
I0206 18:33:52.889891 139806330058560 inference.py:559] [cl 0] Iteration #: 400, Segmented Voxels: 381532
I0206 18:33:57.027353 139806330058560 inference.py:559] [cl 0] Iteration #: 500, Segmented Voxels: 449131
I0206 18:34:00.712187 139806330058560 inference.py:559] [cl 0] Iteration #: 600, Segmented Voxels: 504687
I0206 18:34:04.492268 139806330058560 inference.py:559] [cl 0] Iteration #: 700, Segmented Voxels: 542135
I0206 18:34:08.284849 139806330058560 inference.py:559] [cl 0] Iteration #: 800, Segmented Voxels: 579431
I0206 18:34:12.100232 139806330058560 inference.py:559] [cl 0] Iteration #: 900, Segmented Voxels: 625836
I0206 18:34:15.917274 139806330058560 inference.py:559] [cl 0] Iteration #: 1000, Segmented Voxels: 628852
I0206 18:34:19.583982 139806330058560 inference.py:559] [cl 0] Iteration #: 1100, Segmented Voxels: 661594
I0206 18:34:23.348087 139806330058560 inference.py:559] [cl 0] Iteration #: 1200, Segmented Voxels: 662423
I0206 18:34:27.066484 139806330058560 inference.py:559] [cl 0] Iteration #: 1300, Segmented Voxels: 688875
I0206 18:34:30.799643 139806330058560 inference.py:559] [cl 0] Iteration #: 1400, Segmented Voxels: 702507
I0206 18:34:34.639780 139806330058560 inference.py:559] [cl 0] Iteration #: 1500, Segmented Voxels: 731850
I0206 18:34:38.368188 139806330058560 inference.py:559] [cl 0] Iteration #: 1600, Segmented Voxels: 744189
I0206 18:34:42.081923 139806330058560 inference.py:559] [cl 0] Iteration #: 1700, Segmented Voxels: 782433
I0206 18:34:45.784733 139806330058560 inference.py:559] [cl 0] Iteration #: 1800, Segmented Voxels: 816755
I0206 18:34:49.447582 139806330058560 inference.py:559] [cl 0] Iteration #: 1900, Segmented Voxels: 851787
I0206 18:34:53.101301 139806330058560 inference.py:559] [cl 0] Iteration #: 2000, Segmented Voxels: 886757
I0206 18:34:56.730261 139806330058560 inference.py:559] [cl 0] Iteration #: 2100, Segmented Voxels: 923218
I0206 18:35:00.417783 139806330058560 inference.py:559] [cl 0] Iteration #: 2200, Segmented Voxels: 947516
I0206 18:35:04.045861 139806330058560 inference.py:559] [cl 0] Iteration #: 2300, Segmented Voxels: 964339
I0206 18:35:07.680531 139806330058560 inference.py:559] [cl 0] Iteration #: 2400, Segmented Voxels: 977303
I0206 18:35:11.360903 139806330058560 inference.py:559] [cl 0] Iteration #: 2500, Segmented Voxels: 981504
I0206 18:35:15.026548 139806330058560 inference.py:559] [cl 0] Iteration #: 2600, Segmented Voxels: 1006441
I0206 18:35:18.677943 139806330058560 inference.py:559] [cl 0] Iteration #: 2700, Segmented Voxels: 1039574
I0206 18:35:22.364867 139806330058560 inference.py:559] [cl 0] Iteration #: 2800, Segmented Voxels: 1041076
I0206 18:35:26.065589 139806330058560 inference.py:559] [cl 0] Iteration #: 2900, Segmented Voxels: 1083364
I0206 18:35:29.726141 139806330058560 inference.py:559] [cl 0] Iteration #: 3000, Segmented Voxels: 1093235
I0206 18:35:33.371828 139806330058560 inference.py:559] [cl 0] Iteration #: 3100, Segmented Voxels: 1109261
I0206 18:35:37.099467 139806330058560 inference.py:559] [cl 0] Iteration #: 3200, Segmented Voxels: 1145835
I0206 18:35:38.338817 139806330058560 inference.py:559] [cl 0] saving results to /home/morganlab/Documents/Autoseg_result/Autoseg_exp7/reseg/120-1279_at_672_582_92.npz
I0206 18:35:38.864521 139806330058560 inference.py:559] [cl 0] .. save complete
I0206 18:35:38.864814 139806330058560 inference.py:305] Deregistering client 0
I0206 18:35:38.872232 139797045683968 executor.py:200] client 0 terminating
I0206 18:35:38.878596 139806330058560 resegmentation.py:309] processing 1/2
I0206 18:35:38.879193 139806330058560 inference.py:1011] Process subvolume: array([ 72, 364, 489])
I0206 18:35:38.879605 139806330058560 inference.py:1029] Requested bounds are array([ 72, 364, 489]) + array([ 41, 401, 401])
I0206 18:35:38.879776 139806330058560 inference.py:1030] Destination bounds are array([ 72, 364, 489]) + array([ 41, 401, 401])
I0206 18:35:38.879914 139806330058560 inference.py:1031] Fetch bounds are array([ 72, 364, 489]) + array([ 41, 401, 401])
I0206 18:35:38.887682 139806330058560 inference.py:1046] Fetched image of size (41, 401, 401) prior to transform
I0206 18:35:38.887874 139806330058560 inference.py:1056] Image data loaded, shape: (41, 401, 401).
I0206 18:35:38.903476 139806330058560 inference.py:301] Registered as client 0.
I0206 18:35:38.903738 139797045683968 executor.py:198] client 0 starting
I0206 18:35:38.903888 139806330058560 inference.py:559] [cl 0] Loading initial segmentation from (zyx) array([ 72, 364, 489]):array([113, 765, 890])
I0206 18:35:39.367937 139806330058560 inference.py:559] [cl 0] Segmentation loaded, shape: (41, 401, 401). Canvas segmentation is (41, 401, 401)
I0206 18:35:39.368168 139806330058560 inference.py:559] [cl 0] Segmentation cropped to: (41, 401, 401)
I0206 18:35:39.401217 139806330058560 inference.py:559] [cl 0] Max restored ID is: 169.
I0206 18:35:39.434974 139806330058560 resegmentation.py:212] processing object 0
I0206 18:35:40.730197 139806330058560 inference.py:559] [cl 0] EDT computation done
I0206 18:35:40.736429 139806330058560 inference.py:559] [cl 0] .. starting segmentation at (xyz): 195 211 17
I0206 18:35:44.455511 139806330058560 inference.py:559] [cl 0] Iteration #: 100, Segmented Voxels: 195214
I0206 18:35:48.192401 139806330058560 inference.py:559] [cl 0] Iteration #: 200, Segmented Voxels: 251386
I0206 18:35:52.051831 139806330058560 inference.py:559] [cl 0] Iteration #: 300, Segmented Voxels: 292887
I0206 18:35:55.997452 139806330058560 inference.py:559] [cl 0] Iteration #: 400, Segmented Voxels: 330098
I0206 18:35:59.647700 139806330058560 inference.py:559] [cl 0] Iteration #: 500, Segmented Voxels: 434297
I0206 18:36:03.333142 139806330058560 inference.py:559] [cl 0] Iteration #: 600, Segmented Voxels: 530689
I0206 18:36:07.026841 139806330058560 inference.py:559] [cl 0] Iteration #: 700, Segmented Voxels: 585259
I0206 18:36:10.694053 139806330058560 inference.py:559] [cl 0] Iteration #: 800, Segmented Voxels: 654152
I0206 18:36:14.382555 139806330058560 inference.py:559] [cl 0] Iteration #: 900, Segmented Voxels: 701054
I0206 18:36:18.068037 139806330058560 inference.py:559] [cl 0] Iteration #: 1000, Segmented Voxels: 738034
I0206 18:36:21.877722 139806330058560 inference.py:559] [cl 0] Iteration #: 1100, Segmented Voxels: 768872
I0206 18:36:25.645387 139806330058560 inference.py:559] [cl 0] Iteration #: 1200, Segmented Voxels: 785271
I0206 18:36:29.366058 139806330058560 inference.py:559] [cl 0] Iteration #: 1300, Segmented Voxels: 811875
I0206 18:36:33.126028 139806330058560 inference.py:559] [cl 0] Iteration #: 1400, Segmented Voxels: 838952
I0206 18:36:36.952694 139806330058560 inference.py:559] [cl 0] Iteration #: 1500, Segmented Voxels: 879850
I0206 18:36:40.765382 139806330058560 inference.py:559] [cl 0] Iteration #: 1600, Segmented Voxels: 919767
I0206 18:36:44.506174 139806330058560 inference.py:559] [cl 0] Iteration #: 1700, Segmented Voxels: 945769
I0206 18:36:48.288006 139806330058560 inference.py:559] [cl 0] Iteration #: 1800, Segmented Voxels: 976492
I0206 18:36:52.050613 139806330058560 inference.py:559] [cl 0] Iteration #: 1900, Segmented Voxels: 1013283
I0206 18:36:55.743161 139806330058560 inference.py:559] [cl 0] Iteration #: 2000, Segmented Voxels: 1031615
I0206 18:36:59.395627 139806330058560 inference.py:559] [cl 0] Iteration #: 2100, Segmented Voxels: 1060529
I0206 18:37:03.040572 139806330058560 inference.py:559] [cl 0] Iteration #: 2200, Segmented Voxels: 1134972
I0206 18:37:04.475396 139806330058560 resegmentation.py:212] processing object 1
I0206 18:37:05.784681 139806330058560 inference.py:559] [cl 0] EDT computation done
I0206 18:37:05.790725 139806330058560 inference.py:559] [cl 0] .. starting segmentation at (xyz): 197 191 14
I0206 18:37:09.458373 139806330058560 inference.py:559] [cl 0] Iteration #: 100, Segmented Voxels: 191989
I0206 18:37:13.147021 139806330058560 inference.py:559] [cl 0] Iteration #: 200, Segmented Voxels: 246861
I0206 18:37:16.838764 139806330058560 inference.py:559] [cl 0] Iteration #: 300, Segmented Voxels: 318094
I0206 18:37:20.534349 139806330058560 inference.py:559] [cl 0] Iteration #: 400, Segmented Voxels: 394901
I0206 18:37:24.305977 139806330058560 inference.py:559] [cl 0] Iteration #: 500, Segmented Voxels: 451629
I0206 18:37:28.027674 139806330058560 inference.py:559] [cl 0] Iteration #: 600, Segmented Voxels: 518831
I0206 18:37:31.799351 139806330058560 inference.py:559] [cl 0] Iteration #: 700, Segmented Voxels: 549045
I0206 18:37:35.549641 139806330058560 inference.py:559] [cl 0] Iteration #: 800, Segmented Voxels: 583314
I0206 18:37:39.239378 139806330058560 inference.py:559] [cl 0] Iteration #: 900, Segmented Voxels: 645962
I0206 18:37:42.967864 139806330058560 inference.py:559] [cl 0] Iteration #: 1000, Segmented Voxels: 695797
I0206 18:37:46.704822 139806330058560 inference.py:559] [cl 0] Iteration #: 1100, Segmented Voxels: 762824
I0206 18:37:50.432867 139806330058560 inference.py:559] [cl 0] Iteration #: 1200, Segmented Voxels: 770380
I0206 18:37:54.129934 139806330058560 inference.py:559] [cl 0] Iteration #: 1300, Segmented Voxels: 780273
I0206 18:37:57.816927 139806330058560 inference.py:559] [cl 0] Iteration #: 1400, Segmented Voxels: 799492
I0206 18:38:01.570694 139806330058560 inference.py:559] [cl 0] Iteration #: 1500, Segmented Voxels: 810775
I0206 18:38:05.300662 139806330058560 inference.py:559] [cl 0] Iteration #: 1600, Segmented Voxels: 816133
I0206 18:38:09.105823 139806330058560 inference.py:559] [cl 0] Iteration #: 1700, Segmented Voxels: 787169
I0206 18:38:10.061874 139806330058560 inference.py:559] [cl 0] saving results to /home/morganlab/Documents/Autoseg_result/Autoseg_exp7/reseg/1279-1235_at_689_564_92.npz
I0206 18:38:10.520101 139806330058560 inference.py:559] [cl 0] .. save complete
I0206 18:38:10.520249 139806330058560 inference.py:305] Deregistering client 0
I0206 18:38:10.527390 139797045683968 executor.py:200] client 0 terminating
```

Sample output protobuf! 
```protobuf
point {
  x: 672
  y: 582
  z: 92
}
id_a: 120
id_b: 1279
segmentation_radius {
  x: 200
  y: 200
  z: 20
}
eval {
  radius {
    x: 200
    y: 200
    z: 20
  }
  iou: 0.1778547167778015
  from_a {
    origin {
      x: 737
      y: 603
      z: 83
    }
    num_voxels: 1140842
    deleted_voxels: 116363
    segment_a_consistency: 0.8300750255584717
    segment_b_consistency: 0.9587254524230957
    max_edt: 292.0616455078125
  }
  from_b {
    origin {
      x: 684
      y: 575
      z: 89
    }
    num_voxels: 1179308
    deleted_voxels: 143220
    segment_a_consistency: 0.8942198753356934
    segment_b_consistency: 0.9706785678863525
    max_edt: 260.2229919433594
  }
  max_edt_a: 150.2131805419922
  max_edt_b: 117.72850036621094
  num_voxels_a: 190896
  num_voxels_b: 30285
}
```

## Neuroglancer view
Much faster and much better to visualize and examine neuron!!!!
```json
{
  "layers": [
    {
      "source": "python://150f129bda96d98541dc64e4ae76e73a253a11da.9f56c8c1868ccd7752c32d6ca8117fcceb8619c6",
      "type": "segmentation",
      "selectedAlpha": 0.05,
      "saturation": 0.81,
      "segments": [
        "11720",
        "166055",
        "181071",
        "300398",
        "314827",
        "394367",
        "449220",
        "545077",
        "545620",
        "545642"
      ],
      "name": "Consensus_segment"
    }
  ],
  "navigation": {
    "pose": {
      "position": {
        "voxelSize": [
          8,
          8,
          40
        ],
        "voxelCoordinates": [
          516.708984375,
          706.7274780273438,
          114.35738372802734
        ]
      }
    },
    "zoomFactor": 8
  },
  "perspectiveOrientation": [
    0.8222200870513916,
    -0.23166555166244507,
    0.48113077878952026,
    0.19697305560112
  ],
  "perspectiveZoom": 117.91924196067077,
  "selectedLayer": {
    "layer": "Consensus_segment",
    "visible": true
  },
  "layout": "4panel"
}
```
```
point {
  x: 689
  y: 564
  z: 92
}
id_a: 1279
id_b: 1235
segmentation_radius {
  x: 200
  y: 200
  z: 20
}
eval {
  radius {
    x: 100
    y: 100
    z: 20
  }
  iou: 0.5057328939437866
  from_a {
    origin {
      x: 684
      y: 575
      z: 89
    }
    num_voxels: 633969
    deleted_voxels: 70039
    segment_a_consistency: 0.9667491912841797
    segment_b_consistency: 0.9637969732284546
    max_edt: 256.0
  }
  from_b {
    origin {
      x: 686
      y: 555
      z: 86
    }
    num_voxels: 633441
    deleted_voxels: 77611
    segment_a_consistency: 0.9596499800682068
    segment_b_consistency: 0.999223530292511
    max_edt: 211.36697387695312
  }
  max_edt_a: 117.72850036621094
  max_edt_b: 110.86929321289062
  num_voxels_a: 30285
  num_voxels_b: 10303
}
```
## New Tissue segmentation stiching trial
14512, 14392
12464, 16560
12344, 16440



Pipeline

## Change seed policy and add reverse seed order
```bash
source ~/virtenvs/test1/bin/activate
export Request='image {
 hdf5: "/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/grayscale_maps_LR.h5:raw"
}
image_mean: 128
image_stddev: 33
checkpoint_interval: 1200
seed_policy: "PolicyPeaks2d"
seed_policy_args: "{ \"sort_cmp\": \"descending\"}"
model_checkpoint_path: "/home/morganlab/Downloads/ffn-master/models/LR_model_Longtime/model.ckpt-5339851"
model_name: "convstack_3d.ConvStack3DFFNModel"
model_args: "{\"depth\": 9, \"fov_size\": [55, 37, 17], \"deltas\": [9,6,3]}"
segmentation_output_dir: "/home/morganlab/Downloads/ffn-master/results/LGN/tmp2"
inference_options {
  init_activation: 0.95
  pad_value: 0.05
  move_threshold: 0.95
  min_boundary_dist { x: 3 y: 3 z: 1}
  segment_threshold: 0.5
  min_segment_size: 1000
}'
cd PycharmProjects/FloodFillNetwork-Notes/
python3 run_inference.py \
  --inference_request="$Request" \
  --bounding_box 'start { x:0 y:0 z:0 } size { x:1180 y:1058 z:175 }' 
```
```bash
export Request='image {
 hdf5: "/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/grayscale_maps_LR.h5:raw"
}
image_mean: 128
image_stddev: 33
checkpoint_interval: 1200
seed_policy: "PolicyPeaks"
seed_policy_args: "{ \"reverse\": 1}"
model_checkpoint_path: "/home/morganlab/Downloads/ffn-master/models/LR_model_Longtime/model.ckpt-5339851"
model_name: "convstack_3d.ConvStack3DFFNModel"
model_args: "{\"depth\": 9, \"fov_size\": [55, 37, 17], \"deltas\": [9,6,3]}"
segmentation_output_dir: "/home/morganlab/Downloads/ffn-master/results/LGN/tmp2"
inference_options {
  init_activation: 0.95
  pad_value: 0.05
  move_threshold: 0.95
  min_boundary_dist { x: 3 y: 3 z: 1}
  segment_threshold: 0.5
  min_segment_size: 1000
}'
```

Not really successful!! Lots of spill over and stuff


model.ckpt-5339851 is really  a disaster!!! It has too much spill over!!!! Cannot use! 

branch_upsp, the problem is many seeds are weak but not rejected. Results in a very fragmented segmentation 

```bash
python3 train.py \
--train_coords /home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/tf_record_file_LR \
--data_volumes LGN_LR:/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/grayscale_maps_LR.h5:raw \
--label_volumes LGN_LR:/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/groundtruth_LR.h5:stack \
--train_dir /home/morganlab/Downloads/ffn-master/models/LR_model_Longtime \
--model_name convstack_3d.ConvStack3DFFNModel \
--model_args "{\"depth\": 9, \"fov_size\": [55, 37, 17], \"deltas\": [9,6,3]}" \
--image_mean 138 \
--image_stddev 55 \
--permutable_axes 0
```

## Resegment and Agglomeration 

12 CPUs chunksize=4 15:40 start 
~ 2G memory
No memory leakage!
~ 13min or so 

Consistently, the `imap` works for testing_LR volume without increase in memory, even with 24 process running the code. 
but does not in testing_exp12, exp12 will result in serious memory problem 


Use imap 24 core  chunksize=1, start 18min total


/usr/bin/python3.6 /home/morganlab/pycharm-2018.3.1/helpers/pydev/pydevconsole.py --mode=client --port=41871
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/home/morganlab/Documents/neuroglancer', '/home/morganlab/PycharmProjects/FloodFillNetwork-Notes'])
Python 3.6.7 (default, Oct 22 2018, 11:32:17) 
Type "copyright", "credits" or "license" for more information.
IPython 5.5.0 -- An enhanced Interactive Python.
?         -> Introduction and overview of IPython's features.
%quickref -> Quick reference.
help      -> Python's own help system.
object?   -> Details about 'object', use 'object??' for extra details.
PyDev console: using IPython 5.5.0
Python 3.6.7 (default, Oct 22 2018, 11:32:17) 
[GCC 8.2.0] on linux
runfile('/home/morganlab/PycharmProjects/FloodFillNetwork-Notes/resegment_seed_generation_ED.py', wdir='/home/morganlab/PycharmProjects/FloodFillNetwork-Notes/analysis_script')
[17983] At start Memory usage: 1248052 (kb)
[17983] After cast type Memory usage: 2373812 (kb)
Pairs to process 10087.
[17983] Before starting Pool Memory usage: 5688284 (kb)
[17983] Before writing down Memory usage: 5688284 (kb)
[18087] After calculate segment Memory usage: 3938496 (kb)
[18088] After calculate segment Memory usage: 3938576 (kb)
[18088] After fetching coordinates Memory usage: 3940044 (kb)
[18087] After fetching coordinates Memory usage: 3938960 (kb)
[18085] After calculate segment Memory usage: 3938524 (kb)
[18086] After calculate segment Memory usage: 3938560 (kb)
[18087] After dist_mat Memory usage: 4597812 (kb)
{id_a:18 id_b:395 point {[array([ 13, 109, 277])]} } min dist 8.0 
[18087] After calculate segment Memory usage: 4598152 (kb)
[18085] After fetching coordinates Memory usage: 3939468 (kb)
[18086] After fetching coordinates Memory usage: 4184636 (kb)
[18086] After calculate segment Memory usage: 4184636 (kb)
[18087] After fetching coordinates Memory usage: 4598152 (kb)
[18087] After calculate segment Memory usage: 4598152 (kb)
[18085] After dist_mat Memory usage: 5024656 (kb)
{id_a:1 id_b:3 point {[array([  0,   3, 675])]} } min dist 8.0 
[18086] After fetching coordinates Memory usage: 4184636 (kb)
[18087] After fetching coordinates Memory usage: 4598152 (kb)
[18085] After calculate segment Memory usage: 5024808 (kb)
[18085] After fetching coordinates Memory usage: 5024808 (kb)
[18088] After dist_mat Memory usage: 8090784 (kb)
{id_a:27 id_b:258 point {[array([   0,  138, 1113])]} } min dist 24.0 
[18088] After calculate segment Memory usage: 8091016 (kb)
[18088] After fetching coordinates Memory usage: 8091016 (kb)
[18086] After dist_mat Memory usage: 35733208 (kb)
{id_a:33 id_b:523 point {[array([ 31, 176,  86])]} } min dist 30.0 
[18086] After calculate segment Memory usage: 35733468 (kb)
[18086] After fetching coordinates Memory usage: 35733468 (kb)
[18185] After calculate segment Memory usage: 3938592 (kb)
[18185] After fetching coordinates Memory usage: 3965780 (kb)
[18088] After dist_mat Memory usage: 29290416 (kb)
[18085] After dist_mat Memory usage: 24712160 (kb)
{id_a:1 id_b:7 point {[array([  6,  15, 703])]} } min dist 8.0 
{id_a:27 id_b:411 point {[array([   6,  151, 1080])]} } min dist 8.0 
[18085] After calculate segment Memory usage: 24712160 (kb)
[18088] After calculate segment Memory usage: 29290416 (kb)
[18085] After fetching coordinates Memory usage: 24712160 (kb)
[18088] After fetching coordinates Memory usage: 29290416 (kb)
[18085] After dist_mat Memory usage: 24712160 (kb)
{id_a:1 id_b:410 point {[array([  6,   5, 682])]} } min dist 8.0 
[18085] After calculate segment Memory usage: 24712160 (kb)
[18085] After fetching coordinates Memory usage: 24712160 (kb)
[18085] After dist_mat Memory usage: 24712160 (kb)
[18088] After dist_mat Memory usage: 29290416 (kb)
{id_a:2 id_b:3 point {[array([  0,   2, 602])]} } min dist 8.0 
[18085] After calculate segment Memory usage: 24712160 (kb)
{id_a:27 id_b:425 point {[array([  29,  109, 1067])]} } min dist 26.8 
[18085] After fetching coordinates Memory usage: 24712160 (kb)
[18088] After calculate segment Memory usage: 29290416 (kb)
[18088] After fetching coordinates Memory usage: 29290416 (kb)
[18185] After dist_mat Memory usage: 30764180 (kb)
{id_a:39 id_b:713 point {[array([ 49, 142, 539])]} } min dist 32.3 
[18088] After dist_mat Memory usage: 29290416 (kb)
{id_a:27 id_b:563 point {[array([  36,  114, 1059])]} } min dist 16.0 
[18088] After calculate segment Memory usage: 29290416 (kb)
[18185] After calculate segment Memory usage: 30764368 (kb)
[18088] After fetching coordinates Memory usage: 29290416 (kb)
[18185] After fetching coordinates Memory usage: 30764368 (kb)
[18185] After calculate segment Memory usage: 30764368 (kb)
[18185] After fetching coordinates Memory usage: 30764368 (kb)
[18085] After dist_mat Memory usage: 24712160 (kb)
{id_a:2 id_b:410 point {[array([  0,   2, 570])]} } min dist 8.0 
[18085] After calculate segment Memory usage: 24712160 (kb)
[18085] After fetching coordinates Memory usage: 24712160 (kb)
multiprocessing.pool.RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.6/multiprocessing/pool.py", line 119, in worker
    result = (True, func(*args, **kwds))
  File "/usr/lib/python3.6/multiprocessing/pool.py", line 44, in mapstar
    return list(map(*args))
  File "/home/morganlab/PycharmProjects/FloodFillNetwork-Notes/resegment_seed_generation_ED.py", line 91, in worker_func
    dist_mat = np.zeros((coord_a.shape[1], coord_b.shape[1]))
MemoryError
"""
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
  File "/usr/lib/python3/dist-packages/IPython/core/interactiveshell.py", line 2882, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "<ipython-input-2-d24af42ab27a>", line 1, in <module>
    runfile('/home/morganlab/PycharmProjects/FloodFillNetwork-Notes/resegment_seed_generation_ED.py', wdir='/home/morganlab/PycharmProjects/FloodFillNetwork-Notes/analysis_script')
  File "/home/morganlab/pycharm-2018.3.1/helpers/pydev/_pydev_bundle/pydev_umd.py", line 198, in runfile
    pydev_imports.execfile(filename, global_vars, local_vars)  # execute the script
  File "/home/morganlab/pycharm-2018.3.1/helpers/pydev/_pydev_imps/_pydev_execfile.py", line 18, in execfile
    exec(compile(contents+"\n", file, 'exec'), glob, loc)
  File "/home/morganlab/PycharmProjects/FloodFillNetwork-Notes/resegment_seed_generation_ED.py", line 187, in <module>
    for result_vec, id_pair in zip(result, pair_list):
  File "/usr/lib/python3.6/multiprocessing/pool.py", line 342, in <genexpr>
    return (item for chunk in result for item in chunk)
  File "/usr/lib/python3.6/multiprocessing/pool.py", line 761, in next
    raise value
MemoryError
[18085] After calculate segment Memory usage: 24712160 (kb)
[18085] After fetching coordinates Memory usage: 24712160 (kb)
[18357] After calculate segment Memory usage: 3938988 (kb)
[18088] After dist_mat Memory usage: 29290416 (kb)
[18357] After fetching coordinates Memory usage: 4046440 (kb)
[18357] After calculate segment Memory usage: 4046440 (kb)
{id_a:27 id_b:886 point {[array([  70,  130, 1054])]} } min dist 8.0 
[18357] After fetching coordinates Memory usage: 4046440 (kb)
[18088] After calculate segment Memory usage: 29290416 (kb)
[18088] After fetching coordinates Memory usage: 29290416 (kb)
[18088] After dist_mat Memory usage: 29290416 (kb)
{id_a:27 id_b:932 point {[array([  76,  141, 1075])]} } min dist 28.8 
[18088] After calculate segment Memory usage: 29290416 (kb)
[18088] After fetching coordinates Memory usage: 29290416 (kb)
[18088] After dist_mat Memory usage: 29290416 (kb)
{id_a:27 id_b:972 point {[array([  88,  134, 1078])]} } min dist 31.0 
[18088] After calculate segment Memory usage: 29290416 (kb)
[18088] After fetching coordinates Memory usage: 29290416 (kb)
Process finished with exit code 0






/usr/bin/python3.6 /home/morganlab/pycharm-2018.3.1/helpers/pydev/pydevconsole.py --mode=client --port=45531
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/home/morganlab/Documents/neuroglancer', '/home/morganlab/PycharmProjects/FloodFillNetwork-Notes'])
Python 3.6.7 (default, Oct 22 2018, 11:32:17) 
Type "copyright", "credits" or "license" for more information.
IPython 5.5.0 -- An enhanced Interactive Python.
?         -> Introduction and overview of IPython's features.
%quickref -> Quick reference.
help      -> Python's own help system.
object?   -> Details about 'object', use 'object??' for extra details.
PyDev console: using IPython 5.5.0
Python 3.6.7 (default, Oct 22 2018, 11:32:17) 
[GCC 8.2.0] on linux
runfile('/home/morganlab/PycharmProjects/FloodFillNetwork-Notes/resegment_seed_generation_ED.py', wdir='/home/morganlab/PycharmProjects/FloodFillNetwork-Notes/analysis_script')
[18981] At start Memory usage: 1255496 (kb)
[18981] After cast type Memory usage: 2372384 (kb)
Pairs to process 10087.
[18981] Before starting Pool Memory usage: 5686664 (kb)
[18981] Before writing down Memory usage: 5686664 (kb)
[19206] After calculate segment Memory usage: 3938452 (kb)
[19207] After calculate segment Memory usage: 3938348 (kb)
[19206] After fetching coordinates Memory usage: 3938788 (kb)
[19207] After fetching coordinates Memory usage: 3939904 (kb)
[19205] After calculate segment Memory usage: 3938484 (kb)
[19204] After calculate segment Memory usage: 3938484 (kb)
[19206] After dist_mat Memory usage: 4597760 (kb)
[19204] After fetching coordinates Memory usage: 3939148 (kb)
[19206] Before printing Memory usage: 4597760 (kb)
[19205] After fetching coordinates Memory usage: 4184532 (kb)
[19206] After calculate segment Memory usage: 4597760 (kb)
[19205] After calculate segment Memory usage: 4184532 (kb)
[19204] After dist_mat Memory usage: 5024508 (kb)
[19206] After fetching coordinates Memory usage: 4597760 (kb)
[19204] Before printing Memory usage: 5024508 (kb)
multiprocessing.pool.RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.6/multiprocessing/pool.py", line 119, in worker
    result = (True, func(*args, **kwds))
  File "/usr/lib/python3.6/multiprocessing/pool.py", line 44, in mapstar
    return list(map(*args))
  File "/home/morganlab/PycharmProjects/FloodFillNetwork-Notes/resegment_seed_generation_ED.py", line 108, in worker_func
    print("{id_a:%d id_b:%d point {%s} } min dist %.1f \n" % (cur_idx1, cur_idx2, str(com_vec), dist_mat[i, j]))
UnboundLocalError: local variable 'dist_mat' referenced before assignment
"""
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
  File "/usr/lib/python3/dist-packages/IPython/core/interactiveshell.py", line 2882, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "<ipython-input-2-d24af42ab27a>", line 1, in <module>
    runfile('/home/morganlab/PycharmProjects/FloodFillNetwork-Notes/resegment_seed_generation_ED.py', wdir='/home/morganlab/PycharmProjects/FloodFillNetwork-Notes/analysis_script')
  File "/home/morganlab/pycharm-2018.3.1/helpers/pydev/_pydev_bundle/pydev_umd.py", line 198, in runfile
    pydev_imports.execfile(filename, global_vars, local_vars)  # execute the script
  File "/home/morganlab/pycharm-2018.3.1/helpers/pydev/_pydev_imps/_pydev_execfile.py", line 18, in execfile
    exec(compile(contents+"\n", file, 'exec'), glob, loc)
  File "/home/morganlab/PycharmProjects/FloodFillNetwork-Notes/resegment_seed_generation_ED.py", line 191, in <module>
    for result_vec, id_pair in zip(result, pair_list):
  File "/usr/lib/python3.6/multiprocessing/pool.py", line 342, in <genexpr>
    return (item for chunk in result for item in chunk)
  File "/usr/lib/python3.6/multiprocessing/pool.py", line 761, in next
    raise value
UnboundLocalError: local variable 'dist_mat' referenced before assignment
[19205] After fetching coordinates Memory usage: 4184532 (kb)
[19204] After calculate segment Memory usage: 5024508 (kb)
[19204] After fetching coordinates Memory usage: 5024508 (kb)
[19207] After dist_mat Memory usage: 8090612 (kb)
[19207] Before printing Memory usage: 8090768 (kb)
[19207] After calculate segment Memory usage: 8090768 (kb)
[19207] After fetching coordinates Memory usage: 8090768 (kb)
[19204] After dist_mat Memory usage: 30764892 (kb)
[19204] Before printing Memory usage: 30764976 (kb)
[19440] After calculate segment Memory usage: 3938784 (kb)
[19440] After fetching coordinates Memory usage: 4539736 (kb)
[19440] After calculate segment Memory usage: 4539736 (kb)
[19440] After fetching coordinates Memory usage: 4539736 (kb)
[19444] After calculate segment Memory usage: 3938832 (kb)
[19444] After fetching coordinates Memory usage: 3970172 (kb)







12 core, takes 30 min to process > 2400 points

Approximately 3-4 hours for all segments points

24 core 30 min only half the points 

## Make new well aligned (affine align with SIFT)

p11_4
Overall coordinate 
14400, 15072
20328, 10160

After alignment and cropping
size of volume 
( 2414, 2222, 152) 

Note if the transform is not recorded, then the inverse transform is too hard to obtain. 
And the segement done on local subvolumes will be unable to mapped back to total volume


I0212 19:43:59.986612 139901798192960 resegmentation_analysis.py:93] processing: /home/morganlab/Downloads/ffn-master/results/LGN/testing_exp12/reseg/21-547_at_870_167_64.npz
Traceback (most recent call last):
  File "/usr/lib/python3/dist-packages/IPython/core/interactiveshell.py", line 2882, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "<ipython-input-54-b80e88eabae5>", line 7, in <module>
    seg, reseg_r_zyx, analysis_r_zyx, sampling=voxelsize_zyx)
  File "/home/morganlab/PycharmProjects/FloodFillNetwork-Notes/ffn/inference/resegmentation_analysis.py", line 199, in evaluate_pair_resegmentation
    prob = storage.dequantize_probability(data['probs'])
  File "/home/morganlab/.local/lib/python3.6/site-packages/numpy/lib/npyio.py", line 251, in __getitem__
    bytes = self.zip.open(key)
  File "/usr/lib/python3.6/zipfile.py", line 1396, in open
    raise BadZipFile("Bad magic number for file header")
zipfile.BadZipFile: Bad magic number for file header

processing and evaluation is quick 10 min 3000 point pairs


## New volume p11_5

Extract from VAST: IxD_W002_invert2.vsv
11184, 22256
24656, 31808

Use Fiji to Align!



Use Python script to normalize and stitch into a h5 fi

mean: 138.36, std: 37.90



## Training tissue classififer


Training Window 
7000, 12864
13280, 17744

X 3140, Y 2440!, Z 152

Test Window
16378, 17464
24987, 21747
Try network on 
Mip level1 
Mip level 3



FoV
256 256 
128 128


## Tissue type classification 


Training 65, 65
Inference 

65 
(3,3)
63
(2,2)
31
(3,3)
29
(2,2)
14
(3,3)
12
(2,2)
6
(3,3)
4

## new volume  p11_6 
Extract from VAST: IxD_W002_invert2.vsv
extracted from VAST
8704, 17008
18320, 22144

4468, 2373


## New volume Kasthuri 11 Opendataset

Make the config file and run this download code
```
[Default]
protocol = https
host = api.boss.neurodata.io
token = e300e909333dc834e2c7f1dac87a13e83e907234
```
```bash
ndpull --config_file neurodata.cfg --collection  kasthuri  --experiment  kasthuri11 --channel image --x 4000 14000 --y 15000 20000 --z 500 900 --outdir .
```

use Fiji to preprocess downscaling, use python code to convert that into h5 volume. Examine with 

Image stack shape: 
(400, 2500, 5000)
mean: 175.88, std: 47.30
max: 260.85, min: -93.39 after scaling
mean: 147.48, std: 38.87

After screening the left model is 
Exp1: Clear boundary, too much merge, small fibers are good! 
Exp2: 
exp3: Blurry boundary! not clear! hard to use! 

Longtime_SF_Deep/model.ckpt-15005144
exp1-2: Many merge of large branches! Fibers are good 

Longtime/model.ckpt-14501471
exp1-18 : Large branches are clear and distinct ! lose some fibers and some not solid filling ! 
are good points! Their consensus is quite good ! 18 is quite good at decerning large boundaries, 2 is good at small fibers quite good match ! 

## IPL image stack! 
For un-aligned dataset
Image stack shape: 
(78, 4839, 5146)
mean: 131.75, std: 48.68
max: 259.49, min: -50.65 after scaling
mean: 133.48, std: 38.16


For aligned dataset
Image stack shape: 
(78, 4839, 5146)
mean: 131.17, std: 48.65
max: 260.86, min: -54.71 after scaling
mean: 132.87, std: 38.63

While read loop


2371, 1831, 
1051, 3374
## Visualizations 
p11_6 Json
{
  "layers": [
    {
      "source": "python://1e832ae585cfa43d0a7e021cf7c2f314e2ad4eae.74441e86194596ee862e2a8967e17b1232445e0a",
      "type": "segmentation",
      "segments": [
        "10169",
        "10197",
        "10198",
        "10630",
        "10631",
        "11566",
        "116",
        "11793",
        "12620",
        "1276",
        "13107",
        "1311",
        "13116",
        "1314",
        "1315",
        "1318",
        "1332",
        "13328",
        "13329",
        "13331",
        "13335",
        "1337",
        "13402",
        "1342",
        "1364",
        "13645",
        "13690",
        "13835",
        "1412",
        "14127",
        "14200",
        "14228",
        "1446",
        "15627",
        "16266",
        "16320",
        "16677",
        "16740",
        "16853",
        "16859",
        "17032",
        "17111",
        "17693",
        "1821",
        "18613",
        "1953",
        "1976",
        "19767",
        "1978",
        "19845",
        "2017",
        "20292",
        "20386",
        "20445",
        "20455",
        "20478",
        "20626",
        "20673",
        "20694",
        "20742",
        "20777",
        "20962",
        "20988",
        "21126",
        "21243",
        "23263",
        "23618",
        "23626",
        "24467",
        "25506",
        "25517",
        "25555",
        "25810",
        "2736",
        "27643",
        "28095",
        "28148",
        "29373",
        "29652",
        "30214",
        "30215",
        "31122",
        "31703",
        "32139",
        "32146",
        "32367",
        "32805",
        "32890",
        "33254",
        "33273",
        "33630",
        "33950",
        "43",
        "4438",
        "4721",
        "5002",
        "5019",
        "5056",
        "5063",
        "5111",
        "5158",
        "52",
        "5475",
        "5971",
        "6009",
        "6373",
        "69",
        "711",
        "8498",
        "8760",
        "878",
        "9248",
        "9727",
        "9739"
      ],
      "name": "p11_6_consensus_33_38_full-1"
    },
    {
      "source": "python://1e832ae585cfa43d0a7e021cf7c2f314e2ad4eae.e60f46a751d4e78465a98e1e661c31bd04b6a119",
      "type": "image",
      "opacity": 0.77,
      "name": "EM_image"
    }
  ],
  "navigation": {
    "pose": {
      "position": {
        "voxelSize": [
          8,
          8,
          40
        ],
        "voxelCoordinates": [
          1984.758056640625,
          1273.864990234375,
          22.65532684326172
        ]
      }
    },
    "zoomFactor": 10.427447806226951
  },
  "perspectiveOrientation": [
    0.06947831064462662,
    0.38631299138069153,
    0.9193979501724243,
    0.0253454577177763
  ],
  "perspectiveZoom": 340.3586790717492,
  "showSlices": false,
  "selectedLayer": {
    "layer": "p11_6_consensus_33_38_full-1",
    "visible": true
  },
  "layout": "3d"
}


## IPL Data
{
  "layers": [
    {
      "source": "python://0e39ce5a60b6e5d60e0827f7eda98224f77c2d1c.e2a4e99b65aa043f8df3a67a57aed0a1a8182eaa",
      "type": "segmentation",
      "colorSeed": 2531832856,
      "segments": [
        "10021",
        "10042",
        "10125",
        "10139",
        "10162",
        "10222",
        "10290",
        "10351",
        "10412",
        "10444",
        "10504",
        "10573",
        "10630",
        "10645",
        "10668",
        "10977",
        "11006",
        "11084",
        "11160",
        "11187",
        "11188",
        "11191",
        "11192",
        "11205",
        "11227",
        "11277",
        "11288",
        "11334",
        "11351",
        "11369",
        "11407",
        "11459",
        "11466",
        "12436",
        "12646",
        "14522",
        "15495",
        "15535",
        "15580",
        "15617",
        "15706",
        "15723",
        "15753",
        "15754",
        "15776",
        "15828",
        "16002",
        "16462",
        "16714",
        "16956",
        "16957",
        "16963",
        "16964",
        "16984",
        "17156",
        "17198",
        "17220",
        "17221",
        "17227",
        "17280",
        "17323",
        "17346",
        "17350",
        "17403",
        "17438",
        "17711",
        "17734",
        "17735",
        "18070",
        "18282",
        "18302",
        "19149",
        "19307",
        "19337",
        "19338",
        "21485",
        "22832",
        "23064",
        "23124",
        "23171",
        "23186",
        "24077",
        "24290",
        "24348",
        "24351",
        "24352",
        "24360",
        "24382",
        "24432",
        "24485",
        "24512",
        "24544",
        "24578",
        "24609",
        "24689",
        "24694",
        "24734",
        "24816",
        "24849",
        "24935",
        "24987",
        "25015",
        "25023",
        "25352",
        "25428",
        "25483",
        "25567",
        "25569",
        "25580",
        "25638",
        "25640",
        "25653",
        "25656",
        "25676",
        "2940",
        "31376",
        "31467",
        "31615",
        "32563",
        "3292",
        "3405",
        "3457",
        "3540",
        "3675",
        "4221",
        "4469",
        "4482",
        "4505",
        "4509",
        "4577",
        "4587",
        "4620",
        "4634",
        "4681",
        "4753",
        "479",
        "5498",
        "7546",
        "9967"
      ],
      "name": "IPL_exp1-2_rev_consensus_full-1"
    },
    {
      "source": "python://0e39ce5a60b6e5d60e0827f7eda98224f77c2d1c.0f57a208c2bd931b103d61146f44a649a35d5ee9",
      "type": "image",
      "opacity": 0.81,
      "name": "EM_image"
    }
  ],
  "navigation": {
    "pose": {
      "position": {
        "voxelSize": [
          8,
          8,
          40
        ],
        "voxelCoordinates": [
          2593.770751953125,
          2443.518798828125,
          31.347824096679688
        ]
      }
    },
    "zoomFactor": 13.591458468948415
  },
  "perspectiveOrientation": [
    -0.37206026911735535,
    -0.8213294148445129,
    0.43082988262176514,
    -0.03707733377814293
  ],
  "perspectiveZoom": 340.3586790717492,
  "showSlices": false,
  "selectedLayer": {
    "layer": "EM_image",
    "visible": true
  },
  "layout": "3d"
}



## IPL 
{
  "layers": [
    {
      "source": "python://addbf89eff718c6db5e21e933553dea43c1902f2.be56a13b2b32d85b5a7024b6692b86aaa58c801e",
      "type": "segmentation",
      "selectedAlpha": 1,
      "saturation": 0.56,
      "colorSeed": 2344215094,
      "segments": [
        "10070",
        "10351",
        "11704",
        "11830",
        "12152",
        "12928",
        "13503",
        "1447",
        "1448",
        "1453",
        "17565",
        "18559",
        "18583",
        "18827",
        "19943",
        "21305",
        "26223",
        "4469",
        "5103",
        "7730",
        "9208",
        "9767",
        "9967",
        "9976"
      ],
      "equivalences": [
        [
          "1448",
          "1457",
          "1458",
          "1552",
          "1555"
        ],
        [
          "1453",
          "1516",
          "1774",
          "1870",
          "1871",
          "1904",
          "1936",
          "1996",
          "2059",
          "2394",
          "2487",
          "2727",
          "2728"
        ],
        [
          "3292",
          "8625",
          "8893",
          "8968",
          "8998",
          "9032",
          "9152",
          "9242",
          "9313",
          "9335",
          "9546",
          "9589",
          "9664",
          "9665",
          "9773",
          "9855",
          "9895",
          "10557",
          "10605",
          "10691",
          "10780",
          "10829",
          "11340",
          "11388",
          "11401"
        ],
        [
          "4447",
          "5292",
          "5376",
          "5629",
          "5882",
          "12177",
          "12189",
          "12238",
          "12268",
          "12296",
          "12297",
          "12370",
          "12607"
        ],
        [
          "4469",
          "4482",
          "4505",
          "4509",
          "4577",
          "4587",
          "4620",
          "4634",
          "5455",
          "5460",
          "7546",
          "10169",
          "10272",
          "10291",
          "10392",
          "10446",
          "10546",
          "10549",
          "10645",
          "10753",
          "10955",
          "11063",
          "11084",
          "11160",
          "11161",
          "11222",
          "11234",
          "11235",
          "11282",
          "11289",
          "11299",
          "11326",
          "11350",
          "11351",
          "11459",
          "11462",
          "11466",
          "12433",
          "12436",
          "12646",
          "15495",
          "15776",
          "16130",
          "16462",
          "16647",
          "16759",
          "17737",
          "17928",
          "18148",
          "21267",
          "22229",
          "22644",
          "22652",
          "22653",
          "22832",
          "23189",
          "23379",
          "23744",
          "23757",
          "23990",
          "24022",
          "24042",
          "24048",
          "24063",
          "24126",
          "24165"
        ],
        [
          "5099",
          "5295",
          "5298",
          "5581",
          "5834"
        ],
        [
          "5182",
          "5195",
          "5244",
          "5832",
          "5843",
          "5849"
        ],
        [
          "7730",
          "8617",
          "9079",
          "9761",
          "15836",
          "15852",
          "15873",
          "15887",
          "15903",
          "15983",
          "16460",
          "16549",
          "16659",
          "16774"
        ],
        [
          "8927",
          "11276"
        ],
        [
          "9208",
          "9360",
          "9624",
          "9670",
          "9897",
          "9942",
          "10793",
          "11150"
        ],
        [
          "9967",
          "9985",
          "9989",
          "9990",
          "10222",
          "10334",
          "10362",
          "10390",
          "10412",
          "10448",
          "10504",
          "10830",
          "10895",
          "10922",
          "10961",
          "11022",
          "11277",
          "11290",
          "11334",
          "11408",
          "17317",
          "17323",
          "18026",
          "18081",
          "18082",
          "18282",
          "18294",
          "18295"
        ],
        [
          "9976",
          "9980",
          "10014",
          "10037",
          "10042",
          "10043",
          "10162",
          "10205",
          "10244",
          "10269",
          "10287",
          "10977",
          "10984",
          "11152",
          "11154",
          "11155",
          "11187",
          "11188",
          "11192",
          "11209",
          "11249",
          "11260"
        ],
        [
          "10004",
          "10297",
          "10519",
          "11242",
          "11315"
        ],
        [
          "10070",
          "10143",
          "10246",
          "10365",
          "11010",
          "11179",
          "11206"
        ],
        [
          "10288",
          "17551",
          "18022",
          "18120",
          "18421"
        ],
        [
          "10351",
          "10702",
          "10849",
          "11053",
          "11278",
          "11288",
          "17141",
          "17156",
          "17168",
          "17199",
          "17202",
          "17242",
          "17279",
          "17284",
          "17381",
          "18018",
          "18302"
        ],
        [
          "11511",
          "18629",
          "18731",
          "18752",
          "18767",
          "18769",
          "18870",
          "18922",
          "19024",
          "19052",
          "19085",
          "19197",
          "19198",
          "19208",
          "19242",
          "19243",
          "19325",
          "19380",
          "19458",
          "19521",
          "19565",
          "19593",
          "19674",
          "19677",
          "19704",
          "19785",
          "19824",
          "19850",
          "19851"
        ],
        [
          "11703",
          "11718",
          "13359",
          "13742",
          "13970",
          "13989"
        ],
        [
          "11704",
          "13112",
          "19503",
          "19728",
          "19948",
          "20786",
          "20931",
          "20935",
          "20936"
        ],
        [
          "11830",
          "11856",
          "11975",
          "11995",
          "12539",
          "12559",
          "12735",
          "12749",
          "12777",
          "18732",
          "18844",
          "18845",
          "18906",
          "19500",
          "19570",
          "19740",
          "19748",
          "19768",
          "19782",
          "25896",
          "25900",
          "26927",
          "27112",
          "27168",
          "27179"
        ],
        [
          "12018",
          "13377",
          "13379"
        ],
        [
          "12152",
          "12300",
          "12307",
          "12382",
          "12398",
          "12602",
          "12632",
          "12850",
          "12869",
          "13340",
          "13381",
          "13767"
        ],
        [
          "12928",
          "12956",
          "12957",
          "12970",
          "12971",
          "13025",
          "13059",
          "13068",
          "13093",
          "13117",
          "13152",
          "13198",
          "13224",
          "13689",
          "13728",
          "13736",
          "13755",
          "13842",
          "13854",
          "13880",
          "13883",
          "13894",
          "13906",
          "13919",
          "13920"
        ],
        [
          "12945",
          "13049",
          "13132",
          "13133",
          "13699"
        ],
        [
          "12972",
          "13187",
          "13210",
          "13257",
          "13788"
        ],
        [
          "13092",
          "13150",
          "13179",
          "13235",
          "13735",
          "13760",
          "13917"
        ],
        [
          "13503",
          "17735",
          "19149",
          "19305",
          "19337",
          "19338",
          "19620",
          "19649",
          "19913",
          "20476",
          "20491",
          "20504",
          "20876",
          "20906",
          "21070",
          "21071",
          "21092",
          "21107"
        ],
        [
          "14189",
          "14908",
          "15085",
          "16169",
          "16369",
          "16453",
          "16630",
          "16839",
          "16891",
          "16916"
        ],
        [
          "14809",
          "16394",
          "16608",
          "16879",
          "23233",
          "23320",
          "23563",
          "23823",
          "23922",
          "23928",
          "24086",
          "24087",
          "24197",
          "24198",
          "24932",
          "25497",
          "25789"
        ],
        [
          "15519",
          "15803",
          "15827",
          "15830",
          "15832",
          "15909",
          "16162",
          "16266",
          "16665",
          "16666",
          "16874",
          "17337",
          "18315",
          "23196",
          "23498",
          "23893",
          "24023"
        ],
        [
          "15522",
          "15731",
          "15761",
          "16728",
          "16746",
          "16747",
          "16775"
        ],
        [
          "15533",
          "15544",
          "16476"
        ],
        [
          "15535",
          "15644",
          "15706",
          "15722",
          "16477",
          "16714",
          "16954",
          "16955",
          "16956",
          "16957",
          "16963",
          "16964",
          "24388",
          "24389",
          "24512",
          "24542",
          "24544",
          "24578",
          "24627",
          "25352",
          "25384",
          "25569",
          "25585",
          "25632"
        ],
        [
          "15584",
          "16491",
          "16684",
          "24522",
          "24759",
          "25403",
          "30205",
          "31176",
          "31178",
          "31179",
          "32434"
        ],
        [
          "15794",
          "16755",
          "17276",
          "18271"
        ],
        [
          "16039",
          "16160",
          "16381",
          "16420",
          "16428",
          "16586"
        ],
        [
          "16984",
          "17223",
          "18010"
        ],
        [
          "17036",
          "17060",
          "18229"
        ],
        [
          "17074",
          "24599",
          "25350"
        ],
        [
          "17164",
          "17308",
          "18287"
        ],
        [
          "17240",
          "17281",
          "17333",
          "17347",
          "17393",
          "18073",
          "18267",
          "18277",
          "18279"
        ],
        [
          "17334",
          "17478",
          "17706",
          "18104",
          "24954",
          "25066",
          "25096",
          "25439"
        ],
        [
          "17394",
          "17467",
          "18096",
          "18311",
          "18327"
        ],
        [
          "17558",
          "17593",
          "18361",
          "24910"
        ],
        [
          "17565",
          "17636",
          "17768",
          "17769",
          "17808",
          "17875",
          "18141"
        ],
        [
          "17751",
          "17775",
          "18163",
          "18168"
        ],
        [
          "18088",
          "18293"
        ],
        [
          "18559",
          "18923",
          "19400",
          "26929"
        ],
        [
          "18583",
          "18672",
          "20193",
          "20253",
          "20789",
          "27633",
          "27665",
          "28214",
          "28427"
        ],
        [
          "18584",
          "19940",
          "19942",
          "19946",
          "20792",
          "20793",
          "20934"
        ],
        [
          "18827",
          "18889",
          "18909",
          "19571",
          "19769",
          "19786",
          "20341",
          "21038",
          "25964",
          "27348",
          "28215"
        ],
        [
          "18882",
          "19159",
          "19194"
        ],
        [
          "19943",
          "20110",
          "20132",
          "20140",
          "20257",
          "20794",
          "20800",
          "20968",
          "21013"
        ],
        [
          "23225",
          "23370",
          "23419",
          "23856",
          "24072",
          "24095",
          "24100",
          "24162"
        ]
      ],
      "name": "seg"
    },
    {
      "source": "python://addbf89eff718c6db5e21e933553dea43c1902f2.b07e7df755542b683fddc83f40d9385a7d907d94",
      "type": "image",
      "opacity": 0.72,
      "name": "EM_image"
    },
    {
      "source": "python://addbf89eff718c6db5e21e933553dea43c1902f2.be56a13b2b32d85b5a7024b6692b86aaa58c801e",
      "type": "segmentation",
      "selectedAlpha": 0.15,
      "colorSeed": 3865243033,
      "segments": [
        "10143",
        "11856",
        "12307",
        "13112",
        "13883",
        "17636",
        "18559",
        "18583",
        "18827",
        "19149",
        "19943",
        "21305",
        "26223",
        "5103",
        "7546"
      ],
      "name": "orig",
      "visible": false
    }
  ],
  "navigation": {
    "pose": {
      "position": {
        "voxelSize": [
          8,
          8,
          40
        ],
        "voxelCoordinates": [
          2484.458740234375,
          1674.3963623046875,
          37.450477600097656
        ]
      }
    },
    "zoomFactor": 6.137647599806845
  },
  "perspectiveOrientation": [
    -0.3638368248939514,
    0.2856803238391876,
    0.522793173789978,
    -0.7160285115242004
  ],
  "perspectiveZoom": 443.63404517712655,
  "showSlices": false,
  "selectedLayer": {
    "layer": "seg",
    "visible": true
  },
  "layout": "3d"
}
## p11 LGN 
{
  "layers": [
    {
      "source": "python://bf85460f820651428212a605c79b9105a4609276.253939d7b7ccfae696fc650fe7b0d417349bebb6",
      "type": "segmentation",
      "colorSeed": 3628257092,
      "segments": [
        "10656",
        "1133",
        "1157",
        "1161",
        "1168",
        "1318",
        "13299",
        "13470",
        "13831",
        "14259",
        "14753",
        "15055",
        "15627",
        "16665",
        "16677",
        "20626",
        "20673",
        "20694",
        "20716",
        "2266",
        "23742",
        "2412",
        "24590",
        "43",
        "4317",
        "4721",
        "5",
        "6498",
        "69",
        "6935",
        "7698",
        "8498",
        "91"
      ],
      "equivalences": [
        [
          "5",
          "783",
          "846",
          "1593",
          "5595",
          "5868",
          "6120",
          "9638",
          "9840",
          "16431",
          "16546",
          "16574",
          "20843",
          "24921",
          "25118",
          "25152",
          "28941",
          "29407",
          "29435"
        ],
        [
          "43",
          "177",
          "508",
          "542",
          "753",
          "755",
          "1140",
          "1281",
          "1848",
          "5539",
          "5850",
          "6037",
          "6109",
          "6133",
          "9242",
          "9611",
          "9631",
          "9800",
          "9822",
          "9864",
          "12506",
          "12608",
          "12657",
          "13106",
          "13107",
          "16320",
          "16326",
          "16442",
          "19767",
          "19827",
          "19993",
          "19994",
          "20158",
          "22888",
          "23263",
          "23626",
          "23929",
          "23933",
          "23968",
          "24653",
          "24899",
          "24908",
          "25111",
          "28955",
          "29226",
          "29448",
          "33273",
          "33275",
          "33630",
          "33950"
        ],
        [
          "69",
          "711",
          "4855",
          "5056",
          "5475",
          "5848",
          "5971",
          "5974",
          "6009",
          "9727",
          "9739",
          "12461",
          "12468",
          "12474",
          "12476",
          "12479",
          "13051",
          "13052"
        ],
        [
          "91",
          "5944",
          "5954",
          "9117",
          "12954"
        ],
        [
          "874",
          "1738",
          "2002",
          "2004",
          "2017",
          "4339",
          "4438",
          "5002",
          "5018",
          "5019",
          "5063",
          "5111",
          "8542"
        ],
        [
          "1133",
          "1496",
          "1498",
          "1505",
          "1694",
          "1696",
          "1748",
          "2030",
          "2034",
          "2080",
          "4429",
          "5503",
          "5621",
          "8559",
          "8718",
          "8752",
          "8788",
          "9246",
          "9275",
          "9565",
          "9637",
          "9796",
          "9797",
          "9828",
          "9829",
          "9830",
          "9844"
        ],
        [
          "1157",
          "1222",
          "1797",
          "1810",
          "5830",
          "5960",
          "5962",
          "5983",
          "10435"
        ],
        [
          "1161",
          "1601",
          "5474",
          "9782",
          "10165"
        ],
        [
          "1168",
          "1172",
          "1300",
          "1606",
          "1608",
          "1785",
          "2197",
          "2474",
          "2478",
          "2487",
          "2565",
          "2568",
          "2569",
          "2579",
          "2581",
          "2582",
          "2583",
          "2589",
          "2592",
          "2915",
          "3246",
          "3247",
          "3349",
          "3370",
          "3373",
          "3378",
          "3379",
          "3382",
          "3394",
          "3395",
          "3402",
          "3426",
          "5492",
          "5825",
          "5947",
          "5948",
          "5949",
          "5950",
          "5992",
          "6358",
          "6359",
          "6360",
          "6372",
          "6390",
          "6408",
          "6563",
          "6564",
          "6617",
          "6619",
          "6620",
          "6623",
          "6624",
          "6627",
          "6628",
          "6630",
          "6633",
          "6634",
          "6636",
          "6646",
          "6651",
          "6652",
          "6661",
          "6667",
          "6668",
          "6688",
          "6784",
          "6789",
          "6809",
          "6850",
          "6873",
          "7109",
          "7110",
          "7111",
          "7115",
          "7133",
          "7189",
          "7190",
          "7192",
          "7193",
          "7196",
          "7197",
          "7200",
          "7201",
          "7204",
          "7206",
          "7207",
          "7208",
          "7209",
          "7213",
          "7214",
          "7215",
          "7217",
          "7219",
          "7221",
          "7222",
          "7223",
          "7226",
          "7230",
          "7248",
          "7253",
          "7263",
          "7268",
          "7270",
          "7278",
          "7288",
          "9091",
          "10081",
          "10328",
          "10343",
          "10396",
          "10464",
          "10465",
          "10660",
          "10882",
          "10894",
          "10899",
          "10903",
          "10915",
          "10917",
          "10918",
          "10922",
          "10927",
          "10932",
          "10939",
          "10941",
          "10942",
          "10946",
          "10947"
        ],
        [
          "1175",
          "1495",
          "5714",
          "6244",
          "6245",
          "6266",
          "6731",
          "8581"
        ],
        [
          "1311",
          "1314",
          "1315",
          "1640",
          "1889",
          "6373",
          "10169",
          "10359",
          "10484",
          "13392",
          "13645",
          "16770",
          "17111",
          "17243"
        ],
        [
          "1318",
          "1342",
          "1343",
          "1359",
          "1364",
          "1641",
          "1651",
          "1652",
          "1659",
          "1868",
          "1869",
          "1941",
          "2242",
          "2622",
          "2626",
          "4891",
          "5170",
          "5625",
          "6143",
          "8612"
        ],
        [
          "1412",
          "1418",
          "1976",
          "1977",
          "1978",
          "6132"
        ],
        [
          "1442",
          "1682",
          "2527"
        ],
        [
          "1446",
          "1953"
        ],
        [
          "1567",
          "2375",
          "2409",
          "2425",
          "3023",
          "3301",
          "6286"
        ],
        [
          "1790",
          "2457",
          "2493",
          "2726"
        ],
        [
          "1944",
          "10213",
          "10511",
          "13458",
          "13632",
          "13634",
          "13688",
          "16451",
          "17271",
          "17287",
          "17288",
          "20486",
          "20487",
          "24503",
          "25403",
          "25411",
          "25425",
          "25657",
          "25825",
          "29745",
          "29746",
          "30021",
          "30026",
          "30028",
          "30185",
          "33737"
        ],
        [
          "2076",
          "2078",
          "2317",
          "2690",
          "2714",
          "3073",
          "3283",
          "3452",
          "3477"
        ],
        [
          "2266",
          "2295",
          "2315",
          "2327",
          "2515",
          "2656",
          "3287",
          "3481",
          "6865",
          "6870",
          "7129",
          "7807",
          "9416",
          "10934",
          "11108",
          "11320",
          "11404",
          "11407",
          "11415"
        ],
        [
          "2388",
          "2547",
          "2713",
          "2742",
          "6707",
          "6719",
          "9928",
          "9929",
          "15331"
        ],
        [
          "2408",
          "6604",
          "6726"
        ],
        [
          "2412",
          "2736",
          "2755",
          "3140",
          "3180",
          "3217",
          "3257",
          "3328",
          "3544",
          "3590",
          "3594",
          "3598",
          "3944",
          "3958",
          "3965",
          "4088",
          "4273",
          "4284",
          "6854",
          "6868",
          "6976",
          "7086",
          "7125",
          "7271",
          "7274",
          "7285",
          "7713",
          "7813",
          "10646",
          "10832",
          "10904",
          "10905",
          "14131",
          "14205"
        ],
        [
          "2797",
          "3252"
        ],
        [
          "4317",
          "8141",
          "8220",
          "8551",
          "8659",
          "8750",
          "8758",
          "8762",
          "11653",
          "11700",
          "11737",
          "12051",
          "12078",
          "12179",
          "12229",
          "12230",
          "12239",
          "12240",
          "12248",
          "12262",
          "12274",
          "12275",
          "12293",
          "14880",
          "14891",
          "14903",
          "14914",
          "14919",
          "14978",
          "15292",
          "15295",
          "15303",
          "15455",
          "15461",
          "15471",
          "15488",
          "18121",
          "18197",
          "18485",
          "18508",
          "18629",
          "18651",
          "18654",
          "18661",
          "18677",
          "18684",
          "18686",
          "18689",
          "18701",
          "21893",
          "21997",
          "22377",
          "22487",
          "22496",
          "22509",
          "22518",
          "22527",
          "22531",
          "22534",
          "26459"
        ],
        [
          "4345",
          "4433",
          "4487",
          "4489",
          "4533",
          "4838",
          "4860",
          "4868",
          "5060",
          "5061",
          "5078",
          "5079",
          "5080",
          "5123",
          "5573",
          "5814",
          "6147",
          "6152",
          "6155",
          "6579",
          "7028",
          "8155",
          "8215"
        ],
        [
          "5158",
          "8588",
          "8760"
        ],
        [
          "5961",
          "9130",
          "9715",
          "12927",
          "13383"
        ],
        [
          "6101",
          "9858",
          "9859",
          "10226",
          "13444",
          "13569",
          "13698",
          "13699",
          "16347",
          "16452"
        ],
        [
          "6135",
          "6136",
          "9446"
        ],
        [
          "6243",
          "8071",
          "8561",
          "8686",
          "8696",
          "8709"
        ],
        [
          "6279",
          "6280",
          "9482",
          "12597",
          "15277",
          "16034",
          "18865",
          "18886",
          "19285",
          "19404"
        ],
        [
          "6498",
          "6531",
          "9672",
          "9984",
          "10020",
          "10180",
          "10200",
          "10264",
          "10278",
          "10325",
          "10436",
          "10458",
          "10496",
          "10514",
          "10521",
          "10525",
          "10530",
          "10531",
          "10534",
          "10547",
          "10555",
          "10558",
          "10560",
          "10567",
          "10568",
          "10571",
          "10575",
          "10580",
          "10581",
          "10585",
          "10589",
          "10591",
          "10602",
          "10654",
          "10705",
          "10706",
          "10725",
          "10733",
          "10734",
          "10739",
          "10745",
          "10759",
          "10765",
          "10773",
          "10819",
          "10867",
          "10883",
          "10884",
          "10886",
          "10906",
          "10907",
          "10909",
          "10911",
          "10912",
          "10916",
          "10919",
          "10920",
          "10921",
          "10923",
          "10925",
          "10938",
          "10943",
          "10954",
          "10955",
          "10962",
          "10963",
          "10964",
          "10969",
          "10971",
          "10974",
          "10991",
          "10993",
          "12745",
          "12764",
          "12830",
          "12862",
          "12878",
          "12940",
          "13148",
          "13169",
          "13185",
          "13186",
          "13191",
          "13197",
          "13198",
          "13203",
          "13209",
          "13211",
          "13213",
          "13215",
          "13217",
          "13223",
          "13224",
          "13230",
          "13254",
          "13260",
          "13265",
          "13447",
          "13479",
          "13481",
          "13496",
          "13497",
          "13506",
          "13508",
          "13511",
          "13512",
          "13514",
          "13517",
          "13519",
          "13522",
          "13525",
          "13526",
          "13527",
          "13528",
          "13530",
          "13533",
          "13535",
          "13541",
          "13571",
          "13580",
          "13582",
          "13608",
          "13611",
          "13612",
          "13613",
          "13614",
          "13615",
          "13616",
          "13700",
          "13701",
          "13704",
          "13717",
          "13732",
          "13733",
          "13736",
          "13738",
          "13739",
          "13740",
          "13741",
          "13742",
          "13743",
          "13744",
          "13745",
          "13746",
          "13747",
          "13749",
          "13750",
          "13751",
          "13753",
          "13754",
          "13755",
          "13756",
          "13757",
          "13759",
          "13760",
          "13761",
          "13762",
          "13764",
          "13765",
          "13767",
          "13768",
          "13769",
          "13770",
          "13771",
          "13773",
          "13774",
          "13775",
          "13777",
          "13780",
          "13781",
          "13782",
          "13783",
          "13785",
          "13786",
          "13789",
          "13790",
          "13792",
          "13793",
          "13797",
          "13798",
          "13800",
          "13801",
          "13802",
          "13804",
          "13807",
          "13808",
          "13816",
          "13817",
          "13820",
          "13821",
          "13822",
          "13919",
          "13941",
          "13974",
          "13989",
          "13990",
          "13995",
          "14005",
          "14031",
          "14064",
          "14096",
          "14100",
          "14165",
          "14166",
          "14175",
          "14192",
          "14194",
          "14195",
          "14196",
          "14197",
          "14198",
          "14224",
          "14226",
          "14232",
          "14235",
          "14236",
          "14238",
          "14239",
          "14241",
          "14242",
          "14246",
          "14247",
          "14248",
          "14265",
          "14266",
          "14267",
          "14270",
          "14271",
          "14272",
          "14275",
          "14276",
          "14277",
          "14279",
          "14281",
          "14284",
          "14289",
          "14290",
          "14291",
          "14292",
          "14296",
          "14298",
          "14300",
          "14301",
          "14305",
          "14309",
          "14310",
          "14322",
          "14324",
          "14326",
          "14336",
          "14344",
          "14354",
          "14360",
          "14362",
          "14363",
          "14364",
          "14365",
          "15680",
          "15692",
          "15698",
          "15724",
          "15735",
          "15760",
          "15761",
          "15804",
          "15809",
          "15827",
          "15839",
          "15844",
          "15875",
          "15927",
          "15958",
          "15964",
          "15978",
          "15980",
          "15981",
          "16004",
          "16005",
          "16031",
          "16036",
          "16046",
          "16051",
          "16053",
          "16054",
          "16055",
          "16060",
          "16061",
          "16062",
          "16065",
          "16068",
          "16072",
          "16073",
          "16074",
          "16075",
          "16076",
          "16079",
          "16080",
          "16089",
          "16093",
          "16097",
          "16098",
          "16099",
          "16100",
          "16101",
          "16104",
          "16105",
          "16114",
          "16116",
          "16117",
          "16118",
          "16119",
          "16120",
          "16127",
          "16142",
          "16144",
          "16270",
          "16324",
          "16356",
          "16360",
          "16364",
          "16367",
          "16370",
          "16371",
          "16373",
          "16376",
          "16377",
          "16378",
          "16379",
          "16380",
          "16382",
          "16383",
          "16386",
          "16388",
          "16389",
          "16390",
          "16419",
          "16426",
          "16443",
          "16454",
          "16455",
          "16457",
          "16507",
          "16517",
          "16519",
          "16521",
          "16559",
          "16560",
          "16567",
          "16568",
          "16570",
          "16575",
          "16576",
          "16577",
          "16583",
          "16585",
          "16588",
          "16589",
          "16590",
          "16591",
          "16592",
          "16593",
          "16594",
          "16595",
          "16596",
          "16598",
          "16599",
          "16600",
          "16601",
          "16602",
          "16604",
          "16605",
          "16606",
          "16607",
          "16608",
          "16609",
          "16610",
          "16611",
          "16612",
          "16614",
          "16615",
          "16616",
          "16617",
          "16618",
          "16619",
          "16620",
          "16622",
          "16623",
          "16624",
          "16625",
          "16626",
          "16627",
          "16628",
          "16630",
          "16633",
          "16634",
          "16635",
          "16636",
          "16637",
          "16638",
          "16639",
          "16640",
          "16644",
          "16645",
          "16646",
          "16647",
          "16649",
          "16650",
          "16892",
          "16913",
          "16988",
          "17001",
          "17127",
          "17139",
          "17276",
          "17305",
          "17309",
          "17310",
          "17350",
          "17358",
          "17360",
          "17362",
          "17364",
          "17365",
          "17366",
          "17370",
          "17371",
          "17373",
          "17391",
          "18961",
          "19021",
          "19100",
          "19123",
          "19288",
          "19331",
          "19332",
          "19410",
          "19426",
          "19448",
          "19451",
          "19475",
          "19479",
          "19499",
          "19512",
          "19515",
          "19517",
          "19522",
          "19523",
          "19525",
          "19528",
          "19529",
          "19531",
          "19534",
          "19537",
          "19542",
          "19546",
          "19549",
          "19551",
          "19552",
          "19563",
          "19566",
          "19853",
          "19882",
          "19888",
          "19889",
          "19906",
          "19907",
          "19908",
          "19917",
          "19977",
          "19986",
          "20006",
          "20109",
          "20110",
          "20111",
          "20135",
          "20136",
          "20156",
          "20184",
          "20185",
          "20186",
          "20187",
          "20198",
          "20204",
          "20207",
          "20212",
          "20219",
          "20224",
          "20227",
          "20228",
          "20230",
          "20232",
          "20233",
          "20243",
          "20609",
          "20722",
          "20854",
          "20877",
          "21058",
          "21109",
          "21110",
          "21135",
          "21150",
          "23107",
          "23217",
          "23273",
          "23281",
          "23285",
          "23292",
          "23296",
          "23299",
          "23300",
          "23302",
          "23319",
          "23698",
          "23978",
          "26359",
          "26554",
          "26561",
          "26750",
          "26781",
          "26787",
          "26791",
          "26950",
          "26973",
          "26976",
          "27354",
          "27396",
          "27413",
          "27539",
          "30349",
          "31162",
          "31217",
          "31218",
          "31554",
          "31651",
          "31679",
          "31680"
        ],
        [
          "6935",
          "10869",
          "10970",
          "14325",
          "17622"
        ],
        [
          "8152",
          "8168",
          "9055",
          "9145"
        ],
        [
          "8313",
          "8558",
          "8620",
          "8812",
          "9337",
          "9538"
        ],
        [
          "8463",
          "8466",
          "12814",
          "13205",
          "13252"
        ],
        [
          "9163",
          "16552",
          "23950",
          "25082"
        ],
        [
          "9212",
          "13123"
        ],
        [
          "9494",
          "9649",
          "9687",
          "9880",
          "10024",
          "12835",
          "12999"
        ],
        [
          "10036",
          "10130",
          "10250",
          "10272",
          "10407"
        ],
        [
          "10114",
          "10166",
          "10655",
          "10898",
          "10913"
        ],
        [
          "10656",
          "11381",
          "11793",
          "12076",
          "12111",
          "12309",
          "12310",
          "12952",
          "13167",
          "13168",
          "13468",
          "13551",
          "13575",
          "13683",
          "13706",
          "13929",
          "14130",
          "14156",
          "14260",
          "14274",
          "15504"
        ],
        [
          "11038",
          "11211",
          "11303",
          "11348",
          "11370",
          "11379",
          "11383",
          "11384",
          "11387",
          "11408",
          "11421",
          "11436",
          "11457",
          "11460",
          "11476",
          "14378",
          "14379",
          "14395",
          "14449",
          "14455",
          "14464",
          "14498",
          "14538",
          "14541",
          "14561",
          "14565",
          "14566",
          "14567",
          "14568",
          "14570",
          "14571",
          "14575",
          "14576",
          "14577",
          "14580",
          "14581",
          "14582",
          "14585",
          "14586",
          "14588",
          "14590",
          "14591",
          "14594",
          "14596",
          "14605",
          "14607",
          "14608",
          "14620",
          "14625",
          "14627",
          "14628",
          "14632",
          "14636",
          "14637",
          "14638",
          "14640",
          "14641",
          "14643",
          "14644",
          "14645",
          "14646",
          "14648",
          "14651",
          "14652",
          "14663",
          "14664",
          "14666",
          "14671",
          "14674",
          "14675",
          "14677"
        ],
        [
          "12620",
          "13450",
          "16444"
        ],
        [
          "12625",
          "13690"
        ],
        [
          "13131",
          "13391",
          "13709",
          "16446"
        ],
        [
          "13299",
          "13303",
          "13540",
          "13588",
          "16299",
          "21043",
          "24533",
          "24896",
          "28933",
          "29199",
          "32810",
          "33238"
        ],
        [
          "13336",
          "16405",
          "17103",
          "20798",
          "24348",
          "24350",
          "24863",
          "29180",
          "29183",
          "29332",
          "29333",
          "29630",
          "29631",
          "29658",
          "29992",
          "30138"
        ],
        [
          "13470",
          "13576",
          "13722",
          "16335",
          "16448"
        ],
        [
          "13831",
          "16740",
          "20292",
          "20330",
          "20777",
          "20947",
          "25672",
          "29613",
          "29627",
          "29647",
          "29652",
          "29973",
          "29991",
          "29998",
          "30000",
          "30126",
          "30132",
          "30140",
          "32551",
          "33479"
        ],
        [
          "14753",
          "14755",
          "15257",
          "15396",
          "15700",
          "15920",
          "15999",
          "16017",
          "16021",
          "16022",
          "16041",
          "18855",
          "18979",
          "19045",
          "19292",
          "19434",
          "19436",
          "19458",
          "19880",
          "20166",
          "20218",
          "20615",
          "20876",
          "21113",
          "21117",
          "21147",
          "24930",
          "25174",
          "25207"
        ],
        [
          "15055",
          "15107",
          "15505",
          "15507",
          "15508",
          "15517",
          "15518",
          "18217",
          "18230",
          "18238",
          "18251",
          "18264",
          "18448",
          "18540",
          "18700",
          "18713",
          "18714",
          "18715",
          "18726",
          "19258",
          "21874",
          "21983",
          "22343",
          "22344",
          "22374",
          "22553",
          "22554",
          "22571",
          "22691",
          "22731",
          "22733",
          "22748",
          "22758",
          "23080",
          "23177",
          "23178",
          "23182",
          "23183",
          "23184",
          "23185",
          "23186",
          "23187",
          "23189",
          "23190",
          "23194",
          "23195",
          "23199",
          "23200",
          "23201",
          "23203",
          "23204",
          "23205",
          "23209",
          "23210",
          "23211",
          "23213",
          "23216",
          "23219",
          "23221",
          "23224",
          "23443",
          "23447",
          "23878",
          "24009",
          "24033",
          "24047",
          "24050",
          "26013",
          "26028",
          "26050",
          "26054",
          "26488",
          "26621",
          "26623",
          "26624",
          "26628",
          "26630",
          "26634",
          "26639",
          "26823",
          "26835",
          "27360",
          "27534"
        ],
        [
          "15627",
          "23468",
          "23909",
          "25037",
          "28766",
          "28840",
          "32729"
        ],
        [
          "16669",
          "16711",
          "17094",
          "17212",
          "20290",
          "20921"
        ],
        [
          "16835",
          "20393",
          "25386",
          "25695",
          "29773",
          "30036",
          "30170"
        ],
        [
          "17937",
          "18014",
          "18333",
          "18414",
          "18484",
          "18650",
          "18792"
        ],
        [
          "17968",
          "18469",
          "21814",
          "21986",
          "22359",
          "22620"
        ],
        [
          "18010",
          "18011",
          "18483",
          "18636",
          "18906",
          "19282"
        ],
        [
          "18020",
          "18487",
          "18641",
          "18642",
          "19262",
          "22688",
          "23390"
        ],
        [
          "18199",
          "22004",
          "22373",
          "22399",
          "22537",
          "22539",
          "22592",
          "26087",
          "26088",
          "26489",
          "26652",
          "26657",
          "26668",
          "27108",
          "27114",
          "27318",
          "27417",
          "30473",
          "30989",
          "30995",
          "31007",
          "31290",
          "31721"
        ],
        [
          "18318",
          "31267"
        ],
        [
          "18342",
          "21974",
          "22084",
          "22329",
          "22576",
          "22581",
          "26061",
          "26077",
          "26447",
          "26660"
        ],
        [
          "18624",
          "18864",
          "19244",
          "19291",
          "19367",
          "19412",
          "19425",
          "23086"
        ],
        [
          "18832",
          "19266",
          "19417"
        ],
        [
          "18946",
          "20448",
          "24618",
          "24941",
          "25166"
        ],
        [
          "20623",
          "23732",
          "25146"
        ],
        [
          "20626",
          "21611",
          "21715",
          "23655",
          "23716",
          "23971",
          "24175",
          "25480",
          "27290",
          "27495",
          "27643",
          "28376",
          "28394",
          "28551",
          "28553",
          "30295",
          "31122",
          "31134",
          "31508",
          "31703",
          "31813"
        ],
        [
          "20673",
          "21126",
          "23618",
          "23653",
          "23952",
          "24097",
          "24164",
          "24165",
          "31157"
        ],
        [
          "20694",
          "27281",
          "27316",
          "27493",
          "27501",
          "27647",
          "28380",
          "31547",
          "31628"
        ],
        [
          "21010",
          "24356",
          "24864"
        ],
        [
          "22054",
          "22063",
          "22390",
          "22572",
          "22573",
          "22831",
          "23232",
          "23242",
          "23247",
          "23256",
          "23474",
          "23489",
          "23538",
          "24020",
          "24088",
          "26170",
          "26198",
          "26513",
          "26724",
          "26729"
        ],
        [
          "22080",
          "26053",
          "26482",
          "26626",
          "26693",
          "26695",
          "26708",
          "27405",
          "31573"
        ],
        [
          "22172",
          "27231",
          "27234",
          "27484"
        ],
        [
          "23133",
          "23254",
          "27576",
          "31915",
          "31974"
        ],
        [
          "23351",
          "27739",
          "28296",
          "33097"
        ],
        [
          "23384",
          "26941",
          "28310",
          "30589",
          "31037",
          "31038",
          "31054",
          "31276",
          "31583",
          "31714"
        ],
        [
          "23469",
          "24068",
          "24822",
          "26246",
          "26508",
          "26537",
          "26710",
          "26711",
          "26751",
          "26991",
          "27156",
          "27172",
          "27409",
          "27414",
          "27553",
          "27560",
          "27567",
          "27606",
          "27872",
          "27888",
          "27889",
          "28323",
          "28341",
          "28479",
          "28491",
          "28492"
        ],
        [
          "24376",
          "25050"
        ],
        [
          "24590",
          "24620",
          "28640",
          "28794",
          "28925",
          "28927",
          "29000",
          "29003",
          "29068",
          "29147",
          "29261",
          "29263",
          "29264",
          "29278",
          "29286",
          "29298",
          "29299",
          "29300",
          "29305",
          "29306",
          "29312",
          "29315",
          "29316",
          "29321",
          "29324",
          "29325",
          "29328",
          "29348",
          "29352",
          "29353",
          "29356",
          "29361",
          "29371",
          "29377",
          "29416",
          "29418",
          "29422",
          "29425",
          "29445",
          "32553",
          "32556",
          "32590",
          "32971",
          "32973",
          "33055",
          "33056",
          "33060",
          "33061",
          "33074",
          "33113",
          "33114",
          "33117",
          "33118",
          "33131",
          "33147",
          "33218"
        ],
        [
          "25431",
          "30211",
          "33531",
          "33537"
        ],
        [
          "25921",
          "26573",
          "26581",
          "26589",
          "26591"
        ],
        [
          "25941",
          "25997",
          "26203",
          "26460",
          "26477",
          "26664",
          "30789",
          "30941",
          "30962"
        ],
        [
          "26172",
          "26544",
          "30779"
        ],
        [
          "26230",
          "26369",
          "26536",
          "26752"
        ],
        [
          "28979",
          "29389",
          "33608"
        ],
        [
          "30449",
          "30590"
        ],
        [
          "31196",
          "32367",
          "32805",
          "32890",
          "32929",
          "32956",
          "33000",
          "33020",
          "33271",
          "33309"
        ],
        [
          "31307",
          "31936",
          "32377"
        ]
      ],
      "name": "seg"
    },
    {
      "source": "python://bf85460f820651428212a605c79b9105a4609276.e37f3c6892960f17d31454adb37668f706ebe2aa",
      "type": "image",
      "name": "EM_image"
    },
    {
      "source": "python://bf85460f820651428212a605c79b9105a4609276.253939d7b7ccfae696fc650fe7b0d417349bebb6",
      "type": "segmentation",
      "selectedAlpha": 0.15,
      "name": "orig",
      "visible": false
    }
  ],
  "navigation": {
    "pose": {
      "position": {
        "voxelSize": [
          8,
          8,
          40
        ],
        "voxelCoordinates": [
          2411.16259765625,
          2129.538818359375,
          122.47785949707031
        ]
      }
    },
    "zoomFactor": 23.090967914143672
  },
  "perspectiveOrientation": [
    -0.036830756813287735,
    -0.9205511212348938,
    0.3888457119464874,
    0.005308316554874182
  ],
  "perspectiveZoom": 261.12520370976864,
  "showSlices": false,
  "selectedLayer": {
    "layer": "seg",
    "visible": true
  },
  "layout": "3d"
}

## V1 
{
  "layers": [
    {
      "source": "python://89bca2cae0163b15ee62d6a049ebab48d582fa67.d41ec34f85f1dd47475cd238627afbd9591905dc",
      "type": "segmentation",
      "colorSeed": 1173341320,
      "segments": [
        "1343",
        "1580",
        "16309",
        "18527",
        "21709",
        "23685",
        "24579",
        "24804",
        "29290",
        "30132",
        "30434",
        "37261",
        "3727",
        "3968",
        "6264"
      ],
      "equivalences": [
        [
          "134",
          "6902",
          "6924",
          "6926",
          "6946",
          "6967",
          "12455",
          "12460",
          "12470",
          "12472",
          "12473",
          "12478",
          "12479",
          "12497"
        ],
        [
          "1343",
          "8481",
          "8510",
          "8534",
          "8556",
          "13859",
          "14965",
          "14991",
          "14994",
          "14995",
          "15009",
          "20144",
          "25947",
          "30472"
        ],
        [
          "1580",
          "2740",
          "10991",
          "16123"
        ],
        [
          "2717",
          "2843",
          "4094"
        ],
        [
          "3727",
          "3731",
          "4960",
          "5024",
          "5029",
          "9398",
          "9447",
          "9462",
          "9530",
          "9531",
          "13498",
          "14674",
          "18686",
          "23452",
          "23453"
        ],
        [
          "3968",
          "9468",
          "13412",
          "13451",
          "13491",
          "13496",
          "13601",
          "13612",
          "14684"
        ],
        [
          "6264",
          "11936",
          "11943",
          "11974",
          "17091",
          "18017",
          "28544",
          "33031",
          "33070",
          "37785",
          "38779",
          "43732"
        ],
        [
          "9956",
          "20221",
          "20243",
          "23903",
          "24988",
          "28369",
          "28430",
          "28458",
          "28463",
          "28504",
          "28512"
        ],
        [
          "13714",
          "13848",
          "13892",
          "13914",
          "17958",
          "18990",
          "19000",
          "19040",
          "19041",
          "19043",
          "20035"
        ],
        [
          "15539",
          "15667",
          "19704",
          "20797",
          "20798",
          "20801",
          "24550",
          "24560",
          "24612",
          "27377",
          "28214"
        ],
        [
          "16309",
          "25117",
          "29584",
          "29593"
        ],
        [
          "18143",
          "19075"
        ],
        [
          "18826",
          "29392",
          "30387",
          "32948"
        ],
        [
          "23559",
          "33582",
          "33655",
          "33687",
          "33714",
          "38315",
          "38432",
          "39332",
          "39404"
        ],
        [
          "23685",
          "24576",
          "30157",
          "30258",
          "30279",
          "30291",
          "30318"
        ],
        [
          "24332",
          "24371"
        ],
        [
          "24579",
          "29191",
          "29902",
          "30043"
        ],
        [
          "24804",
          "24855",
          "26146",
          "29202",
          "29251",
          "29259",
          "29289",
          "29301",
          "29312",
          "29334"
        ],
        [
          "29290",
          "30086",
          "30102",
          "32761",
          "32763",
          "32774",
          "33780",
          "33787",
          "36442",
          "37390"
        ],
        [
          "30132",
          "33677",
          "33726",
          "33747",
          "34538",
          "34577",
          "37348",
          "37478",
          "37508",
          "38480",
          "42305"
        ],
        [
          "30434",
          "30509"
        ]
      ],
      "name": "seg"
    },
    {
      "source": "python://89bca2cae0163b15ee62d6a049ebab48d582fa67.b0d9979285940629434b5fb1b6069b01d0972f76",
      "type": "image",
      "name": "EM_image"
    },
    {
      "source": "python://89bca2cae0163b15ee62d6a049ebab48d582fa67.d41ec34f85f1dd47475cd238627afbd9591905dc",
      "type": "segmentation",
      "selectedAlpha": 0.15,
      "colorSeed": 3817938387,
      "segments": [
        "1343",
        "1580",
        "18527",
        "21709",
        "23685",
        "24804",
        "6264"
      ],
      "name": "orig",
      "visible": false
    }
  ],
  "navigation": {
    "pose": {
      "position": {
        "voxelSize": [
          8,
          8,
          40
        ],
        "voxelCoordinates": [
          2617.912353515625,
          1767.51806640625,
          247.33767700195312
        ]
      }
    },
    "zoomFactor": 30.097482839999287
  },
  "perspectiveOrientation": [
    -0.4461931586265564,
    0.04564759135246277,
    -0.05809497833251953,
    0.8918817043304443
  ],
  "perspectiveZoom": 443.63404517712655,
  "showSlices": false,
  "selectedLayer": {
    "layer": "seg",
    "visible": true
  },
  "layout": "3d"
}


```bash
cd PycharmProjects/FloodFillNetwork-Notes
python3.6 compute_partitions.py    \
--input_volume /home/morganlab/Documents/ixQ_IPL/IxQ_retina_groundtruth.h5::stack    \
--output_volume /home/morganlab/Documents/ixQ_IPL/af_LR.h5::af     \
--thresholds 0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9     \
--lom_radius 24,24,24     \
--min_size 10000

cd ~/PycharmProjects/FloodFillNetwork-Notes
python3.6 build_coordinates.py \
     --partition_volumes Retina_hr:/home/morganlab/Documents/ixQ_IPL/af_LR.h5:af  \
     --coordinate_output /home/morganlab/Documents/ixQ_IPL/tf_record_file \
     --margin 24,24,24
```
```
```bash