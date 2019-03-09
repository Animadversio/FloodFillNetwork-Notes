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


```
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

