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
``` 
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
cd /home/morganlab/Downloads/ffn-master/models/
tensorboard --logdir Norm:LR_model_Longtime,Mov:LR_model_Longtime_Mov,SF:LR_model_Longtime_SF_Deep,WF:LR_model_WF_Longtime


Find the ROI cube, export segments, 
On Cluster get the images, gen h5 file
Do saturated or sparse segmentaion


22660 : 21636, 
15388 : 14364, 
0
```bash
python3 generate_h5_file.py \
 --stack_n 152 --name_pattern "IxD_W002_invert2_2_export_s%03d.png" --path "/home/morganlab/Documents/ixP11LGN" --output_name "grayscale_ixP11_1.h5"

```
IxD_W002_invert2_2_export