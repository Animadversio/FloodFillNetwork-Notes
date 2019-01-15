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