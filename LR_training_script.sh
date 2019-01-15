python compute_partitions.py \
--input_volume  third_party/LGN_DATA/groundtruth_LR.h5:stack \
--output_volume third_party/LGN_DATA/af_LR.h5:af \
--thresholds 0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
--lom_radius 36,24,11 \
--min_size 10000

python build_coordinates.py \
--partition_volumes LGN_LR:third_party/LGN_DATA/af_LR.h5:af \
--coordinate_output third_party/LGN_DATA/tf_record_file_LR \
--margin 35,24,11

python train.py \
--train_coords third_party/LGN_DATA/tf_record_file_LR \
--data_volumes LGN_LR:third_party/LGN_DATA/grayscale_maps_LR.h5:raw \
--label_volumes LGN_LR:third_party/LGN_DATA/groundtruth_LR.h5:stack \
--train_dir /tmp/LR_model \
--model_name convstack_3d.ConvStack3DFFNModel \
--model_args "{\"depth\": 9, \"fov_size\": [55, 37, 17], \"deltas\": [9,6,3]}" \
--image_mean 128 \
--image_stddev 33


 