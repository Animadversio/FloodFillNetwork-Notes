# WIDE FIELD WODEL

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