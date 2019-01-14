Reading Note of Flood Fill Network
======

@(Josh Morgan Lab)[Connectomics, machine learning]


[^1]: This is an un-exhaustive reading note of the Flood-Fill Network. 

## Preprocessing

### compute_partitions
Reading in groundtruth segmentation file, it creates a dataset in the output `h5` file, with margin filled with `fillvalue` 255 (uint8), and center filled with `output` from  `compute_partitions`  

The major computation is in`compute_partitions` , which 
* Takes in the input ground truth in `seg_array`, discard the 
* Clear the dust,  threshold out (mark as 0) the too small segmentation label using `min_size` threshold
* `unique` to sort out all `labels`  (See the log `Label number`)
* For each label `l`
	* Find all the voxels with this label, give the `object_mask`
	* Count the same-mark voxels in a cube centered each voxel in `active_fraction`
	* Quantize the `active_fraction` tensor with array `threshold` : 
		* `output[loc]=i, if active_fraction[loc] \in [threshold[i-1], threshold[i])`
* So the `output` is one `lom_radius` smaller the input on each direction (2 `lom_radius` smaller for each axis)
* But it's pad with 255 when write down to `h5` file. 
* **Note** there is no object number information in the compute_partition output file ! The information coded in the tensor is the quantized `activation_fraction` number of each voxel. 

**Major parameters**: 
* `thresholds` 0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9  quantization threshold
* `lom_radius` 24,24,24 can change when 3 axis are not the same 
* `min_size` 10000

### build_coordinate
```bash
python build_coordinates.py ^
     --partition_volumes LGN_HR::C:\Connectomics\af_zyx.h5::af ^
     --coordinate_output C:\Connectomics\tf_record_file_zyx ^
     --margin 24,24,24
```
The program runs as this, 
* Take in the partition volume output from the `compute_partition` above
* The padding borders are stripped off by the `margin` parameter (`mx,my,mz` in each direction) 
	* Note the margin parameter can be independent with `lom_radius` in `compute_partition`
* 

**Major parameters**: 
* `margin` 24,24,24  can be tuned larger or smaller but 

## Model Architecture
`FFNModel` is defined in the `model.py`. 

* `input_image_size`,`input_seed_size` are used in inference.  (Note these are in [x,y,z] direction not but the `_pred_size`,`_input_seed_size`,`_input_image_size` are in z,y,x direction)
* `pred_mask_size` Size of the predicted patch as returned by the model. 
* Specifically, the default model is the children class `convstack_3d.ConvStack3DFFNModel(model.FFNModel)` 
	* In this model, the `fov_size` is used to set the input and output size (`input_seed_size`, `pred_mask_size`) by `self.set_uniform_io_size(fov_size)`,  thus larger `fov_size` gives larger field of view. 
	* `delta` used to specify a collection of moving direction in `self.shifts`

**Outside parameters** that can be adjusted in `model_args` when training

* `depth`: 12, `fov_size`: [33, 33, 33], `deltas`: [8, 8, 8]
* `depth` is the depth of conv layers in the model 

Note the `fov_size` and `deltas` are not free themselves, but related to the `margin` and `lom_radius` in the preprocessing period : "`margin` should normally be set to the radius of the FFN training FoV (i.e. network FoV radius + deltas.)"

## Training

`train.py` define the sample prepare and training control loop 

`fov_policy` is one of the 2 `'fixed', 'max_pred_moves'` first are a fixed 26 direction moving (Can be shuffled). `max_pred_moves` is "moves to the voxel with maximum mask activation within a plane perpendicular to one of the 6 Cartesian directions, offset by +/- model.deltas from the current FOV position.'"

## Inference
Core Instance `inference.Runner()`
`runner.start()`
`runner.run()`

Inside `runner.run()` the core lines: 
```python
canvas.segment_all(seed_policy=self.get_seed_policy(corner, subvol_size))  # Core lines
self.save_segmentation(canvas, alignment, seg_path, prob_path)
```

### Getting Seeds to Focus at 
Important functionality `self.get_seed_policy(corner, subvol_size)` get seeds for the canvas! 
* "`seed_policy`: callable taking the image and the canvas object as arguments and returning an iterator over proposed seed point."
* These policies are defined in the `seed.py` in some children class of `BaseSeedPolicy` class . Default is `seed.PolicyPeaks` . 
	* The Default `PolicyPeaks` algorithm is "*Runs a 3d Sobel filter to detect edges in the raw data, followed by a distance transform and peak finding to identify seed points*." 
* `seed_policy_args` can be given in `request` to  `seed_policy` object, but generally not used! 
* Note the canvas will go through all the `seed` in `seed_policy`, so the running time is kind of proportional to the seed number. Which can be adjusted by parameters in `seed_policy`

### Segmentation Management Object
`Canvas` is the main object managing the subvolume to be annotated. 
**Main properties** are
* `self.segmentation` the volume tensor filled with markers of different hypervoxel. (the )
	* Initially, all 0 (`np.int32`), can be initialized by `init_segmentation_from_volume()`
* `self.seg_prob` the quantized segmentation probability calculated from `seed`
	* Initially, all 0 (`np.uint8`)
* `self.seed` volume tensor, like a temporary, draft version of segmentation, not real or final
	* `np.float32` 
	* It's the only thing that `segment_at` will change! 
	* If `reset_seed_per_segment` is True (default) then the `seed` will be init as `NAN` at each run of `segment_at`!!! 
* `movement_policy_fn`: callable taking the `Canvas` object as its only argument and returning a movement policy object
          (see `movement.BaseMovementPolicy`)
          * Default is `movement.FaceMaxMovementPolicy(self, deltas=model.deltas[::-1], score_threshold=self.options.move_threshold)` 
* `self.options`  ( is set by the pb file reading function `inference_pb2.InferenceOptions()` and input `options`,) finally comes from `self.request.inference_options`  in `Runner()`, which is the `inference_options` option part in the `request.pbtxt`
* `checkpoint_interval_sec` default 0, if >1 these will be checkpoint saved in `checkpoint_path` during `segment_all` process for safety. 
* 


`self_prediction` in `request`, `halt_signaler` signal pathway!  Can be utilized !! 

### Segmentation Procedure
in `segment_all` major thing is to iterate through the seeds in `self.seed_policy` object, for each seed named `pos`, try the following things
* Check if it's a valid seed position
* Use parameter `self.options.min_boundary_dist` (`mbd`) to check if there is any segmented / annotated voxel around the seed (a cube centered at `pos`, with edge length `2*mbd`)
* Start Trying to segment from the seed! `self.segment_at(pos)` . (Log `'Starting segmentation at %r (zyx)', pos` ), directly update 
	* If the seed is too weak `self.seed[pos] < self.options.move_threshold` by `move_threshold`(0.9), then this segmentation is discarded. 
	* If new segmentation is too small, (threshold set at `self.options.min_segment_size`), then this supervoxel is omitted. 
	* Otherwise, **succeed**! 
		* The `mask` got by thresholding the seed `mask = self.seed[sel] >= self.options.segment_threshold`, and exclude the marked voxels `mask &= self.segmentation[sel] <= 0`
		* Mark the masked voxels in segmentation voxel as new id. `self.segmentation[sel][mask] = self._max_id`
		* Mark the`seg_prob` tensor  with probability `self.seg_prob[sel][mask] = storage.quantize_probability( expit(self.seed[sel][mask]))`
		* (all base on the update in `self.seed`)
		* Log `'Created supervoxel:%d  seed(zyx):%s  size:%d  iters:%d'` ! 
* Each time it can only create one supervoxel of one color from the seed! 
* The loop will stop **only if all the possible seeds are used**! So running time is kind-of proportional to the number of seeds. 
* The `Segmentation` will only be saved formally after this. so better `check_point` in case of crash

 ![Alt text](./1544561968008.png)

In `segment_at(pos)` functionality, input start postion `pos`, return `num_iters`.  ( Note optional parameters can be added to dynamic visualize the image! `dynamic_image` `vis_update_every` `vis_fixed_z` super cool visualization )

* Core line is `pred = self.update_at(pos, start_pos)`  functionality
* The `update_at` function is (iteratively) applying the `predict` function from `model` , mainly update the `seed` tensor (the only place the data is directly updated by algorithm). 
	* `self.seed` looks like a draft / casch version of the segmentation, the update can be rejected in the stage of `segment_all` 
	* The `update_at` function only update a cube with dimension `self._input_seed_size` around the `seed` `pos`.  `self._input_seed_size` is defined in the model 

The main loop iterate through `self.movement_policy` which dynamically generate new pos to `update_at(pos, start_pos)`. The movement policy is as such (e.g. default `FaceMaxMovementPolicy`),
 * 

### How to understand the visualization
in function `visualize_state`, esp. in `ortho_plane_visualization`, the functions cut 3 view through the center in  a volume and add cross hair to them! 

Visualized volume is `seed_logits`

**What's the method of making inference, how to explain that?**
**Why there is failure based on small?**


## Agglomeration
`inference.resegmentation`


# Replaceable parts in the pipeline
* Model parameters: `depth`, `fov_radius`, moving vectors `deltas`
* Seed_generating policy in the `inference` part. 