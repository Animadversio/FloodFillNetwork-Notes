Reading Note of Flood Fill Network
======

@(Josh Morgan Lab)

This is an un-exhaustive reading note of the Flood-Fill Network. 

## compute_partitions


## build_coordinate


## Train


## Inference
Core Instance `inference.Runner()`
`runner.start()`
`runner.run()`

Inside `runner.run()` the core lines: 
```python
canvas.segment_all(seed_policy=self.get_seed_policy(corner, subvol_size))  # Core lines
self.save_segmentation(canvas, alignment, seg_path, prob_path)
```
Important functionality `self.get_seed_policy(corner, subvol_size)` get seeds for the canvas! 
* "`seed_policy`: callable taking the image and the canvas object as arguments
          and returning an iterator over proposed seed point."
* These policies are defined in the `seed.py` in some children class of `BaseSeedPolicy` class . Default is `seed.PolicyPeaks` . 
	* The `PolicyPeaks` algorithm is "Runs a 3d Sobel filter to detect edges in the raw data, followed by a distance transform and peak finding to identify seed points."
* `seed_policy_args` can be written in `request` to input into seed_policy object, but generally not used! 


`Canvas` is the main object managing the subvolume to be annotated. 
**Main properties** are
* `self.segmentation` the volume tensor filled with markers of different hypervoxel. (the )
	* Initially, all 0 (`np.int32`), can be initialized by `init_segmentation_from_volume()`
* `self.seg_prob` the segmentation probability
	* Initially, all 0 (`np.uint8`)
* `self.seed` volume tensor, like a draft version of segmentation, not real. 
	* `np.float32`
* `movement_policy_fn`: callable taking the `Canvas` object as its only argument and returning a movement policy object
          (see `movement.BaseMovementPolicy`)
          * Default is `movement.FaceMaxMovementPolicy(self, deltas=model.deltas[::-1], score_threshold=self.options.move_threshold)` 
* `self.options`  ( is set by the pb file reading function `inference_pb2.InferenceOptions()` and input `options`,) finally comes from `self.request.inference_options`  in `Runner()`, which is the `inference_options` option part in the `request.pbtxt`
* `checkpoint_interval_sec` default 0, if >1 these will be checkpoint saved in `checkpoint_path` during `segment_all` process for safety. 


`self_prediction` in `request`, `halt_signaler` signal pathway!  Can be utilized !! 


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
* The loop will stop **only if all the possible seeds are used**! So running time is kind-of proportional to the number of seeds. 
* The `Segmentation` will only be saved formally after this. so better `check_point` in case of crash


In `segment_at(pos)` functionality, input start postion `pos`, return `num_iters`.  ( Note optional parameters can be added to dynamic visualize the image! `dynamic_image` `vis_update_every` `vis_fixed_z` )

* Core line is `pred = self.update_at(pos, start_pos)`  functionality
* The `update_at` function is (iteratively) applying the `predict` function from `model` , mainly update the `seed` tensor (the only place the data is directly updated by algorithm). 
	* `self.seed` looks like a draft / casch version of the segmentation, the update can be rejected in the stage of `segment_all` 

**What's the method of making inference, how to explain that?**
**Why there is failure based on small?**

