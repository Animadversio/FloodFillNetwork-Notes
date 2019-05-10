Flood-Fill Network Tutorial
=======
@(Josh Morgan Lab)[Connectomics, Deep Learning]
## General Segmentation Workflow
First I'll introduce the normal segmentation workflow, I'm currently using. 

* Image preprocessing and stacking (local)
	* If work on linux machine, use `wine` to start the VAST program
	* Export from VAST, downsampling to around 
	* Alignment (`Fiji`) with registration with SIFT 
	* Cropping (`Fiji`)
	* Stacking (`Python`)
	* Normalization (`Python`)
	* Preview of image volume (`Python.neuroglancer`) if it looks bad  tune the parameters and do it again. 
* (Below is all in `Python.ffn` package if not otherwise mentioned)*
* Distributed Model selection for the specific tissue / volume (on Cluster) 
	* Select some models to run on a small typical trunk of image and test their result visually. run distributedly
	* 
* Distributed Large Scale Inference (on Cluster)
	* Use the selected model to do inference on different trunks of images distributedly
* Result Inspection  (`Python.neuroglancer`)
	* See which subvolumes are good, which are bad, use which 2 to do consensus
* Post-processing (local / Cluster)
	* Consensus oversegmentation (local) ~ 1hr for consensus 45 tiles
	* Subvolume stitching (local) ~ 20 min for 5$\times$9 tiles
	* Manual Agglomeration with `neuroglancer` add-on
	* 
	* Distributed Resegmentation seed generation (on Cluster, ~10 min for 45 jobs to be done, very quick)
	* Distributed Resegmentation (on Cluster, Relatively quick) 
	* Generate Agglomeration graph
	* Visualize merged segmentation and tune the thresholding criterion
* Proofreading
	* Use `knossos_utils` to export the segmentation to `KNOSSOS` 
	* Proof reading and change
