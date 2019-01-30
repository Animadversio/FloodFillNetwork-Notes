from google.protobuf import text_format
from ffn.inference import inference
from ffn.inference import inference_pb2
from ffn.inference import storage
from ffn.inference import seed
from scipy import ndimage
import matplotlib.pyplot as plt
import pickle
import h5py
import numpy as np
#%%
config = """inference {
    image {
      hdf5: "/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/grayscale_maps_LR.h5:raw"
    }
    image_mean: 128
    image_stddev: 33
    checkpoint_interval: 1800
    seed_policy: "PolicyPeaks"
    model_checkpoint_path: "/home/morganlab/Downloads/ffn-master/models/LR_model2/model.ckpt-10000"
    model_name: "convstack_3d.ConvStack3DFFNModel"
    model_args: "{\\"depth\\": 9, \\"fov_size\\": [55, 37, 17], \\"deltas\\": [9,6,3]}"
    segmentation_output_dir: "/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR"
    inference_options {
      init_activation: 0.95
      pad_value: 0.05
      move_threshold: 0.9
      min_boundary_dist { x: 3 y: 3 z: 1}
      segment_threshold: 0.6
      min_segment_size: 1000
    }
    init_segmentation {
       npz: "/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR/0/0/seg-0_0_0.npz:segmentation"
    }
}
points {id_a:22 id_b:29 point {x: 158 y: 510 z: 8} }
points {id_a:432 id_b:636 point {x: 334 y: 542 z: 32} }
radius {x: 200 y: 200 z: 20}
output_directory: "/home/morganlab/Downloads/ffn-master/results/LGN/testing_LR_WF/reseg"
max_retry_iters: 2
segment_recovery_fraction: 0.4
analysis_radius {x: 200 y: 200 z: 20}
"""

reseg_req = inference_pb2.ResegmentationRequest()
_ = text_format.Parse(config, reseg_req)
req = reseg_req.inference
#%% Get the canvas and volume
runner = inference.Runner()
runner.start(req)
runner.set_pixelsize([8, 12, 30])
canvas, alignment = runner.make_canvas((0,0,0),(175, 1058, 1180))

#%% Seed policy calculation
seed_policy = seed.PolicyPeaks(canvas)
seed_policy._init_coords()
seed_list = seed_policy.coords
#%%
seeds_segment = np.zeros(canvas.shape, dtype=np.uint8)
seeds_segment[seed_list[:, 0], seed_list[:, 1], seed_list[:, 2]] = 1
#%%
export_segmentation_to_VAST('/home/morganlab/Downloads/LGN_WF_Autoseg_Result2/seeds', seeds_segment)

#%% Check filtering procedure and intermediate result!
edges = ndimage.generic_gradient_magnitude(
        canvas.image.astype(np.float32),
        ndimage.sobel)
#%%
sigma = 49.0 / 6.0
thresh_image = np.zeros(edges.shape, dtype=np.float32)
ndimage.gaussian_filter(edges, sigma, output=thresh_image, mode='reflect')
filt_edges = edges > thresh_image
#%%
dt = ndimage.distance_transform_edt(1 - filt_edges).astype(np.float32)
#%%
#%%
del thresh_image
#%% visualize the result! First generate the seed list
zid = 50
seeds_in_layer = seed_list[seed_list[:, 0] == zid]
#%%
def show_img_with_scatter(img, zid, title, savename=""):
    plt.figure(figsize=[12, 12])
    plt.imshow(img[zid, :, :], cmap='gray')
    plt.scatter(seeds_in_layer[:, 2], seeds_in_layer[:, 1], c='red', s=2)
    plt.title(title+" z=%d"%zid)
    if not savename=="":
        plt.savefig(savename+"z%d.png" % zid)
    plt.show()
    plt.close()

show_img_with_scatter(canvas.image, zid, "EM image, normalized", "EM_img")
show_img_with_scatter(edges, zid, "Edge (Sobel filtered) image", "sobel_filter")
show_img_with_scatter(filt_edges, zid, "Thresholded edge image", "thresh_edge")
show_img_with_scatter(dt, zid, "Euclidean transformed edge image", "edt")


#%%
# pickle.dump(seed_list, open("/home/morganlab/Downloads/LGN_Autoseg_Result/Seed_Distribution/seed_list.pkl", 'wb'))
#%%
# seed_list = pickle.load(open("/home/morganlab/Downloads/LGN_Autoseg_Result/Seed_Distribution/seed_list.pkl", 'rb'))
#%%
#f = h5py.File("/home/morganlab/Downloads/ffn-master/third_party/LGN_DATA/grayscale_maps_LR.h5",'r')
volume = storage.decorated_volume(req.image)
#%%
volume = volume.value.copy()
#%%% Visualization
for zid in range(5):
    seeds_in_layer = seed_list[seed_list[:, 0] == zid]
    plt.imshow(volume[zid, :, :], cmap='gray')
    plt.scatter(seeds_in_layer[:, 2], seeds_in_layer[:, 1], c='red', s=0.5) # note the way they scatter, Xaxis corr to 3rd index
    plt.axis('off')
    plt.savefig('seed_distribution_%d.png'%zid)
    plt.show()
    plt.close()

