import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
from tissue_classify.pixel_classifier2D import pixel_classifier_2d, inference_on_image
from scipy.misc import imresize
from PIL import Image
#%%
img_dir = "/Users/binxu/Connectomics_Code/tissue_classifier/Train_Img/"
img = plt.imread(img_dir+"Soma_s092.png") # "Soma_s081_DS.png"
img = np.uint8(img[:, :, 0]*255)
#%%
pc2 = pixel_classifier_2d(img_rows=65, img_cols=65,)
inference_model = pc2.transfer_weight_to_inference("/Users/binxu/Connectomics_Code/tissue_classifier/Models/net_soma_ds-02-0.97.hdf5")
label_map = inference_on_image(img, inference_model)
plt.imshow(label_map)
plt.show()

plt.imshow(imresize(label_map, 2.0, interp='nearest'))
plt.show()
seg = imresize(label_map, 2.0, interp='nearest')
seg = Image.fromarray(seg).convert('L')
im = Image.open(img_dir + "Soma_s081.png").convert('L')
lut = [0]*256
lut[2] = 100
lut[3] = 50
out_img = Image.merge("HSV", (seg.point(lut=lut), seg.point(lut=lut), im))

#%%
ckpt_path = max(iglob(join(ps2.model_dir, '*')), key=os.path.getctime)
inference_model = ps2.transfer_weight_to_inference(ckpt_path)
#%%

lut = [0]*256
lut[2] = 100
lut[3] = 50
# "/Users/binxu/Connectomics_Code/tissue_classifier/Train_Img/Soma_s091.png"
img_dir = "/scratch/binxu.wang/tissue_classifier/Train_Img/"
out_dir = "/scratch/binxu.wang/tissue_classifier/Train_Result/"
img_list = sorted(glob(img_dir+"Soma_s*DS.png"))
for img_name in img_list:
    print("Process ", img_name)
    im = Image.open(img_name).convert('L')
    label_map = inference_on_image(np.array(im), inference_model)
    print("Label finish ", img_name)
    seg = Image.fromarray(label_map)
    out_img = Image.merge("HSV", (seg.point(lut=lut), seg.point(lut=lut), im))
    _, filename = os.path.split(img_name)
    out_img.convert("RGB").save(
        out_dir+filename[:filename.find(".")]+"_label.png")
    print("Merge finish ", img_name)