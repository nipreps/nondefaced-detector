import matplotlib 
# matplotlib.use('Agg')
import os, sys
sys.path.append('../defacing')

from preprocessing.normalization import clip, standardize, normalize
from preprocessing.conform import conform_data
from helpers.utils import load_vol, save_vol, is_gz_file
import numpy as np
import nibabel as nib
from glob import glob
import matplotlib.pyplot as plt
import tempfile


orig_root_dir = '/work/01329/poldrack/data/mriqc-net/data'
orig_data_face = os.path.join(orig_root_dir, 'face/T1w')
orig_data_mask = os.path.join(orig_root_dir, 'masks')

save_root_dir = '/work/06850/sbansal6/maverick2/mriqc-shared/'

save_preprocessing_face = os.path.join(save_root_dir, 'preprocessing/face')
save_conformed_face = os.path.join(save_root_dir, 'conformed/face')

save_preprocessing_deface = os.path.join(save_root_dir, 'preprocessing/deface')
save_conformed_deface = os.path.join(save_root_dir, 'conformed/deface')

os.makedirs(save_preprocessing_face, exist_ok=True)
os.makedirs(save_preprocessing_deface, exist_ok=True)
os.makedirs(save_conformed_face, exist_ok=True)
os.makedirs(save_conformed_deface, exist_ok=True)

conform_size = (64, 64, 64)
conform_zoom = (4., 4., 4.)


def preprocess(orig_vol_pth, conform_pth, preprocess_pth, DS=None, mask_path=None, debug=False):
    """
    """
    
    filename = orig_vol_pth.split('/')[-1]
    volume, affine, _ = load_vol(orig_vol_pth)
    
    # Preprocessing
    volume = clip(volume, q=90)
    volume = normalize(volume)
    volume = standardize(volume)
    
    # 
    save_preprocessing_path = os.path.join(preprocess_pth, filename)
    save_conformed_path = os.path.join(conform_pth, filename)
    
    print("save_preprocessing_path: ", save_preprocessing_path)
    print("save_conformed_path: ", save_conformed_path)
    
    save_vol(save_preprocessing_path, volume, affine)

    def _plot(data):
        f, axarr = plt.subplots(8, 8, figsize=(12, 12))
        for i in range(8):
            for j in range(8):
                axarr[i, j].imshow(np.rot90(data[:, :, j + 8*i], 1))

        plt.show()
        
#         """
#         plt.subplot(1, 3, 1)
#         plt.imshow(np.rot90(np.mean(data, axis=0)))
#         plt.subplot(1, 3, 2)
#         plt.imshow(np.rot90(np.mean(data, axis=1)))
#         plt.subplot(1, 3, 3)
#         plt.imshow(np.rot90(np.mean(data, axis=2)))
#         plt.show()
#         """
    conform_data(save_preprocessing_path, 
                 out_file=save_conformed_path, 
                 out_size=conform_size, 
                 out_zooms=conform_zoom)

#     if debug: _plot(load_vol(save_conformed_path)[0])
    
    if mask_path and DS:
        mask = np.array(nib.load(mask_path).dataobj)
        masked_volume = volume*mask

        save_mpreprocessing_path = os.path.join(save_preprocessing_deface, DS, filename)
        save_mconformed_path = os.path.join(save_conformed_deface, DS, filename)
        
        print("save_deface_preprocessing_path: ", save_mpreprocessing_path)
        print("save_deface_conformed_path: ", save_mconformed_path)
        
#         os.makedirs(save_mpreprocessing_path, exist_ok=True)
        os.makedirs(os.path.dirname(save_mconformed_path), exist_ok=True)
    
#         save_mpreprocessing_path = os
        
        save_vol(save_mpreprocessing_path, masked_volume, affine)

        conform_data(save_mpreprocessing_path, 
                 out_file=save_mconformed_path, 
                 out_size=conform_size, 
                 out_zooms=conform_zoom)        
        
        return save_conformed_path, save_mconformed_path

    return save_conformed_path


def checkNonConformed(orig_path, save_path):

    conform = []
    orig = []

    for path in glob(save_path + "/*/*.nii*"):
        tempname = path.split("/")[-1]
        ds = path.split("/")[-2]
        conform.append(ds + "/" + tempname)

    print("Total Conformed: ", len(conform))

    for path in glob(orig_path + "/*/*.nii*"):
        tempname = path.split("/")[-1]
        ds = path.split("/")[-2]
        orig.append(ds + "/" + tempname)

    print("Total Original: ", len(orig))

    print("Total not conformed: ", len(orig) - len(conform))

    count = 0
    for f in orig:
        exists = False
        for fc in conform:
            if fc in f:
                exists = True
        if not exists:
            count += 1
            print("Not conformed file: ", f)


for path in glob(orig_data_face + "/*/*.nii*"):
    try:
        if 'ds000140_anat' not in path:
            print("Orig Path: ", path)
            # Example: 
            #        vol_name - sub-22_T1w.nii.gz
            #        DS - ds000140_anat
            vol_name = path.split("/")[-1]
            DS = path.split("/")[-2]
            
            
                
            # directories for saving preprocessed and conformed volumes
            ds_save_conform_path = os.path.join(save_conformed_face, DS)
            ds_save_preprocess_path = os.path.join(save_preprocessing_face, DS)

            # Get the mask path
            mask_path = glob(os.path.join(orig_data_mask, DS, vol_name.split('.')[0] + "*_mask*"))[0]

            print("Mask_path", mask_path)

            if not os.path.exists(ds_save_conform_path):
                os.makedirs(ds_save_conform_path)

            if not os.path.exists(ds_save_preprocess_path):
                os.makedirs(ds_save_preprocess_path)

            # Check if volume is a proper gunzipped
            if not os.path.splitext(path)[1] == ".gz" and is_gz_file(path):
                rename_file = os.path.splitext(vol_name)[0]
                fixed_gz_tmp = os.path.join(save_conformed_face, rename_file)
                print(fixed_gz_tmp)
                subprocess.call(["cp", path, fixed_gz_tmp])

                print(preprocess(fixed_gz_tmp,
                                 ds_save_conform_path,
                                 ds_save_preprocess_path,
                                 mask_path = mask_path,
                                 DS=DS
                                ))
                os.remove(fixed_gz_tmp)

            else:
                print(preprocess(path,
                                 ds_save_conform_path,
                                 ds_save_preprocess_path,
                                 mask_path = mask_path,
                                 DS=DS
                                ))
    except:
        print("Preprocessing incomplete. Exception occurred.")
        pass


# for path in glob(orig_data_deface + "/*/*.nii*"):
#     try:
#         print("Orig Path: ", path)
#         if not is_gz_file(path) and os.path.splitext(path)[1] == ".gz":
#             tempname = path.split("/")[-1]
#             ds = path.split("/")[-2]
#             rename_file = os.path.splitext(tempname)[0]
#             dst = os.path.join(save_data_deface, rename_file)
# #             print(dst)
#             subprocess.call(["cp", path, dst])
#             ds_save_path = os.path.join(save_data_deface, ds)
#             if not os.path.exists(ds_save_path):
#                 os.makedirs(ds_save_path)
#             preprocess(dst, conform_size, save_data_path=ds_save_path))
#         else:
#             ds = path.split("/")[-2]
#             ds_save_path = os.path.join(save_data_deface, ds)
#             if not os.path.exists(ds_save_path):
#                 os.makedirs(ds_save_path)
#             preprocess(path, conform_size, save_data_path=ds_save_path)
#     except:
#         print("Preprocessing incomplete. Exception occurred.")
#         pass
    

checkNonConformed(orig_data_face, save_preprocessing_face)
checkNonConformed(orig_data_face, save_conformed_face)

# checkNonConformed(orig_data_deface, save_data_deface)
