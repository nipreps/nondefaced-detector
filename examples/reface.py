import os
import glob
import subprocess
import time

orig_root_dir = '/work/01329/poldrack/data/mriqc-net/data'
orig_face_defaced = os.path.join(orig_root_dir, 'masks')

orig_reface_dir = '/work/06850/sbansal6/maverick2/mriqc-shared/refaced'
save_orig_faced_root_dir = os.path.join(orig_reface_dir, 'orig_faced')

simg_path = os.path.join(orig_reface_dir,'afni-latest.simg')
print(simg_path)

# list of datasets already refaced from the original defaced datasets
deface_exclude = [
    'ds002033_anat',
    'ds002149_anat',
    'ds001928_anat', 
    'ds002247_anat',
    'ds002001_anat',
    'ds002316_anat',
    'ds002076_anat',
    'ds002156_anat'
]

# list of datasets already refaced from the orignal faced datasets
face_exclude = [
    'ds000140_anat',
    'ds000119_anat',
    'ds000157_anat',
    'ds002509_anat',
    'ds001650_anat',
    'ds000208_anat',
    'ds001569_anat'
]

face_parallel = [
    'ds000157_anat',
    'ds000205_anat',
    'ds000232_anat',
    'ds001019_anat',
    'ds001393_anat',
    'ds001568_anat',
    'ds001650_anat',
    'ds002572_anat',
    'ds000118_anat',
    'ds000140_anat',
    'ds000159_anat',
    'ds000206_anat',
    'ds000245_anat',
    'ds001037_anat',
    'ds001505_anat',
    'ds001569_anat',
    'ds001900_anat',
    'ds002578_anat'
]

ds_root_paths = glob.glob(orig_face_defaced + '/*_anat*')

processes = set()
max_processes = 16

for data_dir in ds_root_paths:
    print("datadir: ", data_dir)
    dd = data_dir.split('/')[-1]
    if dd not in face_exclude and dd in face_parallel:
        print(dd)
        save_dd_dir = os.path.join(save_orig_faced_root_dir, dd)
        os.makedirs(save_dd_dir, exist_ok=True)
        ds = glob.glob(data_dir + '/*_defaced.nii*')

        for vol in ds:
            vol_name = vol.split('/')[-1].split('.')[0]
            vol_save_path = os.path.join(save_dd_dir, vol_name)
            os.makedirs(vol_save_path, exist_ok=True)
            prefix_pre = vol_save_path
            prefix = os.path.join(prefix_pre, vol_name + '_refaced.nii.gz')
            print("Processing Volume: ", vol)
            print("Save Path: ", prefix)
            processes.add(subprocess.Popen(["singularity","exec", simg_path, "@afni_refacer_run",
                                            "-input", vol,
                                            "-mode_reface_plus",
                                            "-prefix", prefix
                                           ]))

            if len(processes) >= max_processes:
                os.wait()
                processes.difference_update([
                    p for p in processes if p.poll() is not None])
