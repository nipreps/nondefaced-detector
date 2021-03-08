import os, sys
sys.path.append('..')
import preprocess

if __name__=="__main__":
    vol_path = '../../sample_vols/IXI002-Guys-0828-T1.nii.gz'

    preprocess(vol_path)

