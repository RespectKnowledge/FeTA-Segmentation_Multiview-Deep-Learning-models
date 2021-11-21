# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 15:43:28 2021

@author: Abdul Qayyum
"""
#%% combining dataset into 3D volume for 2D slices and check from ITK-SNAP

import os
import numpy as np
import os
import nibabel as nib
from skimage import io
import cv2
import numpy as np
from skimage import io, exposure, img_as_uint, img_as_float
from PIL import Image 
import PIL 
import imageio
import scipy.misc
import pandas as pd
import os
import glob
import SimpleITK as sitk
import ast
import json
import os
import nibabel as nib
from skimage import io
import cv2
import numpy as np
from skimage import io, exposure, img_as_uint, img_as_float
from PIL import Image 
import PIL 
import imageio
import scipy.misc
import pandas as pd

path = 'C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\feta_2.0\\fetadatasetnew\\Training\\sub-001\\anat\\sub-001_rec-mial_T2w.nii.gz'
patho=nib.load(path)
img_data = nib.load(path).get_fdata()
 
outputDir='C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\feta_2.0\\fetadatasetnew\\reconstruct'
   
X_train = np.zeros((256,256, 256))
for file in range(0,img_data.shape[2]):
    img=img_data[:,:,file]
    X_train[:,:,file]=img
    img1=img_data[:,file,:]
    X_train[:,file,:]=img1
    img2=img_data[file,:,:]
    X_train[file,:,:]=img2
nib.save(nib.Nifti1Image(X_train, patho.affine), os.path.join(outputDir, str(3) + '_seg_result.nii.gz'))
    


        


#%% axial dataset module
import os
import numpy as np
import os
import nibabel as nib
from skimage import io
import cv2
import numpy as np
from skimage import io, exposure, img_as_uint, img_as_float
from PIL import Image 
import PIL 
import imageio
import scipy.misc
import pandas as pd
import os
import glob
import SimpleITK as sitk
import ast
import json
import os
import nibabel as nib
from skimage import io
import cv2
import numpy as np
from skimage import io, exposure, img_as_uint, img_as_float
from PIL import Image 
import PIL 
import imageio
import scipy.misc
import pandas as pd

path = 'C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\feta_2.0\\fetadatasetnew\\Training'
save_img="C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\feta_2.0\\fetadatasetnew\\trainingdata\\images"
save_msk="C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\feta_2.0\\fetadatasetnew\\trainingdata\\masks"
patients = os.listdir(f'{path}')
len(patients)
pathimg1=[]
data = {
    'patient': [],
    'patientid': [],
    'channel': []
}
for i in patients:
    print(i)
    pathim=os.path.join(path,i)
    pathimg=glob.glob(os.path.join(pathim, 'anat', '*_T2w.nii.gz'))[0]
    pathtestmask=glob.glob(os.path.join(pathim, 'anat', '*_dseg.nii.gz'))[0]
    sub = os.path.split(pathimg)[1].split('_')[0] # 
    pathimg1.append(pathimg)
    print(pathimg)
    # image data file
    img_data = nib.load(pathimg).get_fdata()
    # image data file
    # mask data file
    msk_data=nib.load(pathtestmask).get_fdata()
    
    for file in range(0,img_data.shape[2]):
        img=img_data[:,:,file]
        img = exposure.rescale_intensity(img, out_range='float')
        img = img_as_uint(img)
        msk=msk_data[:,:,file]
        msk=msk.astype(np.uint8)
        io.imsave(os.path.join(save_img,str(i.split("\\")[0])+"_"+str(file)+".png"),img)
        io.imsave(os.path.join(save_msk,str(i.split("\\")[0])+"_"+str(file)+".png"),msk)




# # #x,y,z        
# imgx=img_data[:,:,100] # axial x
# imgy=img_data[:,100,:]  # cornoal y
# imgz=img_data[100,:,:]   # sagtital  

# full= imgx+imgy+imgz

# imgx1=msk_data[:,:,100].T # axial x
# imgy1=msk_data[:,100,:].T  # cornoal y
# imgz1=msk_data[100,:,:].T   # sagtital   



#%% cornal dataset module
import os
import numpy as np
import os
import nibabel as nib
from skimage import io
import cv2
import numpy as np
from skimage import io, exposure, img_as_uint, img_as_float
from PIL import Image 
import PIL 
import imageio
import scipy.misc
import pandas as pd
import os
import glob
import SimpleITK as sitk
import ast
import json
import os
import nibabel as nib
from skimage import io
import cv2
import numpy as np
from skimage import io, exposure, img_as_uint, img_as_float
from PIL import Image 
import PIL 
import imageio
import scipy.misc
import pandas as pd

path = 'C:\\Users\\Administrateur\\Desktop\\micca2021\MICCAI2021\\feta_2.0\\fetadatasetnew\\Validation'
save_img="C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\feta_2.0\\fetadatasetnew\\Validation_coronal\\images"
save_msk="C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\feta_2.0\\fetadatasetnew\\Validation_coronal\\masks"
patients = os.listdir(f'{path}')
len(patients)
pathimg1=[]
data = {
    'patient': [],
    'patientid': [],
    'channel': []
}
for i in patients:
    print(i)
    pathim=os.path.join(path,i)
    pathimg=glob.glob(os.path.join(pathim, 'anat', '*_T2w.nii.gz'))[0]
    pathtestmask=glob.glob(os.path.join(pathim, 'anat', '*_dseg.nii.gz'))[0]
    sub = os.path.split(pathimg)[1].split('_')[0] # 
    pathimg1.append(pathimg)
    print(pathimg)
    # image data file
    img_data = nib.load(pathimg).get_fdata()
    # image data file
    # mask data file
    msk_data=nib.load(pathtestmask).get_fdata()
    
    for file in range(0,img_data.shape[1]):
        img=img_data[:,file,:]
        img = exposure.rescale_intensity(img, out_range='float')
        img = img_as_uint(img)
        msk=msk_data[:,file,:]
        msk=msk.astype(np.uint8)
        io.imsave(os.path.join(save_img,str(i.split("\\")[0])+"_"+str(file)+".png"),img)
        io.imsave(os.path.join(save_msk,str(i.split("\\")[0])+"_"+str(file)+".png"),msk)

#%% satigal dataset module
import os
import numpy as np
import os
import nibabel as nib
from skimage import io
import cv2
import numpy as np
from skimage import io, exposure, img_as_uint, img_as_float
from PIL import Image 
import PIL 
import imageio
import scipy.misc
import pandas as pd
import os
import glob
import SimpleITK as sitk
import ast
import json
import os
import nibabel as nib
from skimage import io
import cv2
import numpy as np
from skimage import io, exposure, img_as_uint, img_as_float
from PIL import Image 
import PIL 
import imageio
import scipy.misc
import pandas as pd

path = 'C:\\Users\\Administrateur\\Desktop\\micca2021\MICCAI2021\\feta_2.0\\fetadatasetnew\\Validation'
save_img="C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\feta_2.0\\fetadatasetnew\\validation_sati\\images"
save_msk="C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\feta_2.0\\fetadatasetnew\\validation_sati\\masks"
patients = os.listdir(f'{path}')
len(patients)
pathimg1=[]
data = {
    'patient': [],
    'patientid': [],
    'channel': []
}
for i in patients:
    print(i)
    pathim=os.path.join(path,i)
    pathimg=glob.glob(os.path.join(pathim, 'anat', '*_T2w.nii.gz'))[0]
    pathtestmask=glob.glob(os.path.join(pathim, 'anat', '*_dseg.nii.gz'))[0]
    sub = os.path.split(pathimg)[1].split('_')[0] # 
    pathimg1.append(pathimg)
    print(pathimg)
    # image data file
    img_data = nib.load(pathimg).get_fdata()
    # image data file
    # mask data file
    msk_data=nib.load(pathtestmask).get_fdata()
    
    for file in range(0,img_data.shape[1]):
        img=img_data[file,:,:]
        img = exposure.rescale_intensity(img, out_range='float')
        img = img_as_uint(img)
        msk=msk_data[file,:,:]
        msk=msk.astype(np.uint8)
        io.imsave(os.path.join(save_img,str(i.split("\\")[0])+"_"+str(file)+".png"),img)
        io.imsave(os.path.join(save_msk,str(i.split("\\")[0])+"_"+str(file)+".png"),msk)
              

        
      
#%% validation dataset
import os
import numpy as np
import os
import nibabel as nib
from skimage import io
import cv2
import numpy as np
from skimage import io, exposure, img_as_uint, img_as_float
from PIL import Image 
import PIL 
import imageio
import scipy.misc
import pandas as pd
import os
import glob
import SimpleITK as sitk
import ast
import json
import os
import nibabel as nib
from skimage import io
import cv2
import numpy as np
from skimage import io, exposure, img_as_uint, img_as_float
from PIL import Image 
import PIL 
import imageio
import scipy.misc
import pandas as pd

path = 'C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\feta_2.0\\fetadatasetnew\\Validation'
save_img="C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\feta_2.0\\fetadatasetnew\\validationdata\\images"
save_msk="C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\feta_2.0\\fetadatasetnew\\validationdata\\masks"
patients = os.listdir(f'{path}')
len(patients)
pathimg1=[]
data = {
    'patient': [],
    'patientid': [],
    'channel': []
}
for i in patients:
    print(i)
    pathim=os.path.join(path,i)
    pathimg=glob.glob(os.path.join(pathim, 'anat', '*_T2w.nii.gz'))[0]
    pathtestmask=glob.glob(os.path.join(pathim, 'anat', '*_dseg.nii.gz'))[0]
    sub = os.path.split(pathimg)[1].split('_')[0] # 
    pathimg1.append(pathimg)
    print(pathimg)
    # image data file
    img_data = nib.load(pathimg).get_fdata()
    # image data file
    # mask data file
    msk_data=nib.load(pathtestmask).get_fdata()
    
    for file in range(0,img_data.shape[2]):
        img=img_data[:,:,file]
        img = exposure.rescale_intensity(img, out_range='float')
        img = img_as_uint(img)
        msk=msk_data[:,:,file]
        msk=msk.astype(np.uint8)
        io.imsave(os.path.join(save_img,str(i.split("\\")[0])+"_"+str(file)+".png"),img)
        io.imsave(os.path.join(save_msk,str(i.split("\\")[0])+"_"+str(file)+".png"),msk)