from genmodel import *

model = genmodel_seg_unet((32,32,1))

model.summary()