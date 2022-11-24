#!/usr/bin/env python3

import numpy as np
import mrcfile 

print ("Reading data from file")

with mrcfile.open('ribo_inplane.mrc') as mrc:
    data = mrc.data

# arranged as (z,y,x)
print (data.shape)

# full autocorrelation
NX = 360
NY = 360

# flatten to save my brain
data = data.flatten()

auto_corr = np.zeros((NY,NX), dtype=np.float16)

counter = 0
for y in range(0, NY):
    for x in range(0, NX):
        if y <= x:
            # we are in the lower triangle so we can just grab the value
            # from the data array
            auto_corr[y,x] = data[counter]
            # only increment the counter in this branch as otherwise we are mirroring from self
            counter += 1
        else:
            if y == x:
                auto_corr[y,x] = 1.0
            else:
                # we are in the upper triangle so we can jsut grab from the lower triangle
                auto_corr[y,x] = auto_corr[x,y]
                


# 
# We recorded angles from 
auto_corr_angles = np.arange(0, 360, 1)

# We want to sub sample the data to speed up the calculation
test_angles = np.arange(0, 360, 10)

# Now (wastefully) just shring the matrix to the subsampled angles
auto_corr = auto_corr[:,test_angles]

print (auto_corr[1,:])