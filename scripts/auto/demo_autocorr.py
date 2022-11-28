#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import csaps
import mrcfile 
import sys

print ("Reading data from file")

with mrcfile.open('auto_0_360_400_4.mrc') as mrc:
    data = mrc.data

# arranged as (z,y,x)
print (data.shape)

# full autocorrelation
NX = 360
NY = 360


# flatten to save my brain
data = data.flatten()

auto_corr = np.zeros((NY,NX), dtype=np.float32)

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
test_angles = np.arange(0, 360, 6)

# Now (wastefully) just shring the matrix to the subsampled angles
auto_corr = auto_corr[:,test_angles ]

# Now we can do the autocorrelation
N = len(test_angles)
smooth = False
smoothval = 0.5
for ifdx in range(6,8):
    file = 'search_noise' + str(ifdx) + '.txt'
    print ("Reading data from file: ", file)
    r = np.loadtxt(file)
    

    # The first entry is bogus
    r = r[1:]

    # print(r[1:10,1])
    
    plot_vals = True
    plot_idx = 31
    if plot_vals:

        # r = r - np.min(r)
        # r = r / np.max(r)
        # r = np.sqrt(r)
        r = r - np.mean(r)
        r = r / np.std(r)
        
        sp = csaps.CubicSmoothingSpline(test_angles, r, smooth=smoothval)
        norm_input = auto_corr[plot_idx,:]

        # norm_input = norm_input - np.min(norm_input)
        # norm_input = norm_input / np.max(norm_input)
        # norm_input = np.sqrt(norm_input)
        norm_input = norm_input - np.mean(norm_input)
        norm_input = norm_input / np.std(norm_input)
        sn = csaps.CubicSmoothingSpline(test_angles, norm_input, smooth=smoothval)

        
        # Now we can plot the data to visualize the smoothing
        xs = np.linspace(test_angles[0], test_angles[-1], 150)
        ys = sp(xs)
        yn = sn(xs)


        plt.plot(test_angles, r, 'bo', xs, ys, 'r-', xs, yn, 'g-')
        plt.show()
        sys.exit(0)


    # What was the max value and index ound
    max_val = np.max(r)
    max_idx = np.argmax(r)

    second = r
    second[max_idx] = 0;
    second_max_val = np.max(second)
    second_max_idx = np.argmax(second)

    print("Max value is %f at index %d at angle %f" % (max_val, max_idx, test_angles[max_idx]))
    print("Second max value is %f at index %d at angle %f" % (second_max_val, second_max_idx, test_angles[second_max_idx]))

    # normalize the result vector
    r = r - np.mean(r)
    r = r / np.std(r)

    if smooth:
        sp = csaps.CubicSmoothingSpline(test_angles, r, smooth=smoothval)

    max_score = -1000
    second_max_score = -1000
    for i in range(auto_corr.shape[0]):
        norm_input = auto_corr[i,:]
        # norm_input = norm_input - np.mean(norm_input)
        # norm_input = norm_input / np.max(norm_input)

        # norm_input = norm_input - np.min(norm_input)
        # norm_input = norm_input / np.max(norm_input)
        # norm_input = np.sqrt(norm_input)
        norm_input = norm_input - np.mean(norm_input)
        norm_input = norm_input / np.std(norm_input)
        # print(norm_input[1:10])
        # sys.exit
        if smooth:
            score = np.dot(sp(test_angles) ,norm_input) / N
        else:
            score = np.dot(r , norm_input) / N

        # print("Score for %d is %f" % (i, score))
        if score > second_max_score:
            if score > max_score:
                second_max_score = max_score
                second_max_idx = max_idx
                max_score = score
                max_idx = i
            else:
                second_max_score = score
                second_max_idx = i


    print("Max score is %f at index %d at angle %f" % (max_score, max_idx, auto_corr_angles[max_idx]))
    print("Second max score is %f at index %d at angle %f \n\n" % (second_max_score, second_max_idx, auto_corr_angles[second_max_idx]))