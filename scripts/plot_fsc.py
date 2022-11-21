#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np



# Create a simply entry point and take command line arguments for the file name and pixel size
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: plot_fsc.py <file name> <pixel size>')
        sys.exit(1)
    pixel_size = float(sys.argv[2])
    data = np.loadtxt(sys.argv[1],comments='C')
    first_val = (data[:,4] < 0.143).argmax()
    interp_val = np.interp(0.143,data[first_val-1:first_val+1,4],data[first_val-1:first_val+1,2])
    print(pixel_size / interp_val)

    plt.subplot(111)
    plt.plot(pixel_size * data[:,2],  data[:,4])
    plt.xlabel('Resolution (1/A)')
    plt.ylabel('FSC')
    plt.title('FSC curve')


    # add a subtitle with  the resolution at 0.143 above the plo
    plt.suptitle('Resolution at 0.143: {:.2f} A'.format(pixel_size / interp_val), horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    # print the resolution at 0.143
    plt.axvline(x=interp_val, color='r', linestyle='--')
    plt.axhline(y=0, color='k', linestyle='--')


    # set the x limit to 0 and the y limit to 1
    plt.xlim(0,pixel_size * data[-1,2])
    plt.show()
