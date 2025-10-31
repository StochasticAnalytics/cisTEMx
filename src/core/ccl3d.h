/*
 * Original Copyright (c) 2017, Howard Hughes Medical Institute
 * Licensed under Janelia Research Campus Software License 1.2
 * See license_details/LICENSE-JANELIA.txt
 *
 * Modifications Copyright (c) 2025, Stochastic Analytics, LLC
 * Modifications licensed under MPL 2.0 for academic use; 
 * commercial license required for commercial use.
 * See LICENSE.md for details.
 */

class ccl3d {

  public:
    int* parent;
    ccl3d(Image& input3d);
    ~ccl3d( );
    int  Max(int i, int j, int k);
    int  Med(int i, int j, int k);
    int  Min(int i, int j, int k);
    int  Find(int x, int parent[]);
    void Union(int big, int small, int parent[]);
    void Two_noZero(int a, int b, int c);
    void GetLargestConnectedDensityMask(Image& input3d, Image& output_largest_connected_density3d);
};
