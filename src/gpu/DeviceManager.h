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

#ifndef _SRC_GPU_DEVICEMANAGER_H_
#define _SRC_GPU_DEVICEMANAGER_H_

class DeviceManager {

  public:
    int  nGPUs;
    int  gpuIDX;
    bool is_manager_initialized = false;

    DeviceManager( );
    ~DeviceManager( );

    void Init(int wanted_number_of_gpus, MyApp* parent_ptr);
    void SetGpu( );
    void ResetGpu( );
    void ListDevices( );

  private:
};

#endif // _SRC_GPU_DEVICEMANAGER_H_
