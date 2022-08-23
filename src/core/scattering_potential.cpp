/*
 * scattering_potential.cpp

 *
 *  Created on: Oct 3, 2019
 *      Author: himesb
 */
#include "core_headers.h"
#include "scattering_potential.h"

ScatteringPotential::ScatteringPotential( ) {

    pdb_file_names.reserve(16);
    pdb_ensemble.reserve(16);
    _wavelength = 0.0;
    _lead_term  = 0.0;
    _pixel_size = 0.0;
}

ScatteringPotential::~ScatteringPotential( ) {
}

void ScatteringPotential::InitPdbEnsemble(bool shift_by_cetner_of_mass, int minimum_padding_x_and_y, int minimum_thickness_z,
                                          int               max_number_of_noise_particles,
                                          float             wanted_noise_particle_radius_as_mutliple_of_particle_radius,
                                          float             wanted_noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius,
                                          float             wanted_noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius,
                                          float             wanted_tilt_angle_to_emulate,
                                          bool              is_alpha_fold_prediction,
                                          cisTEMParameters& wanted_star_file, bool use_star_file) {

    // backwards compatible with tigress where everything is double (ints would make more sense here.)
    long access_type_read = 0;
    long records_per_line = 1;

    MyDebugAssertTrue(_pixel_size > 0.0, "Pixel size not set");

    // Initialize each of the PDB objects, this reads in and centers each PDB, but does not make any copies (instances) of the trajectories.

    for ( int iPDB = 0; iPDB < pdb_file_names.size( ); iPDB++ ) {

        pdb_ensemble[iPDB] = PDB(pdb_file_names[iPDB], access_type_read, _pixel_size, records_per_line, minimum_padding_x_and_y, minimum_thickness_z,
                                 max_number_of_noise_particles,
                                 wanted_noise_particle_radius_as_mutliple_of_particle_radius,
                                 wanted_noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius,
                                 wanted_noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius,
                                 wanted_tilt_angle_to_emulate,
                                 shift_by_cetner_of_mass,
                                 is_alpha_fold_prediction,
                                 wanted_star_file, use_star_file);
    }
}

long ScatteringPotential::ReturnTotalNumberOfNonWaterAtoms( ) {

    long number_of_non_water_atoms = 0;
    // Get a count of the total non water atoms
    for ( int iPDB = 0; iPDB < pdb_ensemble.size( ); iPDB++ ) {
        number_of_non_water_atoms += (pdb_ensemble[iPDB].number_of_real_and_noise_atoms * pdb_ensemble[iPDB].number_of_particles_initialized);
    }

    return number_of_non_water_atoms;
}
