/*
 * scattering_potential.h
 *
 *  Created on: Oct 3, 2019
 *      Author: himesb
 */

#ifndef _SRC_PROGRAMS_SIMULATE_SCATTERING_POTENTIAL_H_
#define _SRC_PROGRAMS_SIMULATE_SCATTERING_POTENTIAL_H_

#include "constants/constants.h"

// TODO: x2 = x1 + pixel size, so it might make more sense to limit memory and just store x1,y1,z1 and pixel size.
struct corners {

    float x1;
    float x2;
    float y1;
    float y2;
    float z1;
    float z2;
};

class ScatteringPotential {

  public:
    ScatteringPotential( );
    virtual ~ScatteringPotential( );

    std::vector<PDB>      pdb_ensemble;
    std::vector<wxString> pdb_file_names;

    void InitPdbEnsemble(bool              shift_by_cetner_of_mass,
                         int               minimum_padding_x_and_y,
                         int               minimum_thickness_z,
                         int               max_number_of_noise_particles,
                         float             wanted_noise_particle_radius_as_mutliple_of_particle_radius,
                         float             wanted_noise_particle_radius_randomizer_lower_bound_as_praction_of_particle_radius,
                         float             wanted_noise_particle_radius_randomizer_upper_bound_as_praction_of_particle_radius,
                         float             wanted_tilt_angle_to_emulat,
                         bool              is_alpha_fold_prediction,
                         cisTEMParameters& wanted_star_file,
                         bool              use_star_file);

    long ReturnTotalNumberOfNonWaterAtoms( );

    static inline float ReturnScatteringParamtersA(AtomType id, int term_number) { return SCATTERING_PARAMETERS_A[id][term_number]; }

    static inline float ReturnScatteringParamtersB(AtomType id, int term_number) { return SCATTERING_PARAMETERS_B[id][term_number]; }

    static inline float ReturnAtomicNumber(AtomType id) { return ATOMIC_NUMBER[id]; }

    void SetImagingParameters(float wanted_pixel_size, float wanted_kV, float wanted_scaling = cistem::bond_scaling) {
        _pixel_size = wanted_pixel_size;
        _wavelength = ReturnWavelenthInAngstroms(wanted_kV);
        _lead_term  = wanted_scaling * _wavelength / 8.0f / _pixel_size / _pixel_size;
    }

    inline float lead_term( ) const { return _lead_term; }

    inline float ReturnScatteringPotentialOfAVoxel(corners& R, float* bPlusB, AtomType& atom_id) {

        MyDebugAssertTrue(_wavelength > 0.0, "Wavelength not set");
        MyDebugAssertTrue(_lead_term > 0.0, "Wavelength not set");
        float temp_potential = 0.0f;
        float t0;
        bool  t1, t2, t3;

        // if product < 0, we need to sum the two independent terms, otherwise we want the difference.
        t1 = R.x1 * R.x2 < 0 ? false : true;
        t2 = R.y1 * R.y2 < 0 ? false : true;
        t3 = R.z1 * R.z2 < 0 ? false : true;

        for ( int iGaussian = 0; iGaussian < 5; iGaussian++ ) {

            t0 = (t1) ? erff(bPlusB[iGaussian] * R.x2) - erff(bPlusB[iGaussian] * R.x1) : fabsf(erff(bPlusB[iGaussian] * R.x2)) + fabsf(erff(bPlusB[iGaussian] * R.x1));
            t0 *= (t2) ? erff(bPlusB[iGaussian] * R.y2) - erff(bPlusB[iGaussian] * R.y1) : fabsf(erff(bPlusB[iGaussian] * R.y2)) + fabsf(erff(bPlusB[iGaussian] * R.y1));
            t0 *= (t3) ? erff(bPlusB[iGaussian] * R.z2) - erff(bPlusB[iGaussian] * R.z1) : fabsf(erff(bPlusB[iGaussian] * R.z2)) + fabsf(erff(bPlusB[iGaussian] * R.z1));

            temp_potential += ReturnScatteringParamtersA(atom_id, iGaussian) * fabsf(t0);

        } // loop over gaussian fits

        return temp_potential *= _lead_term;
    };

  private:
    float _lead_term;
    float _wavelength;
    float _pixel_size;
};

#endif /* PROGRAMS_SIMULATE_SCATTERING_POTENTIAL_H_ */
