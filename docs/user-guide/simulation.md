---
title: "Simulation"
status: legacy
description: "Cryo-EM simulation tools and workflows - extracted from cisTEM developmental-docs"
content_type: tutorial
audience: [user, researcher]
level: intermediate
topics: [simulation, cryo-em, image-processing]
date_extracted: 2025-11-05
source: developmental-docs
---

# Simulation

Simulation capabilities in cisTEMx enable researchers to generate synthetic cryo-EM data for testing algorithms, validating workflows, and training purposes.

=== "Basic"

    ## Quick Start

    Generate a simple particle simulation with default parameters.

    ### Prerequisites

    - cisTEMx installed with simulation support
    - A PDB file of your molecule
    - Basic understanding of cryo-EM imaging parameters

    ### Simple Simulation

    ```bash
    # Simulate a single particle
    cistem simulate \
        --input your_protein.pdb \
        --output simulated_particle.mrc \
        --pixel-size 1.0 \
        --voltage 300 \
        --defocus 10000
    ```

    ### Basic Parameters

    - **pixel-size**: Pixel size in Angstroms (typically 0.5-2.0)
    - **voltage**: Acceleration voltage in kV (typically 200 or 300)
    - **defocus**: Defocus in Angstroms (positive = underfocus)
    - **dose**: Electron dose in e⁻/Å²

    ### Viewing Results

    Use any MRC viewer to examine your simulated images:
    ```bash
    # Example with ImageJ
    imagej simulated_particle.mrc

    # Example with 3dmod (IMOD)
    3dmod simulated_particle.mrc
    ```

    ### Common Use Cases

    - **Algorithm testing**: Generate ground truth data
    - **Pipeline validation**: Test processing workflows
    - **Training**: Learn cryo-EM concepts with controlled data
    - **Method development**: Compare algorithms on identical data

=== "Advanced"

    ## Detailed Simulation Workflows

    Advanced simulation techniques for realistic cryo-EM data generation.

    ### Understanding Simulation Components

    Simulation in TEM involves describing:

    1. **The Coulomb potential** of the sample
    2. **The relativistic wave function** of the imaging electrons
    3. **The interaction** between potential and wave function
    4. **The microscope optics**, including lenses and detectors

    ### The Frozen-Plasmon Method

    Image simulation plays a central role in the development and practice of high-resolution electron microscopy. The **Frozen-Plasmon method** explicitly models spatially variable inelastic scattering processes in cryo-EM specimens.

    **Key advantages:**

    - Produces amplitude contrast that depends on atomic composition
    - Reproduces the total inelastic mean free path as observed experimentally
    - Allows incorporation of radiation damage in the simulation
    - No post hoc scaling required for solvent contrast

    **Recommended reading:** [Frozen-Plasmon theory paper on bioRxiv](https://www.biorxiv.org/content/10.1101/2021.02.19.431636v1)

    ### Shot Noise vs Structural Noise

    Understanding the two primary sources of noise in cryo-EM simulations:

    **Shot Noise:**
    - Arises from low-dose imaging conditions
    - Uncertainty in electron arrival at detector
    - Quantum nature of electron counting
    - Simulated by Poisson statistics

    **Structural Noise:**
    - Additional signal from solvent molecules
    - Neighboring particles in crowded environments
    - Ice thickness variations
    - Non-uniform specimen composition

    ### Realistic Simulation Parameters

    For simulations that match experimental data:

    ```bash
    cistem simulate \
        --input protein.pdb \
        --output realistic_stack.mrc \
        --pixel-size 1.0 \
        --voltage 300 \
        --defocus 12000 \
        --dose 30 \
        --frames 30 \
        --ice-thickness 250 \
        --add-noise-particles 6 \
        --random-angles
    ```

    **Parameters explained:**

    - `--frames 30`: Movie simulation with dose fractionation
    - `--ice-thickness 250`: Minimum ice thickness in Angstroms
    - `--add-noise-particles 6`: Add neighboring particles
    - `--random-angles`: Randomize particle orientations

    ### PDB File Requirements

    !!! warning "PDB Format"
        Only classic [PDB format](https://en.wikipedia.org/wiki/Protein_Data_Bank_(file_format)) is currently supported. For PDBx/mmCIF files, convert using:

        ```bash
        # Using Chimera
        chimera --nogui --script "open model.cif; write format pdb #0 model.pdb"
        ```

    !!! tip "Biological Assembly"
        Some PDB files only include the asymmetric unit. When downloading, select "Biological Assembly" to get all atoms.

    ### Simulating Different Conditions

    **In vacuo (no solvent):**
    ```bash
    cistem simulate --input protein.pdb --output vacuum.mrc --water-scaling 0.0
    ```

    **With solvent:**
    ```bash
    cistem simulate --input protein.pdb --output water.mrc --water-scaling 1.0
    ```

    **High defocus (increased visibility):**
    ```bash
    cistem simulate --input protein.pdb --output highdefocus.mrc --defocus 24000
    ```

    !!! note "Defocus and Information"
        Higher defocus increases particle visibility but reduces high-resolution information content.

    ### Effect of Ice Thickness

    Thicker ice reduces contrast:

    ```bash
    # Thin ice (better contrast)
    cistem simulate --ice-thickness 250 --output thin_ice.mrc

    # Thick ice (reduced contrast)
    cistem simulate --ice-thickness 2000 --output thick_ice.mrc
    ```

    ### Movie Simulation

    Simulate dose-fractionated movies:

    ```bash
    cistem simulate \
        --input protein.pdb \
        --output movie.mrc \
        --dose-per-frame 1.5 \
        --frames 20 \
        --dose-rate 3.0
    ```

    **Parameters:**
    - `--dose-per-frame`: Electron dose per frame (e⁻/Å²)
    - `--frames`: Number of movie frames
    - `--dose-rate`: Dose rate (e⁻/pixel/s)

=== "Developer"

    ## Legacy cisTEM Simulation Tutorial

    This section contains the complete tutorial from the cisTEM developmental-docs repository.

    !!! note "Historical Context"
        This tutorial documents the original cisTEM simulation workflow. cisTEMx may have different commands and options.

    ### Tutorial 1: Get cisTEM Alpha Version

    To run the simulator or template matching you will need an alpha version of cisTEM. For most people, the best choice will be to download a pre-compiled binary. For those interested in compiling from source code, you will need to add the `--enable-experimental` flag to your configure line.

    ### Tutorial 2: Calculate 3D Scattering Potential

    The first step in template matching is calculating the 3D scattering potential. The SNR in template matching depends strongly on how well your calculated specimen potential matches the imaged specimen potential.

    **Materials needed:**
    - Alpha version of cisTEM
    - Imaging condition information
    - PDB file representing your molecule

    ### Tutorial 3: Calculate Stack of Noisy Particles

    **Download materials:**
    ```bash
    wget https://github.com/bHimes/cisTEM_docs/raw/main/tutorial_materials/bgal_flat.pdb
    wget https://github.com/bHimes/cisTEM_docs/raw/main/tutorial_materials/bgal_flat.sh
    chmod u+x bgal_flat.sh
    ```

    #### Part A: Simulate an Isolated Protein

    The tutorial script produces the projected Coulomb potential of beta-galactosidase.

    **Basic parameters:**
    ```bash
    output_filename="betgal_vacuum.mrc"
    input_pdb_file="bgal_flat.pdb"
    output_size=-320  # pixels
    n_threads=16
    optional_args=" --save-detector-wavefunction --skip-random-angles "
    ```

    **Progression through noise levels:**

    1. **Vacuum (no solvent):**
       - Cleanest image
       - Unrealistic - proteins don't exist in vacuum
       - Shows protein structure clearly

    2. **Add solvent (water_scaling=1.0):**
       - Much noisier appearance
       - Structural noise from water molecules
       - More realistic but still "perfect" (no shot noise)

    3. **Add shot noise (remove --save-detector-wavefunction):**
       - Quantum uncertainty in electron arrival
       - Low-dose imaging conditions
       - 1 e⁻/Å² example

    4. **Realistic dose (30 e⁻/Å² over 20 frames):**
       ```bash
       output_filename="betgal_30elec_per_angSq.mrc"
       exposure_per_frame=1.5  # e-/Ang^2
       exposure_rate=3.0       # e-/pixel/s
       n_frames=20
       pre_exposure=0
       ```

    5. **High defocus for visibility:**
       ```bash
       output_filename="betgal_30elec_per_angSq_highdefocus.mrc"
       wanted_defocus=240000  # Angstrom (24000Å)
       ```

    **Important insight:** Improved visibility at high defocus does NOT mean more information - in fact, high-resolution information is reduced.

    #### Part B: Simulate More Realistic Protein

    **Random orientations:**
    ```bash
    output_filename="betgal.mrc"
    wanted_defocus=8000
    n_frames=1
    water_scaling=0.5
    optional_args=" --save-detector-wavefunction"
    ```

    !!! note
        Your particles will be in different orientations than the examples. You may need to adjust image contrast in your display software.

    **Thick ice layer:**
    ```bash
    output_filename="betgal_thick.mrc"
    minimum_thickness=2000  # Angstrom (200nm)
    ```

    !!! question "Visibility Exercise"
        Can you see a ~450kDa protein in 200nm ice when solvent is multiplied by 0.5?

    **Add neighboring particles:**
    ```bash
    output_filename="betgal_neighbors.mrc"
    minimum_thickness=250  # Angstrom (25nm)
    optional_args=" --save-detector-wavefunction --max_number_of_noise_particles=6 "
    ```

    **Fully realistic simulation:**
    ```bash
    output_filename="betgal_realistic.mrc"
    minimum_thickness=250
    water_scaling=1.0
    wanted_defocus=12000
    n_frames=30
    optional_args=" --max_number_of_noise_particles=6 "
    ```

    ### Key Concepts Summary

    - **Shot noise**: Quantum uncertainty from low-dose imaging
    - **Structural noise**: Signal from solvent and neighboring particles
    - **Defocus trade-off**: Visibility vs high-resolution information
    - **Ice thickness**: Thicker ice reduces particle contrast
    - **Dose fractionation**: Spreading exposure across movie frames
    - **Realistic conditions**: Combining all factors for experimental matching

    ### See Also

    - **Template Matching**: Using simulations for particle detection
    - **PDB File Handling**: Working with molecular structures
    - **Dose Calculations**: Electron dose and damage modeling

---

!!! warning "Legacy Documentation"
    This content was extracted from the cisTEM developmental-docs repository and may be outdated.
    It documents the Frozen-Plasmon simulation method implemented in cisTEM. cisTEMx may have evolved significantly.
