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

/*  \brief  SymmetryMatrix class */

// Matrices generated using Frealign code, implemented in tool generate_symmetry_matrices.exe

class SymmetryMatrix {

  public:
    wxString        symmetry_symbol;
    int             number_of_matrices;
    RotationMatrix* rot_mat; /* 3D rotation matrix array*/

    SymmetryMatrix( );
    SymmetryMatrix(wxString wanted_symmetry_symbol);
    ~SymmetryMatrix( );

    SymmetryMatrix& operator=(const SymmetryMatrix& other_matrix);
    SymmetryMatrix& operator=(const SymmetryMatrix* other_matrix);

    void Init(wxString wanted_symmetry_symbol);

    void PrintMatrices( );
};
