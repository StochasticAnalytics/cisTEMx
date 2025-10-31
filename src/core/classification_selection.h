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

class ClassificationSelection {

  public:
    ClassificationSelection( );
    ~ClassificationSelection( );

    long       selection_id;
    wxString   name;
    wxDateTime creation_date;
    long       refinement_package_asset_id;
    long       classification_id;
    int        number_of_classes;
    int        number_of_selections;

    wxArrayLong selections;
};

WX_DECLARE_OBJARRAY(ClassificationSelection, ArrayofClassificationSelections);
