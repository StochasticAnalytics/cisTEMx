#ifndef __SRC_GUI__MyMovieImportDialog__
#define __SRC_GUI__MyMovieImportDialog__

class ImportMovieDialog : public ImportMovieDialogParent {
  public:
    /** Constructor */
    ImportMovieDialog(wxWindow* parent);
    //// end generated class members

    void AddFilesClick(wxCommandEvent& event);
    void ClearClick(wxCommandEvent& event);
    void CancelClick(wxCommandEvent& event);
    void AddDirectoryClick(wxCommandEvent& event);
    void ImportClick(wxCommandEvent& event);
    void OnTextKeyPress(wxKeyEvent& event);
    void CheckImportButtonStatus( );
    void TextChanged(wxCommandEvent& event);

    void OnMoviesAreGainCorrectedCheckBox(wxCommandEvent& event);
    void OnCorrectMagDistortionCheckBox(wxCommandEvent& event);
    void OnGainFilePickerChanged(wxFileDirPickerEvent& event);
    void OnResampleMoviesCheckBox(wxCommandEvent& event);
    void OnSkipFullIntegrityCheckCheckBox(wxCommandEvent& event);

    const char* Type( ) const { return "ImportMovieDialog"; };
};

#endif // __MyMovieImportDialog__
