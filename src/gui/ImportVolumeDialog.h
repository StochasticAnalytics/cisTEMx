#ifndef __SRC_GUI__MyVolumeImportDialog__
#define __SRC_GUI__MyVolumeImportDialog__

class ImportVolumeDialog : public ImportVolumeDialogParent {
  public:
    /** Constructor */
    ImportVolumeDialog(wxWindow* parent);
    //// end generated class members

    void AddFilesClick(wxCommandEvent& event);
    void ClearClick(wxCommandEvent& event);
    void CancelClick(wxCommandEvent& event);
    void AddDirectoryClick(wxCommandEvent& event);
    void ImportClick(wxCommandEvent& event);
    void OnTextKeyPress(wxKeyEvent& event);
    void CheckImportButtonStatus( );
    void TextChanged(wxCommandEvent& event);

    const char* Type( ) const { return "ImportVolumeDialog"; };
};

#endif // __MyMovieImportDialog__
