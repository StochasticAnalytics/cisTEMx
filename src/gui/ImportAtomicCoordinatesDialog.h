#ifndef __SRC_GUI__AtomicCoordinatesImportDialog__
#define __SRC_GUI__AtomicCoordinatesImportDialog__

class ImportAtomicCoordinatesDialog : public ImportAtomicCoordinatesDialogParent {
  public:
    /** Constructor */
    ImportAtomicCoordinatesDialog(wxWindow* parent);
    //// end generated class members

    void AddFilesClick(wxCommandEvent& event);
    void ClearClick(wxCommandEvent& event);
    void CancelClick(wxCommandEvent& event);
    void AddDirectoryClick(wxCommandEvent& event);
    void ImportClick(wxCommandEvent& event);
    void OnTextKeyPress(wxKeyEvent& event);
    void CheckImportButtonStatus( );
    void TextChanged(wxCommandEvent& event);

    const char* Type( ) const { return "ImportAtomicCoordinatesDialog"; };
};

#endif // __MyMovieImportDialog__
