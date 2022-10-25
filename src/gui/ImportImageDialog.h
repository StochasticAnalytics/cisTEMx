#ifndef __SRC_GUI_MyImageImportDialog__
#define __SRC_GUI_MyImageImportDialog__

class ImportImageDialog : public ImportImageDialogParent {
  public:
    /** Constructor */
    ImportImageDialog(wxWindow* parent);
    //// end generated class members

    void AddFilesClick(wxCommandEvent& event);
    void ClearClick(wxCommandEvent& event);
    void CancelClick(wxCommandEvent& event);
    void AddDirectoryClick(wxCommandEvent& event);
    void ImportClick(wxCommandEvent& event);
    void OnTextKeyPress(wxKeyEvent& event);
    void CheckImportButtonStatus( );
    void TextChanged(wxCommandEvent& event);
};

#endif // __MyMovieImportDialog__
