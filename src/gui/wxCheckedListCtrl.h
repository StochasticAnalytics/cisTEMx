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

//#include <wx/listctrl.h>

//IMPLEMENT_CLASS(wxCheckedListCtrl, wxListCtrl)

#ifndef __CHECKED_LISTCTRL_H__
#define __CHECKED_LISTCTRL_H__

#include <wx/listctrl.h>
#include <wx/imaglist.h>

class wxCheckedListCtrl : public wxListCtrl {
  protected:
    wxImageList m_imagelist;

  public:
    wxCheckedListCtrl(wxWindow* parent, wxWindowID id, const wxPoint& pt, const wxSize& sz, long style);
    bool Create(wxWindow* parent, wxWindowID id, const wxPoint& pt, const wxSize& sz, long style, const wxValidator& validator, const wxString& name);

    bool IsChecked(long item) const;
    void SetChecked(long item, bool checked);
};

#endif
