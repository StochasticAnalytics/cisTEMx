///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Jan 30 2016)
// http://www.wxformbuilder.org/
//
// PLEASE DO "NOT" EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#include "../core/gui_core_headers.h"
#include "AngularDistributionPlotPanel.h"
#include "BitmapPanel.h"
#include "CTF1DPanel.h"
#include "ClassificationPlotPanel.h"
#include "DisplayPanel.h"
#include "MyFSCPanel.h"
#include "PickingResultsDisplayPanel.h"
#include "PlotFSCPanel.h"
#include "ResultsDataViewListCtrl.h"
#include "ShowCTFResultsPanel.h"
#include "UnblurResultsPanel.h"
#include "job_panel.h"
#include "my_controls.h"

#include "ProjectX_gui.h"

///////////////////////////////////////////////////////////////////////////

MainFrame::MainFrame( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxFrame( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( 1366,768 ), wxDefaultSize );
	
	wxBoxSizer* bSizer2;
	bSizer2 = new wxBoxSizer( wxVERTICAL );
	
	LeftPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxRAISED_BORDER|wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer8;
	bSizer8 = new wxBoxSizer( wxVERTICAL );
	
	MenuBook = new wxListbook( LeftPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLB_LEFT );
	#ifdef __WXGTK__ // Small icon style not supported in GTK
	wxListView* MenuBookListView = MenuBook->GetListView();
	long MenuBookFlags = MenuBookListView->GetWindowStyleFlag();
	if( MenuBookFlags & wxLC_SMALL_ICON )
	{
		MenuBookFlags = ( MenuBookFlags & ~wxLC_SMALL_ICON ) | wxLC_ICON;
	}
	MenuBookListView->SetWindowStyleFlag( MenuBookFlags );
	#endif
	
	bSizer8->Add( MenuBook, 1, wxEXPAND | wxALL, 5 );
	
	
	LeftPanel->SetSizer( bSizer8 );
	LeftPanel->Layout();
	bSizer8->Fit( LeftPanel );
	bSizer2->Add( LeftPanel, 1, wxEXPAND | wxALL, 5 );
	
	
	this->SetSizer( bSizer2 );
	this->Layout();
	m_menubar1 = new wxMenuBar( 0 );
	FileMenu = new wxMenu();
	wxMenuItem* FileNewProject;
	FileNewProject = new wxMenuItem( FileMenu, wxID_ANY, wxString( wxT("New Project") ) , wxEmptyString, wxITEM_NORMAL );
	FileMenu->Append( FileNewProject );
	
	wxMenuItem* FileOpenProject;
	FileOpenProject = new wxMenuItem( FileMenu, wxID_ANY, wxString( wxT("Open Project") ) + wxT('\t') + wxT("Ctrl-O"), wxEmptyString, wxITEM_NORMAL );
	FileMenu->Append( FileOpenProject );
	
	wxMenuItem* FileCloseProject;
	FileCloseProject = new wxMenuItem( FileMenu, wxID_ANY, wxString( wxT("Close Project") ) , wxEmptyString, wxITEM_NORMAL );
	FileMenu->Append( FileCloseProject );
	
	FileMenu->AppendSeparator();
	
	wxMenuItem* FileExit;
	FileExit = new wxMenuItem( FileMenu, wxID_ANY, wxString( wxT("Exit") ) , wxEmptyString, wxITEM_NORMAL );
	FileMenu->Append( FileExit );
	
	m_menubar1->Append( FileMenu, wxT("Project") ); 
	
	ExportMenu = new wxMenu();
	wxMenuItem* ExportCoordinatesToImagic;
	ExportCoordinatesToImagic = new wxMenuItem( ExportMenu, wxID_ANY, wxString( wxT("Export coordinates to Imagic") ) , wxEmptyString, wxITEM_NORMAL );
	ExportMenu->Append( ExportCoordinatesToImagic );
	
	wxMenuItem* ExportToFrealign;
	ExportToFrealign = new wxMenuItem( ExportMenu, wxID_ANY, wxString( wxT("Export particles to Frealign") ) , wxEmptyString, wxITEM_NORMAL );
	ExportMenu->Append( ExportToFrealign );
	
	wxMenuItem* ExportToRelion;
	ExportToRelion = new wxMenuItem( ExportMenu, wxID_ANY, wxString( wxT("Export particles to Relion") ) , wxEmptyString, wxITEM_NORMAL );
	ExportMenu->Append( ExportToRelion );
	
	m_menubar1->Append( ExportMenu, wxT("Export") ); 
	
	this->SetMenuBar( m_menubar1 );
	
	
	this->Centre( wxBOTH );
	
	// Connect Events
	MenuBook->Connect( wxEVT_COMMAND_LISTBOOK_PAGE_CHANGED, wxListbookEventHandler( MainFrame::OnMenuBookChange ), NULL, this );
	m_menubar1->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( MainFrame::OnFileMenuUpdate ), NULL, this );
	this->Connect( FileNewProject->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnFileNewProject ) );
	this->Connect( FileOpenProject->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnFileOpenProject ) );
	this->Connect( FileCloseProject->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnFileCloseProject ) );
	this->Connect( FileExit->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnFileExit ) );
	this->Connect( ExportCoordinatesToImagic->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnExportCoordinatesToImagic ) );
	this->Connect( ExportToFrealign->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnExportToFrealign ) );
	this->Connect( ExportToRelion->GetId(), wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnExportToRelion ) );
}

MainFrame::~MainFrame()
{
	// Disconnect Events
	MenuBook->Disconnect( wxEVT_COMMAND_LISTBOOK_PAGE_CHANGED, wxListbookEventHandler( MainFrame::OnMenuBookChange ), NULL, this );
	m_menubar1->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( MainFrame::OnFileMenuUpdate ), NULL, this );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnFileNewProject ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnFileOpenProject ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnFileCloseProject ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnFileExit ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnExportCoordinatesToImagic ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnExportToFrealign ) );
	this->Disconnect( wxID_ANY, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler( MainFrame::OnExportToRelion ) );
	
}

Refine2DPanel::Refine2DPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : JobPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer43;
	bSizer43 = new wxBoxSizer( wxVERTICAL );
	
	m_staticline12 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline12, 0, wxEXPAND | wxALL, 5 );
	
	InputParamsPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer451;
	bSizer451 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer441;
	bSizer441 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer200;
	bSizer200 = new wxBoxSizer( wxHORIZONTAL );
	
	wxFlexGridSizer* fgSizer14;
	fgSizer14 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer14->SetFlexibleDirection( wxBOTH );
	fgSizer14->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticText262 = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("Input Refinement Package :"), wxDefaultPosition, wxSize( -1,-1 ), 0 );
	m_staticText262->Wrap( -1 );
	fgSizer14->Add( m_staticText262, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	RefinementPackageComboBox = new MemoryComboBox( InputParamsPanel, wxID_ANY, wxT("Combo!"), wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	RefinementPackageComboBox->SetMinSize( wxSize( 300,-1 ) );
	
	fgSizer14->Add( RefinementPackageComboBox, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText263 = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("Input Starting References :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText263->Wrap( -1 );
	fgSizer14->Add( m_staticText263, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	InputParametersComboBox = new MemoryComboBox( InputParamsPanel, wxID_ANY, wxT("Combo!"), wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	fgSizer14->Add( InputParametersComboBox, 0, wxALL|wxEXPAND, 5 );
	
	
	bSizer200->Add( fgSizer14, 1, wxEXPAND, 5 );
	
	m_staticline57 = new wxStaticLine( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer200->Add( m_staticline57, 0, wxEXPAND | wxALL, 5 );
	
	wxGridSizer* gSizer10;
	gSizer10 = new wxGridSizer( 0, 1, 0, 0 );
	
	wxBoxSizer* bSizer215;
	bSizer215 = new wxBoxSizer( wxHORIZONTAL );
	
	
	bSizer215->Add( 0, 0, 1, wxEXPAND, 5 );
	
	m_staticText3321 = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("No. of Classes :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText3321->Wrap( -1 );
	bSizer215->Add( m_staticText3321, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	NumberClassesSpinCtrl = new wxSpinCtrl( InputParamsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 1000, 50 );
	NumberClassesSpinCtrl->SetMinSize( wxSize( 100,-1 ) );
	
	bSizer215->Add( NumberClassesSpinCtrl, 0, wxALL, 5 );
	
	
	gSizer10->Add( bSizer215, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer216;
	bSizer216 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText264 = new wxStaticText( InputParamsPanel, wxID_ANY, wxT("No. of Cycles to Run :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText264->Wrap( -1 );
	bSizer216->Add( m_staticText264, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	NumberRoundsSpinCtrl = new wxSpinCtrl( InputParamsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 100, 20 );
	NumberRoundsSpinCtrl->SetMinSize( wxSize( 100,-1 ) );
	
	bSizer216->Add( NumberRoundsSpinCtrl, 1, wxALL|wxEXPAND, 5 );
	
	
	gSizer10->Add( bSizer216, 1, wxEXPAND, 5 );
	
	
	bSizer200->Add( gSizer10, 0, wxEXPAND, 5 );
	
	wxGridSizer* gSizer11;
	gSizer11 = new wxGridSizer( 2, 1, 0, 0 );
	
	wxBoxSizer* bSizer214;
	bSizer214 = new wxBoxSizer( wxHORIZONTAL );
	
	
	gSizer11->Add( bSizer214, 1, wxEXPAND, 5 );
	
	
	bSizer200->Add( gSizer11, 0, wxEXPAND, 5 );
	
	
	bSizer200->Add( 0, 0, 1, wxEXPAND, 5 );
	
	ExpertToggleButton = new wxToggleButton( InputParamsPanel, wxID_ANY, wxT("Show Expert Options"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer200->Add( ExpertToggleButton, 0, wxALIGN_BOTTOM|wxALL, 5 );
	
	
	bSizer441->Add( bSizer200, 1, wxEXPAND, 5 );
	
	m_staticline10 = new wxStaticLine( InputParamsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer441->Add( m_staticline10, 0, wxEXPAND | wxALL, 5 );
	
	
	bSizer451->Add( bSizer441, 1, wxEXPAND, 5 );
	
	
	InputParamsPanel->SetSizer( bSizer451 );
	InputParamsPanel->Layout();
	bSizer451->Fit( InputParamsPanel );
	bSizer43->Add( InputParamsPanel, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer46;
	bSizer46 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer279;
	bSizer279 = new wxBoxSizer( wxVERTICAL );
	
	OutputTextPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	OutputTextPanel->Hide();
	
	wxBoxSizer* bSizer56;
	bSizer56 = new wxBoxSizer( wxVERTICAL );
	
	output_textctrl = new wxTextCtrl( OutputTextPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE|wxTE_READONLY );
	bSizer56->Add( output_textctrl, 1, wxALL|wxEXPAND, 5 );
	
	
	OutputTextPanel->SetSizer( bSizer56 );
	OutputTextPanel->Layout();
	bSizer56->Fit( OutputTextPanel );
	bSizer279->Add( OutputTextPanel, 40, wxEXPAND | wxALL, 5 );
	
	PlotPanel = new ClassificationPlotPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	PlotPanel->Hide();
	
	bSizer279->Add( PlotPanel, 60, wxEXPAND | wxALL, 5 );
	
	
	bSizer46->Add( bSizer279, 40, wxEXPAND, 5 );
	
	ResultDisplayPanel = new DisplayPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ResultDisplayPanel->Hide();
	
	bSizer46->Add( ResultDisplayPanel, 60, wxEXPAND | wxALL, 5 );
	
	ExpertPanel = new wxScrolledWindow( this, wxID_ANY, wxDefaultPosition, wxSize( -1,-1 ), wxVSCROLL );
	ExpertPanel->SetScrollRate( 5, 5 );
	ExpertPanel->Hide();
	
	InputSizer = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer258;
	bSizer258 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText318 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Expert Options"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText318->Wrap( -1 );
	m_staticText318->SetFont( wxFont( 10, 74, 90, 92, true, wxT("Sans") ) );
	
	bSizer258->Add( m_staticText318, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer258->Add( 0, 0, 1, wxEXPAND, 5 );
	
	ResetAllDefaultsButton = new wxButton( ExpertPanel, wxID_ANY, wxT("Reset All Defaults"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer258->Add( ResetAllDefaultsButton, 0, wxALL, 5 );
	
	
	bSizer258->Add( 5, 0, 0, wxEXPAND, 5 );
	
	
	InputSizer->Add( bSizer258, 0, wxEXPAND, 5 );
	
	
	InputSizer->Add( 0, 5, 0, wxEXPAND, 5 );
	
	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer1->SetFlexibleDirection( wxBOTH );
	fgSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	NoMovieFramesStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Low-Resolution Limit (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	NoMovieFramesStaticText->Wrap( -1 );
	fgSizer1->Add( NoMovieFramesStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	LowResolutionLimitTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("300.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( LowResolutionLimitTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText188 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("High-Resolution Limit (Å) : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText188->Wrap( -1 );
	fgSizer1->Add( m_staticText188, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	HighResolutionLimitTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("8.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( HighResolutionLimitTextCtrl, 1, wxALL|wxEXPAND, 5 );
	
	m_staticText196 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Mask Radius (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText196->Wrap( -1 );
	fgSizer1->Add( m_staticText196, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	MaskRadiusTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( MaskRadiusTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	AngularStepStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Angular Search Step (°) :"), wxDefaultPosition, wxDefaultSize, 0 );
	AngularStepStaticText->Wrap( -1 );
	fgSizer1->Add( AngularStepStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	AngularStepTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("15.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( AngularStepTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	SearchRangeXStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Search Range in X (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	SearchRangeXStaticText->Wrap( -1 );
	fgSizer1->Add( SearchRangeXStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	SearchRangeXTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( SearchRangeXTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	SearchRangeYStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Search Range in Y (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	SearchRangeYStaticText->Wrap( -1 );
	fgSizer1->Add( SearchRangeYStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	SearchRangeYTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( SearchRangeYTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText330 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Smoothing Factor :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText330->Wrap( -1 );
	fgSizer1->Add( m_staticText330, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	SmoothingFactorTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("1"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( SmoothingFactorTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	PhaseShiftStepStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Exclude Blank Edges?"), wxDefaultPosition, wxDefaultSize, 0 );
	PhaseShiftStepStaticText->Wrap( -1 );
	fgSizer1->Add( PhaseShiftStepStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	wxBoxSizer* bSizer263;
	bSizer263 = new wxBoxSizer( wxHORIZONTAL );
	
	ExcludeBlankEdgesYesRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer263->Add( ExcludeBlankEdgesYesRadio, 0, wxALL, 5 );
	
	ExcludeBlankEdgesNoRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer263->Add( ExcludeBlankEdgesNoRadio, 0, wxALL, 5 );
	
	
	fgSizer1->Add( bSizer263, 1, wxEXPAND, 5 );
	
	m_staticText334 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Auto Percent Used?"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText334->Wrap( -1 );
	fgSizer1->Add( m_staticText334, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	wxBoxSizer* bSizer2631;
	bSizer2631 = new wxBoxSizer( wxHORIZONTAL );
	
	AutoPercentUsedRadioYes = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer2631->Add( AutoPercentUsedRadioYes, 0, wxALL, 5 );
	
	SphereClassificatonNoRadio1 = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer2631->Add( SphereClassificatonNoRadio1, 0, wxALL, 5 );
	
	
	fgSizer1->Add( bSizer2631, 1, wxEXPAND, 5 );
	
	PercentUsedStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tPercent Used :"), wxDefaultPosition, wxDefaultSize, 0 );
	PercentUsedStaticText->Wrap( -1 );
	PercentUsedStaticText->Enable( false );
	
	fgSizer1->Add( PercentUsedStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	PercentUsedTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, 0 );
	PercentUsedTextCtrl->Enable( false );
	
	fgSizer1->Add( PercentUsedTextCtrl, 0, wxALL, 5 );
	
	
	InputSizer->Add( fgSizer1, 1, wxEXPAND, 5 );
	
	
	ExpertPanel->SetSizer( InputSizer );
	ExpertPanel->Layout();
	InputSizer->Fit( ExpertPanel );
	bSizer46->Add( ExpertPanel, 0, wxALL|wxEXPAND, 5 );
	
	InfoPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer61;
	bSizer61 = new wxBoxSizer( wxVERTICAL );
	
	InfoText = new wxRichTextCtrl( InfoPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_READONLY|wxHSCROLL|wxVSCROLL );
	bSizer61->Add( InfoText, 1, wxEXPAND | wxALL, 5 );
	
	
	InfoPanel->SetSizer( bSizer61 );
	InfoPanel->Layout();
	bSizer61->Fit( InfoPanel );
	bSizer46->Add( InfoPanel, 20, wxEXPAND | wxALL, 5 );
	
	
	bSizer43->Add( bSizer46, 1, wxEXPAND, 5 );
	
	m_staticline11 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline11, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer48;
	bSizer48 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer70;
	bSizer70 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer71;
	bSizer71 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer55;
	bSizer55 = new wxBoxSizer( wxHORIZONTAL );
	
	
	bSizer71->Add( bSizer55, 0, wxALIGN_CENTER_HORIZONTAL, 5 );
	
	
	bSizer70->Add( bSizer71, 1, wxALIGN_CENTER_VERTICAL, 5 );
	
	wxBoxSizer* bSizer74;
	bSizer74 = new wxBoxSizer( wxVERTICAL );
	
	
	bSizer70->Add( bSizer74, 0, wxALIGN_CENTER_VERTICAL, 5 );
	
	
	bSizer48->Add( bSizer70, 1, wxEXPAND, 5 );
	
	ProgressPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ProgressPanel->Hide();
	
	wxBoxSizer* bSizer57;
	bSizer57 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer59;
	bSizer59 = new wxBoxSizer( wxHORIZONTAL );
	
	NumberConnectedText = new wxStaticText( ProgressPanel, wxID_ANY, wxT("o"), wxDefaultPosition, wxDefaultSize, 0 );
	NumberConnectedText->Wrap( -1 );
	bSizer59->Add( NumberConnectedText, 0, wxALIGN_CENTER|wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	ProgressBar = new wxGauge( ProgressPanel, wxID_ANY, 100, wxDefaultPosition, wxDefaultSize, wxGA_HORIZONTAL );
	ProgressBar->SetValue( 0 ); 
	bSizer59->Add( ProgressBar, 100, wxALIGN_CENTER|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );
	
	TimeRemainingText = new wxStaticText( ProgressPanel, wxID_ANY, wxT("Time Remaining : ???h:??m:??s   "), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	TimeRemainingText->Wrap( -1 );
	bSizer59->Add( TimeRemainingText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	m_staticline60 = new wxStaticLine( ProgressPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer59->Add( m_staticline60, 0, wxEXPAND | wxALL, 5 );
	
	FinishButton = new wxButton( ProgressPanel, wxID_ANY, wxT("Finish"), wxDefaultPosition, wxDefaultSize, 0 );
	FinishButton->Hide();
	
	bSizer59->Add( FinishButton, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	CancelAlignmentButton = new wxButton( ProgressPanel, wxID_ANY, wxT("Terminate Job"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer59->Add( CancelAlignmentButton, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer57->Add( bSizer59, 1, wxEXPAND, 5 );
	
	
	ProgressPanel->SetSizer( bSizer57 );
	ProgressPanel->Layout();
	bSizer57->Fit( ProgressPanel );
	bSizer48->Add( ProgressPanel, 1, wxEXPAND | wxALL, 5 );
	
	StartPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer58;
	bSizer58 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer268;
	bSizer268 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer267;
	bSizer267 = new wxBoxSizer( wxHORIZONTAL );
	
	RunProfileText = new wxStaticText( StartPanel, wxID_ANY, wxT("Classification Run Profile :"), wxDefaultPosition, wxDefaultSize, 0 );
	RunProfileText->Wrap( -1 );
	bSizer267->Add( RunProfileText, 20, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	RefinementRunProfileComboBox = new MemoryComboBox( StartPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	bSizer267->Add( RefinementRunProfileComboBox, 50, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );
	
	
	bSizer268->Add( bSizer267, 50, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer2671;
	bSizer2671 = new wxBoxSizer( wxHORIZONTAL );
	
	
	bSizer268->Add( bSizer2671, 0, wxEXPAND, 5 );
	
	
	bSizer58->Add( bSizer268, 50, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer60;
	bSizer60 = new wxBoxSizer( wxVERTICAL );
	
	
	bSizer60->Add( 0, 0, 1, wxEXPAND, 5 );
	
	StartRefinementButton = new wxButton( StartPanel, wxID_ANY, wxT("Start Classification"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer60->Add( StartRefinementButton, 0, wxALL, 5 );
	
	
	bSizer58->Add( bSizer60, 50, wxEXPAND, 5 );
	
	
	StartPanel->SetSizer( bSizer58 );
	StartPanel->Layout();
	bSizer58->Fit( StartPanel );
	bSizer48->Add( StartPanel, 1, wxEXPAND | wxALL, 5 );
	
	
	bSizer43->Add( bSizer48, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer43 );
	this->Layout();
	
	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( Refine2DPanel::OnUpdateUI ) );
	RefinementPackageComboBox->Connect( wxEVT_COMMAND_COMBOBOX_SELECTED, wxCommandEventHandler( Refine2DPanel::OnRefinementPackageComboBox ), NULL, this );
	InputParametersComboBox->Connect( wxEVT_COMMAND_COMBOBOX_SELECTED, wxCommandEventHandler( Refine2DPanel::OnInputParametersComboBox ), NULL, this );
	ExpertToggleButton->Connect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( Refine2DPanel::OnExpertOptionsToggle ), NULL, this );
	ResetAllDefaultsButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DPanel::ResetAllDefaultsClick ), NULL, this );
	HighResolutionLimitTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( Refine2DPanel::OnHighResLimitChange ), NULL, this );
	InfoText->Connect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( Refine2DPanel::OnInfoURL ), NULL, this );
	FinishButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DPanel::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DPanel::TerminateButtonClick ), NULL, this );
	StartRefinementButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DPanel::StartClassificationClick ), NULL, this );
}

Refine2DPanel::~Refine2DPanel()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( Refine2DPanel::OnUpdateUI ) );
	RefinementPackageComboBox->Disconnect( wxEVT_COMMAND_COMBOBOX_SELECTED, wxCommandEventHandler( Refine2DPanel::OnRefinementPackageComboBox ), NULL, this );
	InputParametersComboBox->Disconnect( wxEVT_COMMAND_COMBOBOX_SELECTED, wxCommandEventHandler( Refine2DPanel::OnInputParametersComboBox ), NULL, this );
	ExpertToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( Refine2DPanel::OnExpertOptionsToggle ), NULL, this );
	ResetAllDefaultsButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DPanel::ResetAllDefaultsClick ), NULL, this );
	HighResolutionLimitTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( Refine2DPanel::OnHighResLimitChange ), NULL, this );
	InfoText->Disconnect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( Refine2DPanel::OnInfoURL ), NULL, this );
	FinishButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DPanel::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DPanel::TerminateButtonClick ), NULL, this );
	StartRefinementButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DPanel::StartClassificationClick ), NULL, this );
	
}

RefinementResultsPanel::RefinementResultsPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer202;
	bSizer202 = new wxBoxSizer( wxVERTICAL );
	
	m_splitter7 = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitter7->SetSashSize( 10 );
	m_splitter7->Connect( wxEVT_IDLE, wxIdleEventHandler( RefinementResultsPanel::m_splitter7OnIdle ), NULL, this );
	
	m_panel48 = new wxPanel( m_splitter7, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer203;
	bSizer203 = new wxBoxSizer( wxVERTICAL );
	
	wxFlexGridSizer* fgSizer11;
	fgSizer11 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer11->AddGrowableCol( 1 );
	fgSizer11->SetFlexibleDirection( wxBOTH );
	fgSizer11->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticText284 = new wxStaticText( m_panel48, wxID_ANY, wxT("Select Refinement Package :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText284->Wrap( -1 );
	fgSizer11->Add( m_staticText284, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	RefinementPackageComboBox = new MemoryComboBox( m_panel48, wxID_ANY, wxT("Combo!"), wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	fgSizer11->Add( RefinementPackageComboBox, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText285 = new wxStaticText( m_panel48, wxID_ANY, wxT("Select Input Parameters :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText285->Wrap( -1 );
	fgSizer11->Add( m_staticText285, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	InputParametersComboBox = new MemoryComboBox( m_panel48, wxID_ANY, wxT("Combo!"), wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	fgSizer11->Add( InputParametersComboBox, 1, wxALL|wxEXPAND, 5 );
	
	
	bSizer203->Add( fgSizer11, 0, wxEXPAND, 5 );
	
	ParameterListCtrl = new RefinementParametersListCtrl( m_panel48, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_REPORT|wxLC_SINGLE_SEL|wxLC_VIRTUAL );
	bSizer203->Add( ParameterListCtrl, 1, wxALL|wxEXPAND, 5 );
	
	
	m_panel48->SetSizer( bSizer203 );
	m_panel48->Layout();
	bSizer203->Fit( m_panel48 );
	m_panel49 = new wxPanel( m_splitter7, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer206;
	bSizer206 = new wxBoxSizer( wxVERTICAL );
	
	FSCPlotPanel = new MyFSCPanel( m_panel49, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer206->Add( FSCPlotPanel, 60, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer208;
	bSizer208 = new wxBoxSizer( wxHORIZONTAL );
	
	AngularPlotPanel = new AngularDistributionPlotPanel( m_panel49, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer208->Add( AngularPlotPanel, 100, wxEXPAND | wxALL, 5 );
	
	
	bSizer206->Add( bSizer208, 40, wxEXPAND, 5 );
	
	
	m_panel49->SetSizer( bSizer206 );
	m_panel49->Layout();
	bSizer206->Fit( m_panel49 );
	m_splitter7->SplitVertically( m_panel48, m_panel49, 800 );
	bSizer202->Add( m_splitter7, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer202 );
	this->Layout();
	
	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( RefinementResultsPanel::OnUpdateUI ) );
	RefinementPackageComboBox->Connect( wxEVT_COMMAND_COMBOBOX_SELECTED, wxCommandEventHandler( RefinementResultsPanel::OnRefinementPackageComboBox ), NULL, this );
	InputParametersComboBox->Connect( wxEVT_COMMAND_COMBOBOX_SELECTED, wxCommandEventHandler( RefinementResultsPanel::OnInputParametersComboBox ), NULL, this );
}

RefinementResultsPanel::~RefinementResultsPanel()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( RefinementResultsPanel::OnUpdateUI ) );
	RefinementPackageComboBox->Disconnect( wxEVT_COMMAND_COMBOBOX_SELECTED, wxCommandEventHandler( RefinementResultsPanel::OnRefinementPackageComboBox ), NULL, this );
	InputParametersComboBox->Disconnect( wxEVT_COMMAND_COMBOBOX_SELECTED, wxCommandEventHandler( RefinementResultsPanel::OnInputParametersComboBox ), NULL, this );
	
}

Refine2DResultsPanelParent::Refine2DResultsPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer202;
	bSizer202 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer265;
	bSizer265 = new wxBoxSizer( wxVERTICAL );
	
	m_staticline67 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer265->Add( m_staticline67, 0, wxEXPAND | wxALL, 5 );
	
	
	bSizer202->Add( bSizer265, 0, wxEXPAND, 5 );
	
	m_splitter7 = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D|wxSP_3DBORDER|wxSP_3DSASH|wxSP_BORDER );
	m_splitter7->SetSashGravity( 0.5 );
	m_splitter7->SetSashSize( 100 );
	m_splitter7->Connect( wxEVT_IDLE, wxIdleEventHandler( Refine2DResultsPanelParent::m_splitter7OnIdle ), NULL, this );
	
	LeftPanel = new wxPanel( m_splitter7, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer203;
	bSizer203 = new wxBoxSizer( wxVERTICAL );
	
	wxFlexGridSizer* fgSizer11;
	fgSizer11 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer11->AddGrowableCol( 0 );
	fgSizer11->SetFlexibleDirection( wxBOTH );
	fgSizer11->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	wxFlexGridSizer* fgSizer16;
	fgSizer16 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer16->AddGrowableCol( 1 );
	fgSizer16->SetFlexibleDirection( wxBOTH );
	fgSizer16->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticText284 = new wxStaticText( LeftPanel, wxID_ANY, wxT("Select Refinement Package :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText284->Wrap( -1 );
	fgSizer16->Add( m_staticText284, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	RefinementPackageComboBox = new MemoryComboBox( LeftPanel, wxID_ANY, wxT("Combo!"), wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	RefinementPackageComboBox->SetMinSize( wxSize( 350,-1 ) );
	
	fgSizer16->Add( RefinementPackageComboBox, 0, wxALL, 5 );
	
	m_staticText285 = new wxStaticText( LeftPanel, wxID_ANY, wxT("Select Classification :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText285->Wrap( -1 );
	fgSizer16->Add( m_staticText285, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	InputParametersComboBox = new MemoryComboBox( LeftPanel, wxID_ANY, wxT("Combo!"), wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	InputParametersComboBox->SetMinSize( wxSize( 350,-1 ) );
	
	fgSizer16->Add( InputParametersComboBox, 1, wxALL, 5 );
	
	
	fgSizer11->Add( fgSizer16, 0, wxEXPAND, 5 );
	
	
	bSizer203->Add( fgSizer11, 0, wxALIGN_CENTER_VERTICAL|wxEXPAND, 5 );
	
	m_staticline66 = new wxStaticLine( LeftPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer203->Add( m_staticline66, 0, wxEXPAND | wxALL, 5 );
	
	JobDetailsPanel = new wxPanel( LeftPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	JobDetailsPanel->Hide();
	
	wxBoxSizer* bSizer270;
	bSizer270 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer101;
	bSizer101 = new wxBoxSizer( wxVERTICAL );
	
	InfoSizer = new wxFlexGridSizer( 0, 6, 0, 0 );
	InfoSizer->SetFlexibleDirection( wxBOTH );
	InfoSizer->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticText72 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Classification ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText72->Wrap( -1 );
	m_staticText72->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText72, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	ClassificationIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ClassificationIDStaticText->Wrap( -1 );
	InfoSizer->Add( ClassificationIDStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText74 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Date of Run :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText74->Wrap( -1 );
	m_staticText74->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText74, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	DateOfRunStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	DateOfRunStaticText->Wrap( -1 );
	InfoSizer->Add( DateOfRunStaticText, 0, wxALL, 5 );
	
	m_staticText93 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Time Of Run :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText93->Wrap( -1 );
	m_staticText93->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText93, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	TimeOfRunStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	TimeOfRunStaticText->Wrap( -1 );
	InfoSizer->Add( TimeOfRunStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText83 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Refinement Package ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText83->Wrap( -1 );
	m_staticText83->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText83, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	RefinementPackageIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	RefinementPackageIDStaticText->Wrap( -1 );
	InfoSizer->Add( RefinementPackageIDStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText82 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Ref. Classification ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText82->Wrap( -1 );
	m_staticText82->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText82, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	StartClassificationIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	StartClassificationIDStaticText->Wrap( -1 );
	InfoSizer->Add( StartClassificationIDStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText78 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("No. Classes :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText78->Wrap( -1 );
	m_staticText78->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText78, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	NumberClassesStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	NumberClassesStaticText->Wrap( -1 );
	InfoSizer->Add( NumberClassesStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText96 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("No. Input Particles :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText96->Wrap( -1 );
	m_staticText96->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText96, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	NumberParticlesStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	NumberParticlesStaticText->Wrap( -1 );
	InfoSizer->Add( NumberParticlesStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText85 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Low Res. Limit :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText85->Wrap( -1 );
	m_staticText85->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText85, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	LowResLimitStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	LowResLimitStaticText->Wrap( -1 );
	InfoSizer->Add( LowResLimitStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText87 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("High Res. Limit :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText87->Wrap( -1 );
	m_staticText87->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText87, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	HighResLimitStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	HighResLimitStaticText->Wrap( -1 );
	InfoSizer->Add( HighResLimitStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText89 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Mask Radius :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText89->Wrap( -1 );
	m_staticText89->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText89, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MaskRadiusStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaskRadiusStaticText->Wrap( -1 );
	InfoSizer->Add( MaskRadiusStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText91 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Angular Search Step :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText91->Wrap( -1 );
	m_staticText91->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText91, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	AngularSearchStepStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AngularSearchStepStaticText->Wrap( -1 );
	InfoSizer->Add( AngularSearchStepStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText79 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Search Range X :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText79->Wrap( -1 );
	m_staticText79->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText79, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	SearchRangeXStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	SearchRangeXStaticText->Wrap( -1 );
	InfoSizer->Add( SearchRangeXStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText95 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Smoothing Factor :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText95->Wrap( -1 );
	m_staticText95->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText95, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	SmoothingFactorStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	SmoothingFactorStaticText->Wrap( -1 );
	InfoSizer->Add( SmoothingFactorStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	LargeAstigExpectedLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Exclude Blank Edges? :"), wxDefaultPosition, wxDefaultSize, 0 );
	LargeAstigExpectedLabel->Wrap( -1 );
	LargeAstigExpectedLabel->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( LargeAstigExpectedLabel, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	ExcludeBlankEdgesStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ExcludeBlankEdgesStaticText->Wrap( -1 );
	InfoSizer->Add( ExcludeBlankEdgesStaticText, 0, wxALL, 5 );
	
	m_staticText99 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Search Range Y :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText99->Wrap( -1 );
	m_staticText99->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText99, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	SearchRangeYStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	SearchRangeYStaticText->Wrap( -1 );
	InfoSizer->Add( SearchRangeYStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	ToleratedAstigLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Auto Percent Used? :"), wxDefaultPosition, wxDefaultSize, 0 );
	ToleratedAstigLabel->Wrap( -1 );
	ToleratedAstigLabel->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( ToleratedAstigLabel, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	AutoPercentUsedStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AutoPercentUsedStaticText->Wrap( -1 );
	InfoSizer->Add( AutoPercentUsedStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	NumberOfAveragedFramesLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Percent Used :"), wxDefaultPosition, wxDefaultSize, 0 );
	NumberOfAveragedFramesLabel->Wrap( -1 );
	NumberOfAveragedFramesLabel->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( NumberOfAveragedFramesLabel, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	PercentUsedStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	PercentUsedStaticText->Wrap( -1 );
	InfoSizer->Add( PercentUsedStaticText, 0, wxALL, 5 );
	
	
	bSizer101->Add( InfoSizer, 1, wxEXPAND, 5 );
	
	m_staticline30 = new wxStaticLine( JobDetailsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer101->Add( m_staticline30, 0, wxEXPAND | wxALL, 5 );
	
	
	bSizer270->Add( bSizer101, 1, wxEXPAND, 5 );
	
	
	JobDetailsPanel->SetSizer( bSizer270 );
	JobDetailsPanel->Layout();
	bSizer270->Fit( JobDetailsPanel );
	bSizer203->Add( JobDetailsPanel, 0, wxEXPAND | wxALL, 5 );
	
	ClassumDisplayPanel = new DisplayPanel( LeftPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer203->Add( ClassumDisplayPanel, 1, wxEXPAND | wxALL, 5 );
	
	
	LeftPanel->SetSizer( bSizer203 );
	LeftPanel->Layout();
	bSizer203->Fit( LeftPanel );
	m_panel49 = new wxPanel( m_splitter7, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer206;
	bSizer206 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer269;
	bSizer269 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText321 = new wxStaticText( m_panel49, wxID_ANY, wxT("Manage class average selections"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText321->Wrap( -1 );
	m_staticText321->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	bSizer269->Add( m_staticText321, 0, wxALIGN_BOTTOM|wxALL, 5 );
	
	
	bSizer269->Add( 0, 0, 1, wxEXPAND, 5 );
	
	JobDetailsToggleButton = new wxToggleButton( m_panel49, wxID_ANY, wxT("Show Job Details"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer269->Add( JobDetailsToggleButton, 0, wxALL, 5 );
	
	
	bSizer206->Add( bSizer269, 1, wxEXPAND, 5 );
	
	m_staticline671 = new wxStaticLine( m_panel49, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer206->Add( m_staticline671, 0, wxEXPAND | wxALL, 5 );
	
	SelectionPanel = new wxPanel( m_panel49, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer271;
	bSizer271 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer267;
	bSizer267 = new wxBoxSizer( wxVERTICAL );
	
	SelectionManagerListCtrl = new ClassificationSelectionListCtrl( SelectionPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_REPORT|wxLC_SINGLE_SEL|wxLC_VIRTUAL );
	bSizer267->Add( SelectionManagerListCtrl, 0, wxALL|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer268;
	bSizer268 = new wxBoxSizer( wxHORIZONTAL );
	
	AddButton = new wxButton( SelectionPanel, wxID_ANY, wxT("Add"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer268->Add( AddButton, 0, wxALL, 5 );
	
	DeleteButton = new wxButton( SelectionPanel, wxID_ANY, wxT("Delete"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer268->Add( DeleteButton, 0, wxALL, 5 );
	
	RenameButton = new wxButton( SelectionPanel, wxID_ANY, wxT("Rename"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer268->Add( RenameButton, 0, wxALL, 5 );
	
	CopyOtherButton = new wxButton( SelectionPanel, wxID_ANY, wxT("Copy Other"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer268->Add( CopyOtherButton, 0, wxALL, 5 );
	
	m_staticline64 = new wxStaticLine( SelectionPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_VERTICAL );
	bSizer268->Add( m_staticline64, 0, wxEXPAND | wxALL, 5 );
	
	ClearButton = new wxButton( SelectionPanel, wxID_ANY, wxT("Clear"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer268->Add( ClearButton, 0, wxALL, 5 );
	
	InvertButton = new wxButton( SelectionPanel, wxID_ANY, wxT("Invert"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer268->Add( InvertButton, 0, wxALL, 5 );
	
	
	bSizer267->Add( bSizer268, 0, wxEXPAND, 5 );
	
	
	bSizer271->Add( bSizer267, 1, wxEXPAND, 5 );
	
	
	SelectionPanel->SetSizer( bSizer271 );
	SelectionPanel->Layout();
	bSizer271->Fit( SelectionPanel );
	bSizer206->Add( SelectionPanel, 0, wxEXPAND | wxALL, 5 );
	
	ClassNumberStaticText = new wxStaticText( m_panel49, wxID_ANY, wxT("Images for class #1"), wxDefaultPosition, wxDefaultSize, 0 );
	ClassNumberStaticText->Wrap( -1 );
	ClassNumberStaticText->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	bSizer206->Add( ClassNumberStaticText, 0, wxALIGN_LEFT|wxALL|wxBOTTOM, 5 );
	
	m_staticline65 = new wxStaticLine( m_panel49, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer206->Add( m_staticline65, 0, wxEXPAND | wxALL, 5 );
	
	ParticleDisplayPanel = new DisplayPanel( m_panel49, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer206->Add( ParticleDisplayPanel, 90, wxEXPAND | wxALL, 5 );
	
	
	m_panel49->SetSizer( bSizer206 );
	m_panel49->Layout();
	bSizer206->Fit( m_panel49 );
	m_splitter7->SplitVertically( LeftPanel, m_panel49, 800 );
	bSizer202->Add( m_splitter7, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer202 );
	this->Layout();
	
	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( Refine2DResultsPanelParent::OnUpdateUI ) );
	RefinementPackageComboBox->Connect( wxEVT_COMMAND_COMBOBOX_SELECTED, wxCommandEventHandler( Refine2DResultsPanelParent::OnRefinementPackageComboBox ), NULL, this );
	InputParametersComboBox->Connect( wxEVT_COMMAND_COMBOBOX_SELECTED, wxCommandEventHandler( Refine2DResultsPanelParent::OnInputParametersComboBox ), NULL, this );
	JobDetailsToggleButton->Connect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( Refine2DResultsPanelParent::OnJobDetailsToggle ), NULL, this );
	SelectionManagerListCtrl->Connect( wxEVT_COMMAND_LIST_END_LABEL_EDIT, wxListEventHandler( Refine2DResultsPanelParent::OnEndLabelEdit ), NULL, this );
	SelectionManagerListCtrl->Connect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( Refine2DResultsPanelParent::OnActivated ), NULL, this );
	SelectionManagerListCtrl->Connect( wxEVT_COMMAND_LIST_ITEM_DESELECTED, wxListEventHandler( Refine2DResultsPanelParent::OnDeselected ), NULL, this );
	SelectionManagerListCtrl->Connect( wxEVT_COMMAND_LIST_ITEM_SELECTED, wxListEventHandler( Refine2DResultsPanelParent::OnSelected ), NULL, this );
	AddButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DResultsPanelParent::OnAddButtonClick ), NULL, this );
	DeleteButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DResultsPanelParent::OnDeleteButtonClick ), NULL, this );
	RenameButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DResultsPanelParent::OnRenameButtonClick ), NULL, this );
	CopyOtherButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DResultsPanelParent::OnCopyOtherButtonClick ), NULL, this );
	ClearButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DResultsPanelParent::OnClearButtonClick ), NULL, this );
	InvertButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DResultsPanelParent::OnInvertButtonClick ), NULL, this );
}

Refine2DResultsPanelParent::~Refine2DResultsPanelParent()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( Refine2DResultsPanelParent::OnUpdateUI ) );
	RefinementPackageComboBox->Disconnect( wxEVT_COMMAND_COMBOBOX_SELECTED, wxCommandEventHandler( Refine2DResultsPanelParent::OnRefinementPackageComboBox ), NULL, this );
	InputParametersComboBox->Disconnect( wxEVT_COMMAND_COMBOBOX_SELECTED, wxCommandEventHandler( Refine2DResultsPanelParent::OnInputParametersComboBox ), NULL, this );
	JobDetailsToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( Refine2DResultsPanelParent::OnJobDetailsToggle ), NULL, this );
	SelectionManagerListCtrl->Disconnect( wxEVT_COMMAND_LIST_END_LABEL_EDIT, wxListEventHandler( Refine2DResultsPanelParent::OnEndLabelEdit ), NULL, this );
	SelectionManagerListCtrl->Disconnect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( Refine2DResultsPanelParent::OnActivated ), NULL, this );
	SelectionManagerListCtrl->Disconnect( wxEVT_COMMAND_LIST_ITEM_DESELECTED, wxListEventHandler( Refine2DResultsPanelParent::OnDeselected ), NULL, this );
	SelectionManagerListCtrl->Disconnect( wxEVT_COMMAND_LIST_ITEM_SELECTED, wxListEventHandler( Refine2DResultsPanelParent::OnSelected ), NULL, this );
	AddButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DResultsPanelParent::OnAddButtonClick ), NULL, this );
	DeleteButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DResultsPanelParent::OnDeleteButtonClick ), NULL, this );
	RenameButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DResultsPanelParent::OnRenameButtonClick ), NULL, this );
	CopyOtherButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DResultsPanelParent::OnCopyOtherButtonClick ), NULL, this );
	ClearButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DResultsPanelParent::OnClearButtonClick ), NULL, this );
	InvertButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine2DResultsPanelParent::OnInvertButtonClick ), NULL, this );
	
}

ShowCTFResultsParentPanel::ShowCTFResultsParentPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer92;
	bSizer92 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer93;
	bSizer93 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer94;
	bSizer94 = new wxBoxSizer( wxHORIZONTAL );
	
	FitType2DRadioButton = new wxRadioButton( this, wxID_ANY, wxT("Show 2D Fit Output"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer94->Add( FitType2DRadioButton, 0, wxALL, 5 );
	
	FitType1DRadioButton = new wxRadioButton( this, wxID_ANY, wxT("Show 1D Fit Output"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer94->Add( FitType1DRadioButton, 0, wxALL, 5 );
	
	
	bSizer93->Add( bSizer94, 1, wxEXPAND, 5 );
	
	m_staticline26 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer93->Add( m_staticline26, 0, wxEXPAND | wxALL, 5 );
	
	
	bSizer92->Add( bSizer93, 1, wxEXPAND, 5 );
	
	CTF2DResultsPanel = new BitmapPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer92->Add( CTF2DResultsPanel, 50, wxEXPAND | wxALL, 5 );
	
	CTFPlotPanel = new CTF1DPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	CTFPlotPanel->Hide();
	
	bSizer92->Add( CTFPlotPanel, 50, wxEXPAND | wxALL, 5 );
	
	
	this->SetSizer( bSizer92 );
	this->Layout();
	
	// Connect Events
	FitType2DRadioButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( ShowCTFResultsParentPanel::OnFitTypeRadioButton ), NULL, this );
	FitType1DRadioButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( ShowCTFResultsParentPanel::OnFitTypeRadioButton ), NULL, this );
}

ShowCTFResultsParentPanel::~ShowCTFResultsParentPanel()
{
	// Disconnect Events
	FitType2DRadioButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( ShowCTFResultsParentPanel::OnFitTypeRadioButton ), NULL, this );
	FitType1DRadioButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( ShowCTFResultsParentPanel::OnFitTypeRadioButton ), NULL, this );
	
}

PickingResultsDisplayParentPanel::PickingResultsDisplayParentPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer92;
	bSizer92 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer93;
	bSizer93 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer94;
	bSizer94 = new wxBoxSizer( wxHORIZONTAL );
	
	CirclesAroundParticlesCheckBox = new wxCheckBox( this, wxID_ANY, wxT("Circles around particles"), wxDefaultPosition, wxDefaultSize, 0 );
	CirclesAroundParticlesCheckBox->SetValue(true); 
	CirclesAroundParticlesCheckBox->SetToolTip( wxT("Draw circles around picked particles") );
	
	bSizer94->Add( CirclesAroundParticlesCheckBox, 0, wxALL, 5 );
	
	HighPassFilterCheckBox = new wxCheckBox( this, wxID_ANY, wxT("High-pass filter"), wxDefaultPosition, wxDefaultSize, 0 );
	HighPassFilterCheckBox->SetValue(true); 
	HighPassFilterCheckBox->SetToolTip( wxT("Filter the image to remove density ramps") );
	
	bSizer94->Add( HighPassFilterCheckBox, 0, wxALL, 5 );
	
	LowPassFilterCheckBox = new wxCheckBox( this, wxID_ANY, wxT("Low-pass filter"), wxDefaultPosition, wxDefaultSize, 0 );
	LowPassFilterCheckBox->SetToolTip( wxT("Filter the image to remove density ramps") );
	
	bSizer94->Add( LowPassFilterCheckBox, 0, wxALL, 5 );
	
	ScaleBarCheckBox = new wxCheckBox( this, wxID_ANY, wxT("Scale bar"), wxDefaultPosition, wxDefaultSize, 0 );
	ScaleBarCheckBox->SetValue(true); 
	ScaleBarCheckBox->SetToolTip( wxT("Display a scale bar") );
	
	bSizer94->Add( ScaleBarCheckBox, 0, wxALL, 5 );
	
	UndoButton = new wxButton( this, wxID_ANY, wxT("Undo"), wxDefaultPosition, wxDefaultSize, 0 );
	UndoButton->Enable( false );
	UndoButton->SetToolTip( wxT("Undo manual change of particle coordinates") );
	
	bSizer94->Add( UndoButton, 0, wxALL, 5 );
	
	RedoButton = new wxButton( this, wxID_ANY, wxT("Redo"), wxDefaultPosition, wxDefaultSize, 0 );
	RedoButton->Enable( false );
	RedoButton->SetToolTip( wxT("Redo manual change of particle coordinates") );
	
	bSizer94->Add( RedoButton, 0, wxALL, 5 );
	
	
	bSizer93->Add( bSizer94, 1, wxEXPAND, 5 );
	
	m_staticline26 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer93->Add( m_staticline26, 0, wxEXPAND | wxALL, 5 );
	
	
	bSizer92->Add( bSizer93, 1, wxEXPAND, 5 );
	
	PickingResultsImagePanel = new PickingBitmapPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer92->Add( PickingResultsImagePanel, 50, wxEXPAND | wxALL, 5 );
	
	
	this->SetSizer( bSizer92 );
	this->Layout();
	
	// Connect Events
	CirclesAroundParticlesCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( PickingResultsDisplayParentPanel::OnCirclesAroundParticlesCheckBox ), NULL, this );
	HighPassFilterCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( PickingResultsDisplayParentPanel::OnHighPassFilterCheckBox ), NULL, this );
	LowPassFilterCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( PickingResultsDisplayParentPanel::OnLowPassFilterCheckBox ), NULL, this );
	ScaleBarCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( PickingResultsDisplayParentPanel::OnScaleBarCheckBox ), NULL, this );
	UndoButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsDisplayParentPanel::OnUndoButtonClick ), NULL, this );
	RedoButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsDisplayParentPanel::OnRedoButtonClick ), NULL, this );
}

PickingResultsDisplayParentPanel::~PickingResultsDisplayParentPanel()
{
	// Disconnect Events
	CirclesAroundParticlesCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( PickingResultsDisplayParentPanel::OnCirclesAroundParticlesCheckBox ), NULL, this );
	HighPassFilterCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( PickingResultsDisplayParentPanel::OnHighPassFilterCheckBox ), NULL, this );
	LowPassFilterCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( PickingResultsDisplayParentPanel::OnLowPassFilterCheckBox ), NULL, this );
	ScaleBarCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( PickingResultsDisplayParentPanel::OnScaleBarCheckBox ), NULL, this );
	UndoButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsDisplayParentPanel::OnUndoButtonClick ), NULL, this );
	RedoButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsDisplayParentPanel::OnRedoButtonClick ), NULL, this );
	
}

FindCTFResultsPanel::FindCTFResultsPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer63;
	bSizer63 = new wxBoxSizer( wxVERTICAL );
	
	m_staticline25 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer63->Add( m_staticline25, 0, wxEXPAND | wxALL, 5 );
	
	m_splitter4 = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitter4->Connect( wxEVT_IDLE, wxIdleEventHandler( FindCTFResultsPanel::m_splitter4OnIdle ), NULL, this );
	
	m_panel13 = new wxPanel( m_splitter4, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer66;
	bSizer66 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer64;
	bSizer64 = new wxBoxSizer( wxHORIZONTAL );
	
	AllImagesButton = new wxRadioButton( m_panel13, wxID_ANY, wxT("All Images"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer64->Add( AllImagesButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	ByFilterButton = new wxRadioButton( m_panel13, wxID_ANY, wxT("By Filter"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer64->Add( ByFilterButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	FilterButton = new wxButton( m_panel13, wxID_ANY, wxT("Define Filter"), wxDefaultPosition, wxDefaultSize, 0 );
	FilterButton->Enable( false );
	
	bSizer64->Add( FilterButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer66->Add( bSizer64, 0, wxEXPAND, 5 );
	
	ResultDataView = new ResultsDataViewListCtrl( m_panel13, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxDV_VERT_RULES );
	bSizer66->Add( ResultDataView, 1, wxALL|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer68;
	bSizer68 = new wxBoxSizer( wxHORIZONTAL );
	
	PreviousButton = new wxButton( m_panel13, wxID_ANY, wxT("Previous"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer68->Add( PreviousButton, 0, wxALL, 5 );
	
	
	bSizer68->Add( 0, 0, 1, 0, 5 );
	
	NextButton = new wxButton( m_panel13, wxID_ANY, wxT("Next"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer68->Add( NextButton, 0, wxALL, 5 );
	
	
	bSizer66->Add( bSizer68, 0, wxEXPAND, 5 );
	
	
	m_panel13->SetSizer( bSizer66 );
	m_panel13->Layout();
	bSizer66->Fit( m_panel13 );
	RightPanel = new wxPanel( m_splitter4, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer681;
	bSizer681 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer73;
	bSizer73 = new wxBoxSizer( wxVERTICAL );
	
	JobDetailsToggleButton = new wxToggleButton( RightPanel, wxID_ANY, wxT("Show Job Details"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer73->Add( JobDetailsToggleButton, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	m_staticline28 = new wxStaticLine( RightPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer73->Add( m_staticline28, 0, wxEXPAND | wxALL, 5 );
	
	JobDetailsPanel = new wxPanel( RightPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	JobDetailsPanel->Hide();
	
	wxBoxSizer* bSizer101;
	bSizer101 = new wxBoxSizer( wxVERTICAL );
	
	InfoSizer = new wxFlexGridSizer( 0, 6, 0, 0 );
	InfoSizer->SetFlexibleDirection( wxBOTH );
	InfoSizer->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticText72 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Estimation ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText72->Wrap( -1 );
	m_staticText72->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText72, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	EstimationIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	EstimationIDStaticText->Wrap( -1 );
	InfoSizer->Add( EstimationIDStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText74 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Date of Run :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText74->Wrap( -1 );
	m_staticText74->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText74, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	DateOfRunStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	DateOfRunStaticText->Wrap( -1 );
	InfoSizer->Add( DateOfRunStaticText, 0, wxALL, 5 );
	
	m_staticText93 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Time Of Run :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText93->Wrap( -1 );
	m_staticText93->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText93, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	TimeOfRunStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	TimeOfRunStaticText->Wrap( -1 );
	InfoSizer->Add( TimeOfRunStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText83 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Voltage :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText83->Wrap( -1 );
	m_staticText83->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText83, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	VoltageStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	VoltageStaticText->Wrap( -1 );
	InfoSizer->Add( VoltageStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText82 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Cs :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText82->Wrap( -1 );
	m_staticText82->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText82, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	CsStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	CsStaticText->Wrap( -1 );
	InfoSizer->Add( CsStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText78 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Pixel Size :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText78->Wrap( -1 );
	m_staticText78->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText78, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	PixelSizeStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	PixelSizeStaticText->Wrap( -1 );
	InfoSizer->Add( PixelSizeStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText96 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Amp. Contrast :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText96->Wrap( -1 );
	m_staticText96->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText96, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	AmplitudeContrastStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AmplitudeContrastStaticText->Wrap( -1 );
	InfoSizer->Add( AmplitudeContrastStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText85 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Box Size :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText85->Wrap( -1 );
	m_staticText85->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText85, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	BoxSizeStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	BoxSizeStaticText->Wrap( -1 );
	InfoSizer->Add( BoxSizeStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText87 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Min. Res. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText87->Wrap( -1 );
	m_staticText87->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText87, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MinResStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MinResStaticText->Wrap( -1 );
	InfoSizer->Add( MinResStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText89 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Max. Res. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText89->Wrap( -1 );
	m_staticText89->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText89, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MaxResStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaxResStaticText->Wrap( -1 );
	InfoSizer->Add( MaxResStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText91 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Min. Defocus :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText91->Wrap( -1 );
	m_staticText91->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText91, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MinDefocusStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MinDefocusStaticText->Wrap( -1 );
	InfoSizer->Add( MinDefocusStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText79 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Max. Defocus :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText79->Wrap( -1 );
	m_staticText79->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText79, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MaxDefocusStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaxDefocusStaticText->Wrap( -1 );
	InfoSizer->Add( MaxDefocusStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText95 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Defocus Step :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText95->Wrap( -1 );
	m_staticText95->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText95, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	DefocusStepStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	DefocusStepStaticText->Wrap( -1 );
	InfoSizer->Add( DefocusStepStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	LargeAstigExpectedLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("V. Large Astig. Expect. :"), wxDefaultPosition, wxDefaultSize, 0 );
	LargeAstigExpectedLabel->Wrap( -1 );
	LargeAstigExpectedLabel->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( LargeAstigExpectedLabel, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	LargeAstigExpectedStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	LargeAstigExpectedStaticText->Wrap( -1 );
	InfoSizer->Add( LargeAstigExpectedStaticText, 0, wxALL, 5 );
	
	m_staticText99 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Restrain Astig. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText99->Wrap( -1 );
	m_staticText99->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText99, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	RestrainAstigStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	RestrainAstigStaticText->Wrap( -1 );
	InfoSizer->Add( RestrainAstigStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	ToleratedAstigLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Tolerated Astig. :"), wxDefaultPosition, wxDefaultSize, 0 );
	ToleratedAstigLabel->Wrap( -1 );
	ToleratedAstigLabel->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( ToleratedAstigLabel, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	ToleratedAstigStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ToleratedAstigStaticText->Wrap( -1 );
	InfoSizer->Add( ToleratedAstigStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	NumberOfAveragedFramesLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Num. Averaged Frames :"), wxDefaultPosition, wxDefaultSize, 0 );
	NumberOfAveragedFramesLabel->Wrap( -1 );
	NumberOfAveragedFramesLabel->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( NumberOfAveragedFramesLabel, 0, wxALL, 5 );
	
	NumberOfAveragedFramesStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	NumberOfAveragedFramesStaticText->Wrap( -1 );
	InfoSizer->Add( NumberOfAveragedFramesStaticText, 0, wxALL, 5 );
	
	m_staticText103 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Add. Phase Shift :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText103->Wrap( -1 );
	m_staticText103->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText103, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	AddtionalPhaseShiftStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AddtionalPhaseShiftStaticText->Wrap( -1 );
	InfoSizer->Add( AddtionalPhaseShiftStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	MinPhaseShiftLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Min. Phase Shift :"), wxDefaultPosition, wxDefaultSize, 0 );
	MinPhaseShiftLabel->Wrap( -1 );
	MinPhaseShiftLabel->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( MinPhaseShiftLabel, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MinPhaseShiftStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MinPhaseShiftStaticText->Wrap( -1 );
	InfoSizer->Add( MinPhaseShiftStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	MaxPhaseShiftLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Max. Phase Shift :"), wxDefaultPosition, wxDefaultSize, 0 );
	MaxPhaseShiftLabel->Wrap( -1 );
	MaxPhaseShiftLabel->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( MaxPhaseShiftLabel, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MaxPhaseshiftStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaxPhaseshiftStaticText->Wrap( -1 );
	InfoSizer->Add( MaxPhaseshiftStaticText, 0, wxALL, 5 );
	
	PhaseShiftStepLabel = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Phase Shift Step :"), wxDefaultPosition, wxDefaultSize, 0 );
	PhaseShiftStepLabel->Wrap( -1 );
	PhaseShiftStepLabel->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( PhaseShiftStepLabel, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	PhaseShiftStepStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	PhaseShiftStepStaticText->Wrap( -1 );
	InfoSizer->Add( PhaseShiftStepStaticText, 0, wxALL, 5 );
	
	
	bSizer101->Add( InfoSizer, 1, wxEXPAND, 5 );
	
	m_staticline30 = new wxStaticLine( JobDetailsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer101->Add( m_staticline30, 0, wxEXPAND | wxALL, 5 );
	
	
	JobDetailsPanel->SetSizer( bSizer101 );
	JobDetailsPanel->Layout();
	bSizer101->Fit( JobDetailsPanel );
	bSizer73->Add( JobDetailsPanel, 1, wxALL|wxEXPAND, 5 );
	
	
	bSizer681->Add( bSizer73, 0, wxEXPAND, 5 );
	
	wxGridSizer* gSizer5;
	gSizer5 = new wxGridSizer( 0, 6, 0, 0 );
	
	
	bSizer681->Add( gSizer5, 0, wxEXPAND, 5 );
	
	ResultPanel = new ShowCTFResultsPanel( RightPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer681->Add( ResultPanel, 1, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer69;
	bSizer69 = new wxBoxSizer( wxHORIZONTAL );
	
	AddToGroupButton = new wxButton( RightPanel, wxID_ANY, wxT("Add Image To Group"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer69->Add( AddToGroupButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	GroupComboBox = new MemoryComboBox( RightPanel, wxID_ANY, wxT("Combo!"), wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	GroupComboBox->SetMinSize( wxSize( 200,-1 ) );
	
	bSizer69->Add( GroupComboBox, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer681->Add( bSizer69, 0, wxALIGN_RIGHT, 5 );
	
	
	RightPanel->SetSizer( bSizer681 );
	RightPanel->Layout();
	bSizer681->Fit( RightPanel );
	m_splitter4->SplitVertically( m_panel13, RightPanel, 500 );
	bSizer63->Add( m_splitter4, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer63 );
	this->Layout();
	
	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( FindCTFResultsPanel::OnUpdateUI ) );
	AllImagesButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( FindCTFResultsPanel::OnAllMoviesSelect ), NULL, this );
	ByFilterButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( FindCTFResultsPanel::OnByFilterSelect ), NULL, this );
	FilterButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFResultsPanel::OnDefineFilterClick ), NULL, this );
	PreviousButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFResultsPanel::OnPreviousButtonClick ), NULL, this );
	NextButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFResultsPanel::OnNextButtonClick ), NULL, this );
	JobDetailsToggleButton->Connect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( FindCTFResultsPanel::OnJobDetailsToggle ), NULL, this );
	AddToGroupButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFResultsPanel::OnAddToGroupClick ), NULL, this );
}

FindCTFResultsPanel::~FindCTFResultsPanel()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( FindCTFResultsPanel::OnUpdateUI ) );
	AllImagesButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( FindCTFResultsPanel::OnAllMoviesSelect ), NULL, this );
	ByFilterButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( FindCTFResultsPanel::OnByFilterSelect ), NULL, this );
	FilterButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFResultsPanel::OnDefineFilterClick ), NULL, this );
	PreviousButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFResultsPanel::OnPreviousButtonClick ), NULL, this );
	NextButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFResultsPanel::OnNextButtonClick ), NULL, this );
	JobDetailsToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( FindCTFResultsPanel::OnJobDetailsToggle ), NULL, this );
	AddToGroupButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFResultsPanel::OnAddToGroupClick ), NULL, this );
	
}

PickingResultsPanel::PickingResultsPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer63;
	bSizer63 = new wxBoxSizer( wxVERTICAL );
	
	m_staticline25 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer63->Add( m_staticline25, 0, wxEXPAND | wxALL, 5 );
	
	m_splitter4 = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitter4->Connect( wxEVT_IDLE, wxIdleEventHandler( PickingResultsPanel::m_splitter4OnIdle ), NULL, this );
	
	m_panel13 = new wxPanel( m_splitter4, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer66;
	bSizer66 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer64;
	bSizer64 = new wxBoxSizer( wxHORIZONTAL );
	
	AllImagesButton = new wxRadioButton( m_panel13, wxID_ANY, wxT("All Images"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer64->Add( AllImagesButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	ByFilterButton = new wxRadioButton( m_panel13, wxID_ANY, wxT("By Filter"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer64->Add( ByFilterButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	FilterButton = new wxButton( m_panel13, wxID_ANY, wxT("Define Filter"), wxDefaultPosition, wxDefaultSize, 0 );
	FilterButton->Enable( false );
	
	bSizer64->Add( FilterButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer66->Add( bSizer64, 0, wxEXPAND, 5 );
	
	ResultDataView = new ResultsDataViewListCtrl( m_panel13, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxDV_VERT_RULES );
	bSizer66->Add( ResultDataView, 1, wxALL|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer68;
	bSizer68 = new wxBoxSizer( wxHORIZONTAL );
	
	PreviousButton = new wxButton( m_panel13, wxID_ANY, wxT("Previous"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer68->Add( PreviousButton, 0, wxALL, 5 );
	
	
	bSizer68->Add( 0, 0, 1, 0, 5 );
	
	NextButton = new wxButton( m_panel13, wxID_ANY, wxT("Next"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer68->Add( NextButton, 0, wxALL, 5 );
	
	
	bSizer66->Add( bSizer68, 0, wxEXPAND, 5 );
	
	
	m_panel13->SetSizer( bSizer66 );
	m_panel13->Layout();
	bSizer66->Fit( m_panel13 );
	RightPanel = new wxPanel( m_splitter4, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer681;
	bSizer681 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer73;
	bSizer73 = new wxBoxSizer( wxVERTICAL );
	
	JobDetailsToggleButton = new wxToggleButton( RightPanel, wxID_ANY, wxT("Show Job Details"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer73->Add( JobDetailsToggleButton, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	m_staticline28 = new wxStaticLine( RightPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer73->Add( m_staticline28, 0, wxEXPAND | wxALL, 5 );
	
	JobDetailsPanel = new wxPanel( RightPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	JobDetailsPanel->Hide();
	
	wxBoxSizer* bSizer101;
	bSizer101 = new wxBoxSizer( wxVERTICAL );
	
	InfoSizer = new wxFlexGridSizer( 0, 6, 0, 0 );
	InfoSizer->SetFlexibleDirection( wxBOTH );
	InfoSizer->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticText72 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Pick ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText72->Wrap( -1 );
	m_staticText72->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText72, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	PickIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	PickIDStaticText->Wrap( -1 );
	InfoSizer->Add( PickIDStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText74 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Date of Run :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText74->Wrap( -1 );
	m_staticText74->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText74, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	DateOfRunStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	DateOfRunStaticText->Wrap( -1 );
	InfoSizer->Add( DateOfRunStaticText, 0, wxALL, 5 );
	
	m_staticText93 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Time Of Run :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText93->Wrap( -1 );
	m_staticText93->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText93, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	TimeOfRunStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	TimeOfRunStaticText->Wrap( -1 );
	InfoSizer->Add( TimeOfRunStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText83 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Algorithm :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText83->Wrap( -1 );
	m_staticText83->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText83, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	AlgorithmStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AlgorithmStaticText->Wrap( -1 );
	InfoSizer->Add( AlgorithmStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText82 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Manual edit :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText82->Wrap( -1 );
	m_staticText82->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText82, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	ManualEditStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ManualEditStaticText->Wrap( -1 );
	InfoSizer->Add( ManualEditStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText78 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Threshold :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText78->Wrap( -1 );
	m_staticText78->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText78, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	ThresholdStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ThresholdStaticText->Wrap( -1 );
	InfoSizer->Add( ThresholdStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText96 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Max. Radius :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText96->Wrap( -1 );
	m_staticText96->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText96, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MaximumRadiusStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaximumRadiusStaticText->Wrap( -1 );
	InfoSizer->Add( MaximumRadiusStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText85 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Charact. Radius :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText85->Wrap( -1 );
	m_staticText85->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText85, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	CharacteristicRadiusStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	CharacteristicRadiusStaticText->Wrap( -1 );
	InfoSizer->Add( CharacteristicRadiusStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText87 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Highest Res. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText87->Wrap( -1 );
	m_staticText87->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText87, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	HighestResStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	HighestResStaticText->Wrap( -1 );
	InfoSizer->Add( HighestResStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText89 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Min. Edge Dist. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText89->Wrap( -1 );
	m_staticText89->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText89, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MinEdgeDistStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MinEdgeDistStaticText->Wrap( -1 );
	InfoSizer->Add( MinEdgeDistStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText91 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Avoid High Var. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText91->Wrap( -1 );
	m_staticText91->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText91, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	AvoidHighVarStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AvoidHighVarStaticText->Wrap( -1 );
	InfoSizer->Add( AvoidHighVarStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText79 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Avoid Hi/Lo Mean :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText79->Wrap( -1 );
	m_staticText79->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText79, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	AvoidHighLowMeanStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AvoidHighLowMeanStaticText->Wrap( -1 );
	InfoSizer->Add( AvoidHighLowMeanStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText95 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Num. Bckgd. Boxes :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText95->Wrap( -1 );
	m_staticText95->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText95, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	NumBackgroundBoxesStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	NumBackgroundBoxesStaticText->Wrap( -1 );
	InfoSizer->Add( NumBackgroundBoxesStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	
	bSizer101->Add( InfoSizer, 1, wxEXPAND, 5 );
	
	m_staticline30 = new wxStaticLine( JobDetailsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer101->Add( m_staticline30, 0, wxEXPAND | wxALL, 5 );
	
	
	JobDetailsPanel->SetSizer( bSizer101 );
	JobDetailsPanel->Layout();
	bSizer101->Fit( JobDetailsPanel );
	bSizer73->Add( JobDetailsPanel, 1, wxALL|wxEXPAND, 5 );
	
	
	bSizer681->Add( bSizer73, 0, wxEXPAND, 5 );
	
	wxGridSizer* gSizer5;
	gSizer5 = new wxGridSizer( 0, 6, 0, 0 );
	
	
	bSizer681->Add( gSizer5, 0, wxEXPAND, 5 );
	
	ResultDisplayPanel = new PickingResultsDisplayPanel( RightPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer681->Add( ResultDisplayPanel, 1, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer69;
	bSizer69 = new wxBoxSizer( wxHORIZONTAL );
	
	AddToGroupButton = new wxButton( RightPanel, wxID_ANY, wxT("Add Image To Group"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer69->Add( AddToGroupButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	GroupComboBox = new MemoryComboBox( RightPanel, wxID_ANY, wxT("Combo!"), wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	GroupComboBox->SetMinSize( wxSize( 200,-1 ) );
	
	bSizer69->Add( GroupComboBox, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer681->Add( bSizer69, 0, wxALIGN_RIGHT, 5 );
	
	
	RightPanel->SetSizer( bSizer681 );
	RightPanel->Layout();
	bSizer681->Fit( RightPanel );
	m_splitter4->SplitVertically( m_panel13, RightPanel, 500 );
	bSizer63->Add( m_splitter4, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer63 );
	this->Layout();
	
	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( PickingResultsPanel::OnUpdateUI ) );
	AllImagesButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( PickingResultsPanel::OnAllMoviesSelect ), NULL, this );
	ByFilterButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( PickingResultsPanel::OnByFilterSelect ), NULL, this );
	FilterButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsPanel::OnDefineFilterClick ), NULL, this );
	PreviousButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsPanel::OnPreviousButtonClick ), NULL, this );
	NextButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsPanel::OnNextButtonClick ), NULL, this );
	JobDetailsToggleButton->Connect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( PickingResultsPanel::OnJobDetailsToggle ), NULL, this );
	AddToGroupButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsPanel::OnAddToGroupClick ), NULL, this );
}

PickingResultsPanel::~PickingResultsPanel()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( PickingResultsPanel::OnUpdateUI ) );
	AllImagesButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( PickingResultsPanel::OnAllMoviesSelect ), NULL, this );
	ByFilterButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( PickingResultsPanel::OnByFilterSelect ), NULL, this );
	FilterButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsPanel::OnDefineFilterClick ), NULL, this );
	PreviousButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsPanel::OnPreviousButtonClick ), NULL, this );
	NextButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsPanel::OnNextButtonClick ), NULL, this );
	JobDetailsToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( PickingResultsPanel::OnJobDetailsToggle ), NULL, this );
	AddToGroupButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( PickingResultsPanel::OnAddToGroupClick ), NULL, this );
	
}

MovieAlignResultsPanel::MovieAlignResultsPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer63;
	bSizer63 = new wxBoxSizer( wxVERTICAL );
	
	m_staticline25 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer63->Add( m_staticline25, 0, wxEXPAND | wxALL, 5 );
	
	m_splitter4 = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitter4->Connect( wxEVT_IDLE, wxIdleEventHandler( MovieAlignResultsPanel::m_splitter4OnIdle ), NULL, this );
	
	m_panel13 = new wxPanel( m_splitter4, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer66;
	bSizer66 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer64;
	bSizer64 = new wxBoxSizer( wxHORIZONTAL );
	
	AllMoviesButton = new wxRadioButton( m_panel13, wxID_ANY, wxT("All Movies"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer64->Add( AllMoviesButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	ByFilterButton = new wxRadioButton( m_panel13, wxID_ANY, wxT("By Filter"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer64->Add( ByFilterButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	FilterButton = new wxButton( m_panel13, wxID_ANY, wxT("Define Filter"), wxDefaultPosition, wxDefaultSize, 0 );
	FilterButton->Enable( false );
	
	bSizer64->Add( FilterButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer66->Add( bSizer64, 0, wxEXPAND, 5 );
	
	ResultDataView = new ResultsDataViewListCtrl( m_panel13, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxDV_VERT_RULES );
	bSizer66->Add( ResultDataView, 1, wxALL|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer68;
	bSizer68 = new wxBoxSizer( wxHORIZONTAL );
	
	PreviousButton = new wxButton( m_panel13, wxID_ANY, wxT("Previous"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer68->Add( PreviousButton, 0, wxALL, 5 );
	
	
	bSizer68->Add( 0, 0, 1, 0, 5 );
	
	NextButton = new wxButton( m_panel13, wxID_ANY, wxT("Next"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer68->Add( NextButton, 0, wxALL, 5 );
	
	
	bSizer66->Add( bSizer68, 0, wxEXPAND, 5 );
	
	
	m_panel13->SetSizer( bSizer66 );
	m_panel13->Layout();
	bSizer66->Fit( m_panel13 );
	RightPanel = new wxPanel( m_splitter4, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer681;
	bSizer681 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer73;
	bSizer73 = new wxBoxSizer( wxVERTICAL );
	
	JobDetailsToggleButton = new wxToggleButton( RightPanel, wxID_ANY, wxT("Show Job Details"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer73->Add( JobDetailsToggleButton, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	JobDetailsPanel = new wxPanel( RightPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	JobDetailsPanel->Hide();
	
	InfoSizer = new wxFlexGridSizer( 0, 6, 0, 0 );
	InfoSizer->SetFlexibleDirection( wxBOTH );
	InfoSizer->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticText72 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Alignment ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText72->Wrap( -1 );
	m_staticText72->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText72, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	AlignmentIDStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	AlignmentIDStaticText->Wrap( -1 );
	InfoSizer->Add( AlignmentIDStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText74 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Date of Run :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText74->Wrap( -1 );
	m_staticText74->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText74, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	DateOfRunStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	DateOfRunStaticText->Wrap( -1 );
	InfoSizer->Add( DateOfRunStaticText, 0, wxALL, 5 );
	
	m_staticText93 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Time Of Run :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText93->Wrap( -1 );
	m_staticText93->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText93, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	TimeOfRunStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	TimeOfRunStaticText->Wrap( -1 );
	InfoSizer->Add( TimeOfRunStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText83 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Voltage :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText83->Wrap( -1 );
	m_staticText83->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText83, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	VoltageStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	VoltageStaticText->Wrap( -1 );
	InfoSizer->Add( VoltageStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText78 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Pixel Size :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText78->Wrap( -1 );
	m_staticText78->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText78, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	PixelSizeStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	PixelSizeStaticText->Wrap( -1 );
	InfoSizer->Add( PixelSizeStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText82 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Exp. per Frame :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText82->Wrap( -1 );
	m_staticText82->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText82, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	ExposureStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ExposureStaticText->Wrap( -1 );
	InfoSizer->Add( ExposureStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText96 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Pre Exp. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText96->Wrap( -1 );
	m_staticText96->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText96, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	PreExposureStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	PreExposureStaticText->Wrap( -1 );
	InfoSizer->Add( PreExposureStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText85 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Min. Shift :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText85->Wrap( -1 );
	m_staticText85->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText85, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MinShiftStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MinShiftStaticText->Wrap( -1 );
	InfoSizer->Add( MinShiftStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText87 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Max. Shift :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText87->Wrap( -1 );
	m_staticText87->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText87, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MaxShiftStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaxShiftStaticText->Wrap( -1 );
	InfoSizer->Add( MaxShiftStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText89 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Term. Threshold :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText89->Wrap( -1 );
	m_staticText89->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText89, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	TerminationThresholdStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	TerminationThresholdStaticText->Wrap( -1 );
	InfoSizer->Add( TerminationThresholdStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText91 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Max. Iterations :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText91->Wrap( -1 );
	m_staticText91->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText91, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MaxIterationsStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaxIterationsStaticText->Wrap( -1 );
	InfoSizer->Add( MaxIterationsStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText79 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("b-factor :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText79->Wrap( -1 );
	m_staticText79->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText79, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	BfactorStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	BfactorStaticText->Wrap( -1 );
	InfoSizer->Add( BfactorStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText95 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Exp. Filter :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText95->Wrap( -1 );
	m_staticText95->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText95, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	ExposureFilterStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ExposureFilterStaticText->Wrap( -1 );
	InfoSizer->Add( ExposureFilterStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText99 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Restore Power :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText99->Wrap( -1 );
	m_staticText99->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText99, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	RestorePowerStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	RestorePowerStaticText->Wrap( -1 );
	InfoSizer->Add( RestorePowerStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText101 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Mask Cross :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText101->Wrap( -1 );
	m_staticText101->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText101, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MaskCrossStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MaskCrossStaticText->Wrap( -1 );
	InfoSizer->Add( MaskCrossStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText103 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Horiz. Mask :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText103->Wrap( -1 );
	m_staticText103->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText103, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	HorizontalMaskStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	HorizontalMaskStaticText->Wrap( -1 );
	InfoSizer->Add( HorizontalMaskStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	m_staticText105 = new wxStaticText( JobDetailsPanel, wxID_ANY, wxT("Vert. Mask :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText105->Wrap( -1 );
	m_staticText105->SetFont( wxFont( 10, 74, 90, 92, false, wxT("Sans") ) );
	
	InfoSizer->Add( m_staticText105, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	VerticalMaskStaticText = new wxStaticText( JobDetailsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	VerticalMaskStaticText->Wrap( -1 );
	InfoSizer->Add( VerticalMaskStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	
	JobDetailsPanel->SetSizer( InfoSizer );
	JobDetailsPanel->Layout();
	InfoSizer->Fit( JobDetailsPanel );
	bSizer73->Add( JobDetailsPanel, 1, wxALL|wxEXPAND, 5 );
	
	
	bSizer681->Add( bSizer73, 0, wxEXPAND, 5 );
	
	wxGridSizer* gSizer5;
	gSizer5 = new wxGridSizer( 0, 6, 0, 0 );
	
	
	bSizer681->Add( gSizer5, 0, wxEXPAND, 5 );
	
	ResultPanel = new UnblurResultsPanel( RightPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer681->Add( ResultPanel, 1, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer69;
	bSizer69 = new wxBoxSizer( wxHORIZONTAL );
	
	AddToGroupButton = new wxButton( RightPanel, wxID_ANY, wxT("Add Movie To Group"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer69->Add( AddToGroupButton, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	GroupComboBox = new MemoryComboBox( RightPanel, wxID_ANY, wxT("Combo!"), wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	GroupComboBox->SetMinSize( wxSize( 200,-1 ) );
	
	bSizer69->Add( GroupComboBox, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer681->Add( bSizer69, 0, wxALIGN_RIGHT, 5 );
	
	
	RightPanel->SetSizer( bSizer681 );
	RightPanel->Layout();
	bSizer681->Fit( RightPanel );
	m_splitter4->SplitVertically( m_panel13, RightPanel, 500 );
	bSizer63->Add( m_splitter4, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer63 );
	this->Layout();
	
	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( MovieAlignResultsPanel::OnUpdateUI ) );
	AllMoviesButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( MovieAlignResultsPanel::OnAllMoviesSelect ), NULL, this );
	ByFilterButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( MovieAlignResultsPanel::OnByFilterSelect ), NULL, this );
	FilterButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieAlignResultsPanel::OnDefineFilterClick ), NULL, this );
	PreviousButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieAlignResultsPanel::OnPreviousButtonClick ), NULL, this );
	NextButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieAlignResultsPanel::OnNextButtonClick ), NULL, this );
	JobDetailsToggleButton->Connect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( MovieAlignResultsPanel::OnJobDetailsToggle ), NULL, this );
	AddToGroupButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieAlignResultsPanel::OnAddToGroupClick ), NULL, this );
}

MovieAlignResultsPanel::~MovieAlignResultsPanel()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( MovieAlignResultsPanel::OnUpdateUI ) );
	AllMoviesButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( MovieAlignResultsPanel::OnAllMoviesSelect ), NULL, this );
	ByFilterButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( MovieAlignResultsPanel::OnByFilterSelect ), NULL, this );
	FilterButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieAlignResultsPanel::OnDefineFilterClick ), NULL, this );
	PreviousButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieAlignResultsPanel::OnPreviousButtonClick ), NULL, this );
	NextButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieAlignResultsPanel::OnNextButtonClick ), NULL, this );
	JobDetailsToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( MovieAlignResultsPanel::OnJobDetailsToggle ), NULL, this );
	AddToGroupButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieAlignResultsPanel::OnAddToGroupClick ), NULL, this );
	
}

ActionsPanel::ActionsPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer12;
	bSizer12 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticline3 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_VERTICAL );
	bSizer12->Add( m_staticline3, 0, wxEXPAND | wxALL, 5 );
	
	ActionsBook = new wxListbook( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLB_TOP );
	#ifdef __WXGTK__ // Small icon style not supported in GTK
	wxListView* ActionsBookListView = ActionsBook->GetListView();
	long ActionsBookFlags = ActionsBookListView->GetWindowStyleFlag();
	if( ActionsBookFlags & wxLC_SMALL_ICON )
	{
		ActionsBookFlags = ( ActionsBookFlags & ~wxLC_SMALL_ICON ) | wxLC_ICON;
	}
	ActionsBookListView->SetWindowStyleFlag( ActionsBookFlags );
	#endif
	
	bSizer12->Add( ActionsBook, 1, wxEXPAND | wxALL, 5 );
	
	
	this->SetSizer( bSizer12 );
	this->Layout();
}

ActionsPanel::~ActionsPanel()
{
}

SettingsPanel::SettingsPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer12;
	bSizer12 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticline3 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_VERTICAL );
	bSizer12->Add( m_staticline3, 0, wxEXPAND | wxALL, 5 );
	
	SettingsBook = new wxListbook( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLB_TOP );
	#ifdef __WXGTK__ // Small icon style not supported in GTK
	wxListView* SettingsBookListView = SettingsBook->GetListView();
	long SettingsBookFlags = SettingsBookListView->GetWindowStyleFlag();
	if( SettingsBookFlags & wxLC_SMALL_ICON )
	{
		SettingsBookFlags = ( SettingsBookFlags & ~wxLC_SMALL_ICON ) | wxLC_ICON;
	}
	SettingsBookListView->SetWindowStyleFlag( SettingsBookFlags );
	#endif
	
	bSizer12->Add( SettingsBook, 1, wxEXPAND | wxALL, 5 );
	
	
	this->SetSizer( bSizer12 );
	this->Layout();
}

SettingsPanel::~SettingsPanel()
{
}

ResultsPanel::ResultsPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer12;
	bSizer12 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticline3 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_VERTICAL );
	bSizer12->Add( m_staticline3, 0, wxEXPAND | wxALL, 5 );
	
	ResultsBook = new wxListbook( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLB_TOP );
	#ifdef __WXGTK__ // Small icon style not supported in GTK
	wxListView* ResultsBookListView = ResultsBook->GetListView();
	long ResultsBookFlags = ResultsBookListView->GetWindowStyleFlag();
	if( ResultsBookFlags & wxLC_SMALL_ICON )
	{
		ResultsBookFlags = ( ResultsBookFlags & ~wxLC_SMALL_ICON ) | wxLC_ICON;
	}
	ResultsBookListView->SetWindowStyleFlag( ResultsBookFlags );
	#endif
	
	bSizer12->Add( ResultsBook, 1, wxEXPAND | wxALL, 5 );
	
	
	this->SetSizer( bSizer12 );
	this->Layout();
	
	// Connect Events
	ResultsBook->Connect( wxEVT_COMMAND_LISTBOOK_PAGE_CHANGED, wxListbookEventHandler( ResultsPanel::OnResultsBookPageChanged ), NULL, this );
}

ResultsPanel::~ResultsPanel()
{
	// Disconnect Events
	ResultsBook->Disconnect( wxEVT_COMMAND_LISTBOOK_PAGE_CHANGED, wxListbookEventHandler( ResultsPanel::OnResultsBookPageChanged ), NULL, this );
	
}

AssetsPanel::AssetsPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer76;
	bSizer76 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticline68 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_VERTICAL );
	bSizer76->Add( m_staticline68, 0, wxEXPAND | wxALL, 5 );
	
	AssetsBook = new wxListbook( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLB_TOP );
	#ifdef __WXGTK__ // Small icon style not supported in GTK
	wxListView* AssetsBookListView = AssetsBook->GetListView();
	long AssetsBookFlags = AssetsBookListView->GetWindowStyleFlag();
	if( AssetsBookFlags & wxLC_SMALL_ICON )
	{
		AssetsBookFlags = ( AssetsBookFlags & ~wxLC_SMALL_ICON ) | wxLC_ICON;
	}
	AssetsBookListView->SetWindowStyleFlag( AssetsBookFlags );
	#endif
	
	bSizer76->Add( AssetsBook, 1, wxEXPAND | wxALL, 5 );
	
	
	this->SetSizer( bSizer76 );
	this->Layout();
}

AssetsPanel::~AssetsPanel()
{
}

OverviewPanel::OverviewPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer10;
	bSizer10 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticline2 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_VERTICAL );
	bSizer10->Add( m_staticline2, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer198;
	bSizer198 = new wxBoxSizer( wxVERTICAL );
	
	WelcomePanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer281;
	bSizer281 = new wxBoxSizer( wxVERTICAL );
	
	InfoText = new wxRichTextCtrl( WelcomePanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_READONLY|wxHSCROLL|wxVSCROLL|wxWANTS_CHARS );
	bSizer281->Add( InfoText, 1, wxEXPAND | wxALL, 5 );
	
	
	WelcomePanel->SetSizer( bSizer281 );
	WelcomePanel->Layout();
	bSizer281->Fit( WelcomePanel );
	bSizer198->Add( WelcomePanel, 1, wxEXPAND | wxALL, 5 );
	
	
	bSizer10->Add( bSizer198, 1, wxALIGN_CENTER|wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer10 );
	this->Layout();
	
	// Connect Events
	InfoText->Connect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( OverviewPanel::OnInfoURL ), NULL, this );
}

OverviewPanel::~OverviewPanel()
{
	// Disconnect Events
	InfoText->Disconnect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( OverviewPanel::OnInfoURL ), NULL, this );
	
}

MovieImportDialog::MovieImportDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxBoxSizer* bSizer26;
	bSizer26 = new wxBoxSizer( wxVERTICAL );
	
	PathListCtrl = new wxListCtrl( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_LIST|wxLC_NO_HEADER|wxLC_REPORT|wxVSCROLL );
	bSizer26->Add( PathListCtrl, 0, wxALL|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer27;
	bSizer27 = new wxBoxSizer( wxHORIZONTAL );
	
	m_button10 = new wxButton( this, wxID_ANY, wxT("Add Files"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( m_button10, 33, wxALL, 5 );
	
	m_button11 = new wxButton( this, wxID_ANY, wxT("Add Directory"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( m_button11, 33, wxALL, 5 );
	
	s = new wxButton( this, wxID_ANY, wxT("Clear"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( s, 33, wxALL, 5 );
	
	
	bSizer26->Add( bSizer27, 0, wxEXPAND, 5 );
	
	m_staticline7 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer26->Add( m_staticline7, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer28;
	bSizer28 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText19 = new wxStaticText( this, wxID_ANY, wxT("Voltage (kV) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText19->Wrap( -1 );
	bSizer28->Add( m_staticText19, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	VoltageCombo = new wxComboBox( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, 0 );
	VoltageCombo->Append( wxT("300") );
	VoltageCombo->Append( wxT("200") );
	VoltageCombo->Append( wxT("120") );
	bSizer28->Add( VoltageCombo, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer26->Add( bSizer28, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer30;
	bSizer30 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText21 = new wxStaticText( this, wxID_ANY, wxT("Spherical Aberration (mm) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21->Wrap( -1 );
	bSizer30->Add( m_staticText21, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	CsText = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer30->Add( CsText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer26->Add( bSizer30, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer29;
	bSizer29 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText20 = new wxStaticText( this, wxID_ANY, wxT("Pixel Size (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText20->Wrap( -1 );
	bSizer29->Add( m_staticText20, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	PixelSizeText = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer29->Add( PixelSizeText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer26->Add( bSizer29, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer32;
	bSizer32 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText22 = new wxStaticText( this, wxID_ANY, wxT("Exposure per frame (e¯/Å²) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText22->Wrap( -1 );
	bSizer32->Add( m_staticText22, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	DoseText = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer32->Add( DoseText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer26->Add( bSizer32, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer321;
	bSizer321 = new wxBoxSizer( wxHORIZONTAL );
	
	MoviesAreGainCorrectedCheckBox = new wxCheckBox( this, wxID_ANY, wxT("Movies are gain corrected"), wxDefaultPosition, wxDefaultSize, 0 );
	MoviesAreGainCorrectedCheckBox->SetValue(true); 
	bSizer321->Add( MoviesAreGainCorrectedCheckBox, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer321->Add( 0, 0, 50, wxEXPAND, 5 );
	
	
	bSizer26->Add( bSizer321, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer229;
	bSizer229 = new wxBoxSizer( wxHORIZONTAL );
	
	GainFileStaticText = new wxStaticText( this, wxID_ANY, wxT("        Camera gain image :"), wxDefaultPosition, wxDefaultSize, 0 );
	GainFileStaticText->Wrap( -1 );
	GainFileStaticText->Enable( false );
	
	bSizer229->Add( GainFileStaticText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	GainFilePicker = new wxFilePickerCtrl( this, wxID_ANY, wxEmptyString, wxT("Select a file"), wxT("Any supported type (*.mrc;*.tif;*.dm;*.dm3;*.dm4)|*.mrc;*.tif;*.dm;*.dm3;*.dm4|MRC files (*.mrc) | *.mrc |TIFF files (*.tif) | *.tif | DM files (*.dm;*.dm3;*.dm4)|*.dm;*.dm3;*.dm4"), wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE|wxFLP_FILE_MUST_EXIST );
	GainFilePicker->Enable( false );
	
	bSizer229->Add( GainFilePicker, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer26->Add( bSizer229, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer3211;
	bSizer3211 = new wxBoxSizer( wxHORIZONTAL );
	
	ResampleMoviesCheckBox = new wxCheckBox( this, wxID_ANY, wxT("Resample movies during processing"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer3211->Add( ResampleMoviesCheckBox, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer3211->Add( 0, 0, 50, wxEXPAND, 5 );
	
	
	bSizer26->Add( bSizer3211, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer2291;
	bSizer2291 = new wxBoxSizer( wxHORIZONTAL );
	
	DesiredPixelSizeStaticText = new wxStaticText( this, wxID_ANY, wxT("        Desired pixel size (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	DesiredPixelSizeStaticText->Wrap( -1 );
	DesiredPixelSizeStaticText->Enable( false );
	
	bSizer2291->Add( DesiredPixelSizeStaticText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	DesiredPixelSizeTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("1.00"), wxDefaultPosition, wxDefaultSize, 0 );
	DesiredPixelSizeTextCtrl->Enable( false );
	
	bSizer2291->Add( DesiredPixelSizeTextCtrl, 50, wxALL, 5 );
	
	
	bSizer26->Add( bSizer2291, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer282;
	bSizer282 = new wxBoxSizer( wxVERTICAL );
	
	CorrectMagDistortionCheckBox = new wxCheckBox( this, wxID_ANY, wxT("Correct magnification distortion"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer282->Add( CorrectMagDistortionCheckBox, 0, wxALL, 5 );
	
	wxBoxSizer* bSizer283;
	bSizer283 = new wxBoxSizer( wxHORIZONTAL );
	
	DistortionAngleStaticText = new wxStaticText( this, wxID_ANY, wxT("        Distortion Angle (°) :"), wxDefaultPosition, wxDefaultSize, 0 );
	DistortionAngleStaticText->Wrap( -1 );
	DistortionAngleStaticText->Enable( false );
	
	bSizer283->Add( DistortionAngleStaticText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	DistortionAngleTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("0.00"), wxDefaultPosition, wxDefaultSize, 0 );
	DistortionAngleTextCtrl->Enable( false );
	
	bSizer283->Add( DistortionAngleTextCtrl, 50, wxALL, 5 );
	
	
	bSizer282->Add( bSizer283, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer2831;
	bSizer2831 = new wxBoxSizer( wxHORIZONTAL );
	
	MajorScaleStaticText = new wxStaticText( this, wxID_ANY, wxT("        Major Scale  :"), wxDefaultPosition, wxDefaultSize, 0 );
	MajorScaleStaticText->Wrap( -1 );
	MajorScaleStaticText->Enable( false );
	
	bSizer2831->Add( MajorScaleStaticText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	MajorScaleTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("1.000"), wxDefaultPosition, wxDefaultSize, 0 );
	MajorScaleTextCtrl->Enable( false );
	
	bSizer2831->Add( MajorScaleTextCtrl, 50, wxALL, 5 );
	
	
	bSizer282->Add( bSizer2831, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer28311;
	bSizer28311 = new wxBoxSizer( wxHORIZONTAL );
	
	MinorScaleStaticText = new wxStaticText( this, wxID_ANY, wxT("        Minor Scale  :"), wxDefaultPosition, wxDefaultSize, 0 );
	MinorScaleStaticText->Wrap( -1 );
	MinorScaleStaticText->Enable( false );
	
	bSizer28311->Add( MinorScaleStaticText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	MinorScaleTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("1.000"), wxDefaultPosition, wxDefaultSize, 0 );
	MinorScaleTextCtrl->Enable( false );
	
	bSizer28311->Add( MinorScaleTextCtrl, 50, wxALL, 5 );
	
	
	bSizer282->Add( bSizer28311, 1, wxEXPAND, 5 );
	
	
	bSizer26->Add( bSizer282, 0, wxEXPAND, 5 );
	
	m_staticline8 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer26->Add( m_staticline8, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer33;
	bSizer33 = new wxBoxSizer( wxHORIZONTAL );
	
	m_button13 = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer33->Add( m_button13, 0, wxALL, 5 );
	
	
	bSizer33->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	bSizer33->Add( 0, 0, 1, wxEXPAND, 5 );
	
	ImportButton = new wxButton( this, wxID_ANY, wxT("Import"), wxDefaultPosition, wxDefaultSize, 0 );
	ImportButton->Enable( false );
	
	bSizer33->Add( ImportButton, 0, wxALL, 5 );
	
	
	bSizer26->Add( bSizer33, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer26 );
	this->Layout();
	bSizer26->Fit( this );
	
	this->Centre( wxBOTH );
	
	// Connect Events
	m_button10->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::AddFilesClick ), NULL, this );
	m_button11->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::AddDirectoryClick ), NULL, this );
	s->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::ClearClick ), NULL, this );
	VoltageCombo->Connect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	VoltageCombo->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	CsText->Connect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	CsText->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	PixelSizeText->Connect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	PixelSizeText->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	DoseText->Connect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	DoseText->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	MoviesAreGainCorrectedCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MovieImportDialog::OnMoviesAreGainCorrectedCheckBox ), NULL, this );
	GainFilePicker->Connect( wxEVT_COMMAND_FILEPICKER_CHANGED, wxFileDirPickerEventHandler( MovieImportDialog::OnGainFilePickerChanged ), NULL, this );
	ResampleMoviesCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MovieImportDialog::OnResampleMoviesCheckBox ), NULL, this );
	DesiredPixelSizeTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	CorrectMagDistortionCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MovieImportDialog::OnCorrectMagDistortionCheckBox ), NULL, this );
	DistortionAngleTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	MajorScaleTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	MinorScaleTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	m_button13->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::CancelClick ), NULL, this );
	ImportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::ImportClick ), NULL, this );
}

MovieImportDialog::~MovieImportDialog()
{
	// Disconnect Events
	m_button10->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::AddFilesClick ), NULL, this );
	m_button11->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::AddDirectoryClick ), NULL, this );
	s->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::ClearClick ), NULL, this );
	VoltageCombo->Disconnect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	VoltageCombo->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	CsText->Disconnect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	CsText->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	PixelSizeText->Disconnect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	PixelSizeText->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	DoseText->Disconnect( wxEVT_CHAR, wxKeyEventHandler( MovieImportDialog::OnTextKeyPress ), NULL, this );
	DoseText->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	MoviesAreGainCorrectedCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MovieImportDialog::OnMoviesAreGainCorrectedCheckBox ), NULL, this );
	GainFilePicker->Disconnect( wxEVT_COMMAND_FILEPICKER_CHANGED, wxFileDirPickerEventHandler( MovieImportDialog::OnGainFilePickerChanged ), NULL, this );
	ResampleMoviesCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MovieImportDialog::OnResampleMoviesCheckBox ), NULL, this );
	DesiredPixelSizeTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	CorrectMagDistortionCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( MovieImportDialog::OnCorrectMagDistortionCheckBox ), NULL, this );
	DistortionAngleTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	MajorScaleTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	MinorScaleTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( MovieImportDialog::TextChanged ), NULL, this );
	m_button13->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::CancelClick ), NULL, this );
	ImportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( MovieImportDialog::ImportClick ), NULL, this );
	
}

VolumeImportDialog::VolumeImportDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxBoxSizer* bSizer26;
	bSizer26 = new wxBoxSizer( wxVERTICAL );
	
	PathListCtrl = new wxListCtrl( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_LIST|wxLC_NO_HEADER|wxLC_REPORT|wxVSCROLL );
	bSizer26->Add( PathListCtrl, 100, wxALL|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer27;
	bSizer27 = new wxBoxSizer( wxHORIZONTAL );
	
	m_button10 = new wxButton( this, wxID_ANY, wxT("Add Files"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( m_button10, 33, wxALL, 5 );
	
	m_button11 = new wxButton( this, wxID_ANY, wxT("Add Directory"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( m_button11, 33, wxALL, 5 );
	
	ClearButton = new wxButton( this, wxID_ANY, wxT("Clear"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( ClearButton, 33, wxALL, 5 );
	
	
	bSizer26->Add( bSizer27, 0, wxEXPAND, 5 );
	
	m_staticline7 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer26->Add( m_staticline7, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer29;
	bSizer29 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText20 = new wxStaticText( this, wxID_ANY, wxT("Pixel Size (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText20->Wrap( -1 );
	bSizer29->Add( m_staticText20, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	PixelSizeText = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer29->Add( PixelSizeText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer26->Add( bSizer29, 0, wxEXPAND, 5 );
	
	m_staticline8 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer26->Add( m_staticline8, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer33;
	bSizer33 = new wxBoxSizer( wxHORIZONTAL );
	
	m_button13 = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer33->Add( m_button13, 0, wxALL, 5 );
	
	
	bSizer33->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	bSizer33->Add( 0, 0, 1, wxEXPAND, 5 );
	
	ImportButton = new wxButton( this, wxID_ANY, wxT("Import"), wxDefaultPosition, wxDefaultSize, 0 );
	ImportButton->Enable( false );
	
	bSizer33->Add( ImportButton, 0, wxALL, 5 );
	
	
	bSizer26->Add( bSizer33, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer26 );
	this->Layout();
	
	this->Centre( wxBOTH );
	
	// Connect Events
	m_button10->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::AddFilesClick ), NULL, this );
	m_button11->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::AddDirectoryClick ), NULL, this );
	ClearButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::ClearClick ), NULL, this );
	PixelSizeText->Connect( wxEVT_CHAR, wxKeyEventHandler( VolumeImportDialog::OnTextKeyPress ), NULL, this );
	PixelSizeText->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( VolumeImportDialog::TextChanged ), NULL, this );
	m_button13->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::CancelClick ), NULL, this );
	ImportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::ImportClick ), NULL, this );
}

VolumeImportDialog::~VolumeImportDialog()
{
	// Disconnect Events
	m_button10->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::AddFilesClick ), NULL, this );
	m_button11->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::AddDirectoryClick ), NULL, this );
	ClearButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::ClearClick ), NULL, this );
	PixelSizeText->Disconnect( wxEVT_CHAR, wxKeyEventHandler( VolumeImportDialog::OnTextKeyPress ), NULL, this );
	PixelSizeText->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( VolumeImportDialog::TextChanged ), NULL, this );
	m_button13->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::CancelClick ), NULL, this );
	ImportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeImportDialog::ImportClick ), NULL, this );
	
}

FindParticlesPanel::FindParticlesPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : JobPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer43;
	bSizer43 = new wxBoxSizer( wxVERTICAL );
	
	m_staticline12 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline12, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer1211;
	bSizer1211 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer45;
	bSizer45 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer44;
	bSizer44 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText21 = new wxStaticText( this, wxID_ANY, wxT("Input Group :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21->Wrap( -1 );
	bSizer44->Add( m_staticText21, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	GroupComboBox = new MemoryComboBox( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	bSizer44->Add( GroupComboBox, 40, wxALL, 5 );
	
	m_staticText211 = new wxStaticText( this, wxID_ANY, wxT("Picking algorithm :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText211->Wrap( -1 );
	bSizer44->Add( m_staticText211, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	PickingAlgorithmComboBox = new wxComboBox( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	PickingAlgorithmComboBox->SetMinSize( wxSize( 150,-1 ) );
	
	bSizer44->Add( PickingAlgorithmComboBox, 0, wxALL, 5 );
	
	
	bSizer44->Add( 0, 0, 60, wxEXPAND, 5 );
	
	
	bSizer45->Add( bSizer44, 1, wxEXPAND, 5 );
	
	ExpertToggleButton = new wxToggleButton( this, wxID_ANY, wxT("Show Expert Options"), wxDefaultPosition, wxDefaultSize, 0 );
	ExpertToggleButton->Enable( false );
	
	bSizer45->Add( ExpertToggleButton, 0, wxALL, 5 );
	
	
	bSizer1211->Add( bSizer45, 1, wxEXPAND, 5 );
	
	PleaseEstimateCTFStaticText = new wxStaticText( this, wxID_ANY, wxT("Please run CTF estimation on this group before picking particles"), wxDefaultPosition, wxDefaultSize, 0 );
	PleaseEstimateCTFStaticText->Wrap( -1 );
	PleaseEstimateCTFStaticText->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	PleaseEstimateCTFStaticText->SetForegroundColour( wxColour( 180, 0, 0 ) );
	PleaseEstimateCTFStaticText->Hide();
	
	bSizer1211->Add( PleaseEstimateCTFStaticText, 0, wxALL, 5 );
	
	
	bSizer43->Add( bSizer1211, 0, wxEXPAND, 5 );
	
	m_staticline10 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline10, 0, wxEXPAND | wxALL, 5 );
	
	FindParticlesSplitterWindow = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D|wxSP_PERMIT_UNSPLIT );
	FindParticlesSplitterWindow->Connect( wxEVT_IDLE, wxIdleEventHandler( FindParticlesPanel::FindParticlesSplitterWindowOnIdle ), NULL, this );
	FindParticlesSplitterWindow->SetMinimumPaneSize( 10 );
	
	LeftPanel = new wxPanel( FindParticlesSplitterWindow, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	LeftPanel->Hide();
	
	wxBoxSizer* bSizer214;
	bSizer214 = new wxBoxSizer( wxVERTICAL );
	
	PickingParametersPanel = new wxScrolledWindow( LeftPanel, wxID_ANY, wxDefaultPosition, wxSize( -1,-1 ), wxHSCROLL|wxVSCROLL );
	PickingParametersPanel->SetScrollRate( 5, 5 );
	PickingParametersPanel->Hide();
	
	InputSizer = new wxBoxSizer( wxVERTICAL );
	
	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer1->AddGrowableCol( 1 );
	fgSizer1->SetFlexibleDirection( wxBOTH );
	fgSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticText196 = new wxStaticText( PickingParametersPanel, wxID_ANY, wxT("Maximum particle radius (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText196->Wrap( -1 );
	fgSizer1->Add( m_staticText196, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	MaximumParticleRadiusNumericCtrl = new NumericTextCtrl( PickingParametersPanel, wxID_ANY, wxT("120.0"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( MaximumParticleRadiusNumericCtrl, 0, wxALL|wxEXPAND, 5 );
	
	CharacteristicParticleRadiusStaticText = new wxStaticText( PickingParametersPanel, wxID_ANY, wxT("Characteristic particle radius (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	CharacteristicParticleRadiusStaticText->Wrap( -1 );
	fgSizer1->Add( CharacteristicParticleRadiusStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	CharacteristicParticleRadiusNumericCtrl = new NumericTextCtrl( PickingParametersPanel, wxID_ANY, wxT("80.0"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( CharacteristicParticleRadiusNumericCtrl, 0, wxALL|wxEXPAND, 5 );
	
	ThresholdPeakHeightStaticText1 = new wxStaticText( PickingParametersPanel, wxID_ANY, wxT("Threshold peak height :"), wxDefaultPosition, wxDefaultSize, 0 );
	ThresholdPeakHeightStaticText1->Wrap( -1 );
	fgSizer1->Add( ThresholdPeakHeightStaticText1, 0, wxALL, 5 );
	
	ThresholdPeakHeightNumericCtrl = new NumericTextCtrl( PickingParametersPanel, wxID_ANY, wxT("6.0"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( ThresholdPeakHeightNumericCtrl, 0, wxALL|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer212;
	bSizer212 = new wxBoxSizer( wxHORIZONTAL );
	
	TestOnCurrentMicrographButton = new wxButton( PickingParametersPanel, wxID_ANY, wxT("Preview"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer212->Add( TestOnCurrentMicrographButton, 0, wxALL, 5 );
	
	AutoPickRefreshCheckBox = new wxCheckBox( PickingParametersPanel, wxID_ANY, wxT("Auto preview"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer212->Add( AutoPickRefreshCheckBox, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	fgSizer1->Add( bSizer212, 1, wxEXPAND, 5 );
	
	ImageComboBox = new wxComboBox( PickingParametersPanel, wxID_ANY, wxT("List of images in current group"), wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	fgSizer1->Add( ImageComboBox, 0, wxALL|wxEXPAND, 5 );
	
	
	InputSizer->Add( fgSizer1, 0, wxEXPAND, 5 );
	
	ExpertOptionsPanel = new wxPanel( PickingParametersPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ExpertOptionsPanel->Hide();
	
	ExpertInputSizer = new wxBoxSizer( wxVERTICAL );
	
	m_staticline35 = new wxStaticLine( ExpertOptionsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	ExpertInputSizer->Add( m_staticline35, 0, wxEXPAND | wxALL, 5 );
	
	wxFlexGridSizer* ExpertOptionsSizer;
	ExpertOptionsSizer = new wxFlexGridSizer( 0, 2, 0, 0 );
	ExpertOptionsSizer->SetFlexibleDirection( wxBOTH );
	ExpertOptionsSizer->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	ExpertOptionsStaticText = new wxStaticText( ExpertOptionsPanel, wxID_ANY, wxT("Expert Options"), wxDefaultPosition, wxDefaultSize, 0 );
	ExpertOptionsStaticText->Wrap( -1 );
	ExpertOptionsStaticText->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, true, wxEmptyString ) );
	
	ExpertOptionsSizer->Add( ExpertOptionsStaticText, 0, wxALL, 5 );
	
	
	ExpertOptionsSizer->Add( 0, 0, 1, wxEXPAND, 5 );
	
	HighestResolutionStaticText = new wxStaticText( ExpertOptionsPanel, wxID_ANY, wxT("Highest resolution used in picking (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	HighestResolutionStaticText->Wrap( -1 );
	ExpertOptionsSizer->Add( HighestResolutionStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	HighestResolutionNumericCtrl = new NumericTextCtrl( ExpertOptionsPanel, wxID_ANY, wxT("15.0"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	ExpertOptionsSizer->Add( HighestResolutionNumericCtrl, 0, wxALL|wxEXPAND, 5 );
	
	SetMinimumDistanceFromEdgesCheckBox = new wxCheckBox( ExpertOptionsPanel, wxID_ANY, wxT("Set minimum distance from edges (pix.) : "), wxDefaultPosition, wxDefaultSize, 0 );
	ExpertOptionsSizer->Add( SetMinimumDistanceFromEdgesCheckBox, 0, wxALL, 5 );
	
	MinimumDistanceFromEdgesSpinCtrl = new wxSpinCtrl( ExpertOptionsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 999999999, 128 );
	MinimumDistanceFromEdgesSpinCtrl->Enable( false );
	
	ExpertOptionsSizer->Add( MinimumDistanceFromEdgesSpinCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_checkBox8 = new wxCheckBox( ExpertOptionsPanel, wxID_ANY, wxT("Use radial averages of templates"), wxDefaultPosition, wxDefaultSize, 0 );
	m_checkBox8->SetValue(true); 
	m_checkBox8->Enable( false );
	
	ExpertOptionsSizer->Add( m_checkBox8, 0, wxALL, 5 );
	
	
	ExpertOptionsSizer->Add( 0, 0, 1, wxEXPAND, 5 );
	
	m_checkBox9 = new wxCheckBox( ExpertOptionsPanel, wxID_ANY, wxT("Rotate each template this many times : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_checkBox9->Enable( false );
	
	ExpertOptionsSizer->Add( m_checkBox9, 0, wxALL, 5 );
	
	NumberOfTemplateRotationsSpinCtrl = new wxSpinCtrl( ExpertOptionsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 360, 72 );
	NumberOfTemplateRotationsSpinCtrl->Enable( false );
	
	ExpertOptionsSizer->Add( NumberOfTemplateRotationsSpinCtrl, 0, wxALL|wxEXPAND, 5 );
	
	AvoidHighVarianceAreasCheckBox = new wxCheckBox( ExpertOptionsPanel, wxID_ANY, wxT("Avoid high variance areas"), wxDefaultPosition, wxDefaultSize, 0 );
	AvoidHighVarianceAreasCheckBox->SetValue(true); 
	ExpertOptionsSizer->Add( AvoidHighVarianceAreasCheckBox, 0, wxALL, 5 );
	
	
	ExpertOptionsSizer->Add( 0, 0, 1, wxEXPAND, 5 );
	
	AvoidAbnormalLocalMeanAreasCheckBox = new wxCheckBox( ExpertOptionsPanel, wxID_ANY, wxT("Avoid areas with abnormal local mean"), wxDefaultPosition, wxDefaultSize, 0 );
	AvoidAbnormalLocalMeanAreasCheckBox->SetValue(true); 
	ExpertOptionsSizer->Add( AvoidAbnormalLocalMeanAreasCheckBox, 0, wxALL, 5 );
	
	
	ExpertOptionsSizer->Add( 0, 0, 1, wxEXPAND, 5 );
	
	ShowEstimatedBackgroundSpectrumCheckBox = new wxCheckBox( ExpertOptionsPanel, wxID_ANY, wxT("Show estimated background spectrum"), wxDefaultPosition, wxDefaultSize, 0 );
	ShowEstimatedBackgroundSpectrumCheckBox->Enable( false );
	
	ExpertOptionsSizer->Add( ShowEstimatedBackgroundSpectrumCheckBox, 0, wxALL, 5 );
	
	
	ExpertOptionsSizer->Add( 0, 0, 1, wxEXPAND, 5 );
	
	ShowPositionsOfBackgroundBoxesCheckBox = new wxCheckBox( ExpertOptionsPanel, wxID_ANY, wxT("Show positions of background boxes"), wxDefaultPosition, wxDefaultSize, 0 );
	ShowPositionsOfBackgroundBoxesCheckBox->Enable( false );
	
	ExpertOptionsSizer->Add( ShowPositionsOfBackgroundBoxesCheckBox, 0, wxALL, 5 );
	
	
	ExpertOptionsSizer->Add( 0, 0, 1, wxEXPAND, 5 );
	
	m_staticText170 = new wxStaticText( ExpertOptionsPanel, wxID_ANY, wxT("Number of background boxes : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText170->Wrap( -1 );
	ExpertOptionsSizer->Add( m_staticText170, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	NumberOfBackgroundBoxesSpinCtrl = new wxSpinCtrl( ExpertOptionsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 999, 50 );
	ExpertOptionsSizer->Add( NumberOfBackgroundBoxesSpinCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText169 = new wxStaticText( ExpertOptionsPanel, wxID_ANY, wxT("Algorithm to find background areas : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText169->Wrap( -1 );
	ExpertOptionsSizer->Add( m_staticText169, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	wxString AlgorithmToFindBackgroundChoiceChoices[] = { wxT("Lowest variance"), wxT("Variance near mode") };
	int AlgorithmToFindBackgroundChoiceNChoices = sizeof( AlgorithmToFindBackgroundChoiceChoices ) / sizeof( wxString );
	AlgorithmToFindBackgroundChoice = new wxChoice( ExpertOptionsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, AlgorithmToFindBackgroundChoiceNChoices, AlgorithmToFindBackgroundChoiceChoices, 0 );
	AlgorithmToFindBackgroundChoice->SetSelection( 0 );
	ExpertOptionsSizer->Add( AlgorithmToFindBackgroundChoice, 0, wxALL|wxEXPAND, 5 );
	
	
	ExpertInputSizer->Add( ExpertOptionsSizer, 1, wxEXPAND, 5 );
	
	
	ExpertOptionsPanel->SetSizer( ExpertInputSizer );
	ExpertOptionsPanel->Layout();
	ExpertInputSizer->Fit( ExpertOptionsPanel );
	InputSizer->Add( ExpertOptionsPanel, 0, wxALL|wxEXPAND, 0 );
	
	
	PickingParametersPanel->SetSizer( InputSizer );
	PickingParametersPanel->Layout();
	InputSizer->Fit( PickingParametersPanel );
	bSizer214->Add( PickingParametersPanel, 1, wxALL|wxEXPAND, 5 );
	
	OutputTextPanel = new wxPanel( LeftPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	OutputTextPanel->Hide();
	
	wxBoxSizer* bSizer56;
	bSizer56 = new wxBoxSizer( wxVERTICAL );
	
	output_textctrl = new wxTextCtrl( OutputTextPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE|wxTE_READONLY );
	bSizer56->Add( output_textctrl, 1, wxALL|wxEXPAND, 5 );
	
	
	OutputTextPanel->SetSizer( bSizer56 );
	OutputTextPanel->Layout();
	bSizer56->Fit( OutputTextPanel );
	bSizer214->Add( OutputTextPanel, 30, wxEXPAND | wxALL, 5 );
	
	
	LeftPanel->SetSizer( bSizer214 );
	LeftPanel->Layout();
	bSizer214->Fit( LeftPanel );
	RightPanel = new wxPanel( FindParticlesSplitterWindow, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer215;
	bSizer215 = new wxBoxSizer( wxVERTICAL );
	
	PickingResultsPanel = new PickingResultsDisplayPanel( RightPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	PickingResultsPanel->Hide();
	
	bSizer215->Add( PickingResultsPanel, 1, wxEXPAND | wxALL, 5 );
	
	InfoPanel = new wxPanel( RightPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer61;
	bSizer61 = new wxBoxSizer( wxVERTICAL );
	
	InfoText = new wxRichTextCtrl( InfoPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_READONLY|wxHSCROLL|wxVSCROLL );
	bSizer61->Add( InfoText, 1, wxEXPAND | wxALL, 5 );
	
	
	InfoPanel->SetSizer( bSizer61 );
	InfoPanel->Layout();
	bSizer61->Fit( InfoPanel );
	bSizer215->Add( InfoPanel, 1, wxEXPAND | wxALL, 5 );
	
	
	RightPanel->SetSizer( bSizer215 );
	RightPanel->Layout();
	bSizer215->Fit( RightPanel );
	FindParticlesSplitterWindow->SplitVertically( LeftPanel, RightPanel, 528 );
	bSizer43->Add( FindParticlesSplitterWindow, 1, wxEXPAND, 5 );
	
	m_staticline11 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline11, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer48;
	bSizer48 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer70;
	bSizer70 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer71;
	bSizer71 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer55;
	bSizer55 = new wxBoxSizer( wxHORIZONTAL );
	
	
	bSizer71->Add( bSizer55, 0, wxALIGN_CENTER_HORIZONTAL, 5 );
	
	
	bSizer70->Add( bSizer71, 1, wxALIGN_CENTER_VERTICAL, 5 );
	
	wxBoxSizer* bSizer74;
	bSizer74 = new wxBoxSizer( wxVERTICAL );
	
	
	bSizer70->Add( bSizer74, 0, wxALIGN_CENTER_VERTICAL, 5 );
	
	ProgressPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ProgressPanel->Hide();
	
	wxBoxSizer* bSizer57;
	bSizer57 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer59;
	bSizer59 = new wxBoxSizer( wxHORIZONTAL );
	
	NumberConnectedText = new wxStaticText( ProgressPanel, wxID_ANY, wxT("o"), wxDefaultPosition, wxDefaultSize, 0 );
	NumberConnectedText->Wrap( -1 );
	bSizer59->Add( NumberConnectedText, 0, wxALIGN_CENTER|wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	ProgressBar = new wxGauge( ProgressPanel, wxID_ANY, 100, wxDefaultPosition, wxDefaultSize, wxGA_HORIZONTAL );
	ProgressBar->SetValue( 0 ); 
	bSizer59->Add( ProgressBar, 100, wxALIGN_CENTER|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );
	
	TimeRemainingText = new wxStaticText( ProgressPanel, wxID_ANY, wxT("Time Remaining : ???h:??m:??s   "), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	TimeRemainingText->Wrap( -1 );
	bSizer59->Add( TimeRemainingText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	m_staticline60 = new wxStaticLine( ProgressPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer59->Add( m_staticline60, 0, wxEXPAND | wxALL, 5 );
	
	FinishButton = new wxButton( ProgressPanel, wxID_ANY, wxT("Finish"), wxDefaultPosition, wxDefaultSize, 0 );
	FinishButton->Hide();
	
	bSizer59->Add( FinishButton, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	CancelAlignmentButton = new wxButton( ProgressPanel, wxID_ANY, wxT("Terminate Job"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer59->Add( CancelAlignmentButton, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer57->Add( bSizer59, 1, wxEXPAND, 5 );
	
	
	ProgressPanel->SetSizer( bSizer57 );
	ProgressPanel->Layout();
	bSizer57->Fit( ProgressPanel );
	bSizer70->Add( ProgressPanel, 1, wxEXPAND | wxALL, 5 );
	
	
	bSizer48->Add( bSizer70, 1, wxEXPAND, 5 );
	
	StartPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer58;
	bSizer58 = new wxBoxSizer( wxHORIZONTAL );
	
	RunProfileText = new wxStaticText( StartPanel, wxID_ANY, wxT("Run Profile :"), wxDefaultPosition, wxDefaultSize, 0 );
	RunProfileText->Wrap( -1 );
	bSizer58->Add( RunProfileText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	RunProfileComboBox = new MemoryComboBox( StartPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	bSizer58->Add( RunProfileComboBox, 50, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer60;
	bSizer60 = new wxBoxSizer( wxVERTICAL );
	
	StartPickingButton = new wxButton( StartPanel, wxID_ANY, wxT("Start Picking"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer60->Add( StartPickingButton, 0, wxALL, 5 );
	
	
	bSizer58->Add( bSizer60, 50, wxEXPAND, 5 );
	
	
	StartPanel->SetSizer( bSizer58 );
	StartPanel->Layout();
	bSizer58->Fit( StartPanel );
	bSizer48->Add( StartPanel, 1, wxEXPAND | wxALL, 5 );
	
	
	bSizer43->Add( bSizer48, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer43 );
	this->Layout();
	
	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( FindParticlesPanel::OnUpdateUI ) );
	GroupComboBox->Connect( wxEVT_COMMAND_COMBOBOX_SELECTED, wxCommandEventHandler( FindParticlesPanel::OnGroupComboBox ), NULL, this );
	PickingAlgorithmComboBox->Connect( wxEVT_COMMAND_COMBOBOX_SELECTED, wxCommandEventHandler( FindParticlesPanel::OnPickingAlgorithmComboBox ), NULL, this );
	ExpertToggleButton->Connect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( FindParticlesPanel::OnExpertOptionsToggle ), NULL, this );
	MaximumParticleRadiusNumericCtrl->Connect( wxEVT_KILL_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnMaximumParticleRadiusNumericTextKillFocus ), NULL, this );
	MaximumParticleRadiusNumericCtrl->Connect( wxEVT_SET_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnMaximumParticleRadiusNumericTextSetFocus ), NULL, this );
	MaximumParticleRadiusNumericCtrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( FindParticlesPanel::OnMaximumParticleRadiusNumericTextEnter ), NULL, this );
	CharacteristicParticleRadiusNumericCtrl->Connect( wxEVT_KILL_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnCharacteristicParticleRadiusNumericTextKillFocus ), NULL, this );
	CharacteristicParticleRadiusNumericCtrl->Connect( wxEVT_SET_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnCharacteristicParticleRadiusNumericTextSetFocus ), NULL, this );
	CharacteristicParticleRadiusNumericCtrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( FindParticlesPanel::OnCharacteristicParticleRadiusNumericTextEnter ), NULL, this );
	ThresholdPeakHeightNumericCtrl->Connect( wxEVT_KILL_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnThresholdPeakHeightNumericTextKillFocus ), NULL, this );
	ThresholdPeakHeightNumericCtrl->Connect( wxEVT_SET_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnThresholdPeakHeightNumericTextSetFocus ), NULL, this );
	ThresholdPeakHeightNumericCtrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( FindParticlesPanel::OnThresholdPeakHeightNumericTextEnter ), NULL, this );
	TestOnCurrentMicrographButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindParticlesPanel::OnTestOnCurrentMicrographButtonClick ), NULL, this );
	AutoPickRefreshCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindParticlesPanel::OnAutoPickRefreshCheckBox ), NULL, this );
	ImageComboBox->Connect( wxEVT_COMMAND_COMBOBOX_SELECTED, wxCommandEventHandler( FindParticlesPanel::OnImageComboBox ), NULL, this );
	HighestResolutionNumericCtrl->Connect( wxEVT_KILL_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnHighestResolutionNumericKillFocus ), NULL, this );
	HighestResolutionNumericCtrl->Connect( wxEVT_SET_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnHighestResolutionNumericSetFocus ), NULL, this );
	HighestResolutionNumericCtrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( FindParticlesPanel::OnHighestResolutionNumericTextEnter ), NULL, this );
	SetMinimumDistanceFromEdgesCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindParticlesPanel::OnSetMinimumDistanceFromEdgesCheckBox ), NULL, this );
	MinimumDistanceFromEdgesSpinCtrl->Connect( wxEVT_COMMAND_SPINCTRL_UPDATED, wxSpinEventHandler( FindParticlesPanel::OnMinimumDistanceFromEdgesSpinCtrl ), NULL, this );
	AvoidHighVarianceAreasCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindParticlesPanel::OnAvoidHighVarianceAreasCheckBox ), NULL, this );
	AvoidAbnormalLocalMeanAreasCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindParticlesPanel::OnAvoidAbnormalLocalMeanAreasCheckBox ), NULL, this );
	NumberOfBackgroundBoxesSpinCtrl->Connect( wxEVT_COMMAND_SPINCTRL_UPDATED, wxSpinEventHandler( FindParticlesPanel::OnNumberOfBackgroundBoxesSpinCtrl ), NULL, this );
	AlgorithmToFindBackgroundChoice->Connect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( FindParticlesPanel::OnAlgorithmToFindBackgroundChoice ), NULL, this );
	InfoText->Connect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( FindParticlesPanel::OnInfoURL ), NULL, this );
	FinishButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindParticlesPanel::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindParticlesPanel::TerminateButtonClick ), NULL, this );
	StartPickingButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindParticlesPanel::StartPickingClick ), NULL, this );
}

FindParticlesPanel::~FindParticlesPanel()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( FindParticlesPanel::OnUpdateUI ) );
	GroupComboBox->Disconnect( wxEVT_COMMAND_COMBOBOX_SELECTED, wxCommandEventHandler( FindParticlesPanel::OnGroupComboBox ), NULL, this );
	PickingAlgorithmComboBox->Disconnect( wxEVT_COMMAND_COMBOBOX_SELECTED, wxCommandEventHandler( FindParticlesPanel::OnPickingAlgorithmComboBox ), NULL, this );
	ExpertToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( FindParticlesPanel::OnExpertOptionsToggle ), NULL, this );
	MaximumParticleRadiusNumericCtrl->Disconnect( wxEVT_KILL_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnMaximumParticleRadiusNumericTextKillFocus ), NULL, this );
	MaximumParticleRadiusNumericCtrl->Disconnect( wxEVT_SET_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnMaximumParticleRadiusNumericTextSetFocus ), NULL, this );
	MaximumParticleRadiusNumericCtrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( FindParticlesPanel::OnMaximumParticleRadiusNumericTextEnter ), NULL, this );
	CharacteristicParticleRadiusNumericCtrl->Disconnect( wxEVT_KILL_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnCharacteristicParticleRadiusNumericTextKillFocus ), NULL, this );
	CharacteristicParticleRadiusNumericCtrl->Disconnect( wxEVT_SET_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnCharacteristicParticleRadiusNumericTextSetFocus ), NULL, this );
	CharacteristicParticleRadiusNumericCtrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( FindParticlesPanel::OnCharacteristicParticleRadiusNumericTextEnter ), NULL, this );
	ThresholdPeakHeightNumericCtrl->Disconnect( wxEVT_KILL_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnThresholdPeakHeightNumericTextKillFocus ), NULL, this );
	ThresholdPeakHeightNumericCtrl->Disconnect( wxEVT_SET_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnThresholdPeakHeightNumericTextSetFocus ), NULL, this );
	ThresholdPeakHeightNumericCtrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( FindParticlesPanel::OnThresholdPeakHeightNumericTextEnter ), NULL, this );
	TestOnCurrentMicrographButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindParticlesPanel::OnTestOnCurrentMicrographButtonClick ), NULL, this );
	AutoPickRefreshCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindParticlesPanel::OnAutoPickRefreshCheckBox ), NULL, this );
	ImageComboBox->Disconnect( wxEVT_COMMAND_COMBOBOX_SELECTED, wxCommandEventHandler( FindParticlesPanel::OnImageComboBox ), NULL, this );
	HighestResolutionNumericCtrl->Disconnect( wxEVT_KILL_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnHighestResolutionNumericKillFocus ), NULL, this );
	HighestResolutionNumericCtrl->Disconnect( wxEVT_SET_FOCUS, wxFocusEventHandler( FindParticlesPanel::OnHighestResolutionNumericSetFocus ), NULL, this );
	HighestResolutionNumericCtrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( FindParticlesPanel::OnHighestResolutionNumericTextEnter ), NULL, this );
	SetMinimumDistanceFromEdgesCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindParticlesPanel::OnSetMinimumDistanceFromEdgesCheckBox ), NULL, this );
	MinimumDistanceFromEdgesSpinCtrl->Disconnect( wxEVT_COMMAND_SPINCTRL_UPDATED, wxSpinEventHandler( FindParticlesPanel::OnMinimumDistanceFromEdgesSpinCtrl ), NULL, this );
	AvoidHighVarianceAreasCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindParticlesPanel::OnAvoidHighVarianceAreasCheckBox ), NULL, this );
	AvoidAbnormalLocalMeanAreasCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindParticlesPanel::OnAvoidAbnormalLocalMeanAreasCheckBox ), NULL, this );
	NumberOfBackgroundBoxesSpinCtrl->Disconnect( wxEVT_COMMAND_SPINCTRL_UPDATED, wxSpinEventHandler( FindParticlesPanel::OnNumberOfBackgroundBoxesSpinCtrl ), NULL, this );
	AlgorithmToFindBackgroundChoice->Disconnect( wxEVT_COMMAND_CHOICE_SELECTED, wxCommandEventHandler( FindParticlesPanel::OnAlgorithmToFindBackgroundChoice ), NULL, this );
	InfoText->Disconnect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( FindParticlesPanel::OnInfoURL ), NULL, this );
	FinishButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindParticlesPanel::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindParticlesPanel::TerminateButtonClick ), NULL, this );
	StartPickingButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindParticlesPanel::StartPickingClick ), NULL, this );
	
}

ImageImportDialog::ImageImportDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxBoxSizer* bSizer26;
	bSizer26 = new wxBoxSizer( wxVERTICAL );
	
	PathListCtrl = new wxListCtrl( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_LIST|wxLC_NO_HEADER|wxLC_REPORT|wxVSCROLL );
	bSizer26->Add( PathListCtrl, 100, wxALL|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer27;
	bSizer27 = new wxBoxSizer( wxHORIZONTAL );
	
	m_button10 = new wxButton( this, wxID_ANY, wxT("Add Files"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( m_button10, 33, wxALL, 5 );
	
	m_button11 = new wxButton( this, wxID_ANY, wxT("Add Directory"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( m_button11, 33, wxALL, 5 );
	
	ClearButton = new wxButton( this, wxID_ANY, wxT("Clear"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer27->Add( ClearButton, 33, wxALL, 5 );
	
	
	bSizer26->Add( bSizer27, 0, wxEXPAND, 5 );
	
	m_staticline7 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer26->Add( m_staticline7, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer28;
	bSizer28 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText19 = new wxStaticText( this, wxID_ANY, wxT("Voltage (kV) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText19->Wrap( -1 );
	bSizer28->Add( m_staticText19, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	VoltageCombo = new wxComboBox( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, 0 );
	VoltageCombo->Append( wxT("300") );
	VoltageCombo->Append( wxT("200") );
	VoltageCombo->Append( wxT("120") );
	bSizer28->Add( VoltageCombo, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer26->Add( bSizer28, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer30;
	bSizer30 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText21 = new wxStaticText( this, wxID_ANY, wxT("Spherical Aberration (mm) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21->Wrap( -1 );
	bSizer30->Add( m_staticText21, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	CsText = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer30->Add( CsText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer26->Add( bSizer30, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer29;
	bSizer29 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText20 = new wxStaticText( this, wxID_ANY, wxT("Pixel Size (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText20->Wrap( -1 );
	bSizer29->Add( m_staticText20, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	PixelSizeText = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer29->Add( PixelSizeText, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer26->Add( bSizer29, 0, wxEXPAND, 5 );
	
	m_staticline8 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer26->Add( m_staticline8, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer33;
	bSizer33 = new wxBoxSizer( wxHORIZONTAL );
	
	m_button13 = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer33->Add( m_button13, 0, wxALL, 5 );
	
	
	bSizer33->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	bSizer33->Add( 0, 0, 1, wxEXPAND, 5 );
	
	ImportButton = new wxButton( this, wxID_ANY, wxT("Import"), wxDefaultPosition, wxDefaultSize, 0 );
	ImportButton->Enable( false );
	
	bSizer33->Add( ImportButton, 0, wxALL, 5 );
	
	
	bSizer26->Add( bSizer33, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer26 );
	this->Layout();
	
	this->Centre( wxBOTH );
	
	// Connect Events
	m_button10->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::AddFilesClick ), NULL, this );
	m_button11->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::AddDirectoryClick ), NULL, this );
	ClearButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::ClearClick ), NULL, this );
	VoltageCombo->Connect( wxEVT_CHAR, wxKeyEventHandler( ImageImportDialog::OnTextKeyPress ), NULL, this );
	VoltageCombo->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImageImportDialog::TextChanged ), NULL, this );
	CsText->Connect( wxEVT_CHAR, wxKeyEventHandler( ImageImportDialog::OnTextKeyPress ), NULL, this );
	CsText->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImageImportDialog::TextChanged ), NULL, this );
	PixelSizeText->Connect( wxEVT_CHAR, wxKeyEventHandler( ImageImportDialog::OnTextKeyPress ), NULL, this );
	PixelSizeText->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImageImportDialog::TextChanged ), NULL, this );
	m_button13->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::CancelClick ), NULL, this );
	ImportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::ImportClick ), NULL, this );
}

ImageImportDialog::~ImageImportDialog()
{
	// Disconnect Events
	m_button10->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::AddFilesClick ), NULL, this );
	m_button11->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::AddDirectoryClick ), NULL, this );
	ClearButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::ClearClick ), NULL, this );
	VoltageCombo->Disconnect( wxEVT_CHAR, wxKeyEventHandler( ImageImportDialog::OnTextKeyPress ), NULL, this );
	VoltageCombo->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImageImportDialog::TextChanged ), NULL, this );
	CsText->Disconnect( wxEVT_CHAR, wxKeyEventHandler( ImageImportDialog::OnTextKeyPress ), NULL, this );
	CsText->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImageImportDialog::TextChanged ), NULL, this );
	PixelSizeText->Disconnect( wxEVT_CHAR, wxKeyEventHandler( ImageImportDialog::OnTextKeyPress ), NULL, this );
	PixelSizeText->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( ImageImportDialog::TextChanged ), NULL, this );
	m_button13->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::CancelClick ), NULL, this );
	ImportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ImageImportDialog::ImportClick ), NULL, this );
	
}

AssetParentPanel::AssetParentPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	this->SetMinSize( wxSize( 680,400 ) );
	
	wxBoxSizer* bSizer15;
	bSizer15 = new wxBoxSizer( wxVERTICAL );
	
	m_staticline5 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer15->Add( m_staticline5, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer20;
	bSizer20 = new wxBoxSizer( wxHORIZONTAL );
	
	m_splitter2 = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitter2->SetSashGravity( 0.2 );
	m_splitter2->Connect( wxEVT_IDLE, wxIdleEventHandler( AssetParentPanel::m_splitter2OnIdle ), NULL, this );
	
	m_panel4 = new wxPanel( m_splitter2, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer27;
	bSizer27 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText18 = new wxStaticText( m_panel4, wxID_ANY, wxT("Groups:"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText18->Wrap( -1 );
	bSizer27->Add( m_staticText18, 0, wxALL, 5 );
	
	wxBoxSizer* bSizer221;
	bSizer221 = new wxBoxSizer( wxHORIZONTAL );
	
	GroupListBox = new wxListCtrl( m_panel4, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_EDIT_LABELS|wxLC_NO_HEADER|wxLC_REPORT|wxLC_SINGLE_SEL );
	bSizer221->Add( GroupListBox, 1, wxALL|wxEXPAND, 5 );
	
	
	bSizer27->Add( bSizer221, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer49;
	bSizer49 = new wxBoxSizer( wxHORIZONTAL );
	
	AddGroupButton = new wxButton( m_panel4, wxID_ANY, wxT("Add"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer49->Add( AddGroupButton, 0, wxALL, 5 );
	
	RenameGroupButton = new wxButton( m_panel4, wxID_ANY, wxT("Rename"), wxDefaultPosition, wxDefaultSize, 0 );
	RenameGroupButton->Hide();
	
	bSizer49->Add( RenameGroupButton, 0, wxALL, 5 );
	
	RemoveGroupButton = new wxButton( m_panel4, wxID_ANY, wxT("Remove"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer49->Add( RemoveGroupButton, 0, wxALL, 5 );
	
	InvertGroupButton = new wxButton( m_panel4, wxID_ANY, wxT("Invert"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer49->Add( InvertGroupButton, 0, wxALL, 5 );
	
	NewFromParentButton = new wxButton( m_panel4, wxID_ANY, wxT("New From Parent"), wxDefaultPosition, wxDefaultSize, 0 );
	NewFromParentButton->Enable( false );
	
	bSizer49->Add( NewFromParentButton, 0, wxALL, 5 );
	
	
	bSizer27->Add( bSizer49, 0, wxEXPAND, 5 );
	
	
	m_panel4->SetSizer( bSizer27 );
	m_panel4->Layout();
	bSizer27->Fit( m_panel4 );
	m_panel3 = new wxPanel( m_splitter2, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer30;
	bSizer30 = new wxBoxSizer( wxVERTICAL );
	
	AssetTypeText = new wxStaticText( m_panel3, wxID_ANY, wxT("Asset Type :"), wxDefaultPosition, wxDefaultSize, 0 );
	AssetTypeText->Wrap( -1 );
	bSizer30->Add( AssetTypeText, 0, wxALL, 5 );
	
	wxBoxSizer* bSizer25;
	bSizer25 = new wxBoxSizer( wxVERTICAL );
	
	ContentsListBox = new ContentsList( m_panel3, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_ALIGN_LEFT|wxLC_NO_SORT_HEADER|wxLC_REPORT|wxLC_VIRTUAL|wxLC_VRULES );
	bSizer25->Add( ContentsListBox, 100, wxALL|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer28;
	bSizer28 = new wxBoxSizer( wxHORIZONTAL );
	
	ImportAsset = new wxButton( m_panel3, wxID_ANY, wxT("Import"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer28->Add( ImportAsset, 0, wxALL|wxEXPAND, 5 );
	
	RemoveSelectedAssetButton = new wxButton( m_panel3, wxID_ANY, wxT("Remove"), wxDefaultPosition, wxDefaultSize, 0 );
	RemoveSelectedAssetButton->Enable( false );
	
	bSizer28->Add( RemoveSelectedAssetButton, 0, wxALL|wxEXPAND, 5 );
	
	RemoveAllAssetsButton = new wxButton( m_panel3, wxID_ANY, wxT("Remove All"), wxDefaultPosition, wxDefaultSize, 0 );
	RemoveAllAssetsButton->Enable( false );
	
	bSizer28->Add( RemoveAllAssetsButton, 0, wxALL|wxEXPAND, 5 );
	
	RenameAssetButton = new wxButton( m_panel3, wxID_ANY, wxT("Rename"), wxDefaultPosition, wxDefaultSize, 0 );
	RenameAssetButton->Enable( false );
	
	bSizer28->Add( RenameAssetButton, 0, wxALL, 5 );
	
	AddSelectedAssetButton = new wxButton( m_panel3, wxID_ANY, wxT("Add To Group"), wxDefaultPosition, wxDefaultSize, 0 );
	AddSelectedAssetButton->Enable( false );
	
	bSizer28->Add( AddSelectedAssetButton, 0, wxALL|wxEXPAND, 5 );
	
	DisplayButton = new wxButton( m_panel3, wxID_ANY, wxT("Display"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer28->Add( DisplayButton, 0, wxALL, 5 );
	
	
	bSizer25->Add( bSizer28, 0, wxEXPAND, 5 );
	
	
	bSizer30->Add( bSizer25, 100, wxEXPAND, 5 );
	
	
	m_panel3->SetSizer( bSizer30 );
	m_panel3->Layout();
	bSizer30->Fit( m_panel3 );
	m_splitter2->SplitVertically( m_panel4, m_panel3, 405 );
	bSizer20->Add( m_splitter2, 100, wxEXPAND, 5 );
	
	
	bSizer15->Add( bSizer20, 1, wxEXPAND, 5 );
	
	m_staticline6 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer15->Add( m_staticline6, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer34;
	bSizer34 = new wxBoxSizer( wxHORIZONTAL );
	
	wxGridSizer* gSizer1;
	gSizer1 = new wxGridSizer( 4, 6, 0, 0 );
	
	Label0Title = new wxStaticText( this, wxID_ANY, wxT("Label 0 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label0Title->Wrap( -1 );
	Label0Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	gSizer1->Add( Label0Title, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	Label0Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label0Text->Wrap( -1 );
	gSizer1->Add( Label0Text, 0, wxALIGN_LEFT|wxALL, 5 );
	
	
	gSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	gSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	gSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	gSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	Label1Title = new wxStaticText( this, wxID_ANY, wxT("Label 1 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label1Title->Wrap( -1 );
	Label1Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	gSizer1->Add( Label1Title, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	Label1Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label1Text->Wrap( -1 );
	gSizer1->Add( Label1Text, 0, wxALL, 5 );
	
	Label2Title = new wxStaticText( this, wxID_ANY, wxT("Label 2 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label2Title->Wrap( -1 );
	Label2Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	gSizer1->Add( Label2Title, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	Label2Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label2Text->Wrap( -1 );
	gSizer1->Add( Label2Text, 0, wxALIGN_LEFT|wxALL, 5 );
	
	Label3Title = new wxStaticText( this, wxID_ANY, wxT("Label 3 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label3Title->Wrap( -1 );
	Label3Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	gSizer1->Add( Label3Title, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	Label3Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label3Text->Wrap( -1 );
	gSizer1->Add( Label3Text, 0, wxALIGN_LEFT|wxALL, 5 );
	
	Label4Title = new wxStaticText( this, wxID_ANY, wxT("Label 4 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label4Title->Wrap( -1 );
	Label4Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	gSizer1->Add( Label4Title, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	Label4Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label4Text->Wrap( -1 );
	gSizer1->Add( Label4Text, 0, wxALIGN_LEFT|wxALL, 5 );
	
	Label5Title = new wxStaticText( this, wxID_ANY, wxT("Label 5 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label5Title->Wrap( -1 );
	Label5Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	gSizer1->Add( Label5Title, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	Label5Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label5Text->Wrap( -1 );
	gSizer1->Add( Label5Text, 0, wxALL, 5 );
	
	Label6Title = new wxStaticText( this, wxID_ANY, wxT("Label 6 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label6Title->Wrap( -1 );
	Label6Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	gSizer1->Add( Label6Title, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	Label6Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label6Text->Wrap( -1 );
	gSizer1->Add( Label6Text, 0, wxALIGN_LEFT|wxALL, 5 );
	
	Label7Title = new wxStaticText( this, wxID_ANY, wxT("Label 7 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label7Title->Wrap( -1 );
	Label7Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	gSizer1->Add( Label7Title, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	Label7Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label7Text->Wrap( -1 );
	gSizer1->Add( Label7Text, 0, wxALIGN_LEFT|wxALL, 5 );
	
	Label8Title = new wxStaticText( this, wxID_ANY, wxT("Label 8 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label8Title->Wrap( -1 );
	Label8Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	gSizer1->Add( Label8Title, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	Label8Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label8Text->Wrap( -1 );
	gSizer1->Add( Label8Text, 0, wxALL, 5 );
	
	Label9Title = new wxStaticText( this, wxID_ANY, wxT("Label 9 :"), wxDefaultPosition, wxDefaultSize, 0 );
	Label9Title->Wrap( -1 );
	Label9Title->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	gSizer1->Add( Label9Title, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	Label9Text = new wxStaticText( this, wxID_ANY, wxT("-"), wxDefaultPosition, wxDefaultSize, 0 );
	Label9Text->Wrap( -1 );
	gSizer1->Add( Label9Text, 0, wxALL, 5 );
	
	
	bSizer34->Add( gSizer1, 90, wxEXPAND, 5 );
	
	
	bSizer15->Add( bSizer34, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer15 );
	this->Layout();
	
	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( AssetParentPanel::OnUpdateUI ) );
	GroupListBox->Connect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( AssetParentPanel::MouseCheckGroupsVeto ), NULL, this );
	GroupListBox->Connect( wxEVT_LEFT_DOWN, wxMouseEventHandler( AssetParentPanel::MouseCheckGroupsVeto ), NULL, this );
	GroupListBox->Connect( wxEVT_LEFT_UP, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	GroupListBox->Connect( wxEVT_COMMAND_LIST_BEGIN_LABEL_EDIT, wxListEventHandler( AssetParentPanel::OnBeginEdit ), NULL, this );
	GroupListBox->Connect( wxEVT_COMMAND_LIST_END_LABEL_EDIT, wxListEventHandler( AssetParentPanel::OnEndEdit ), NULL, this );
	GroupListBox->Connect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( AssetParentPanel::OnGroupActivated ), NULL, this );
	GroupListBox->Connect( wxEVT_COMMAND_LIST_ITEM_FOCUSED, wxListEventHandler( AssetParentPanel::OnGroupFocusChange ), NULL, this );
	GroupListBox->Connect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	GroupListBox->Connect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	GroupListBox->Connect( wxEVT_MIDDLE_UP, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	GroupListBox->Connect( wxEVT_MOTION, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	GroupListBox->Connect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	GroupListBox->Connect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	GroupListBox->Connect( wxEVT_RIGHT_UP, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	AddGroupButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::NewGroupClick ), NULL, this );
	RenameGroupButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::RenameGroupClick ), NULL, this );
	RemoveGroupButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::RemoveGroupClick ), NULL, this );
	InvertGroupButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::InvertGroupClick ), NULL, this );
	NewFromParentButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::NewFromParentClick ), NULL, this );
	ContentsListBox->Connect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( AssetParentPanel::MouseCheckContentsVeto ), NULL, this );
	ContentsListBox->Connect( wxEVT_LEFT_DOWN, wxMouseEventHandler( AssetParentPanel::MouseCheckContentsVeto ), NULL, this );
	ContentsListBox->Connect( wxEVT_LEFT_UP, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ContentsListBox->Connect( wxEVT_COMMAND_LIST_BEGIN_DRAG, wxListEventHandler( AssetParentPanel::OnBeginContentsDrag ), NULL, this );
	ContentsListBox->Connect( wxEVT_COMMAND_LIST_ITEM_SELECTED, wxListEventHandler( AssetParentPanel::OnContentsSelected ), NULL, this );
	ContentsListBox->Connect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ContentsListBox->Connect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ContentsListBox->Connect( wxEVT_MIDDLE_UP, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ContentsListBox->Connect( wxEVT_MOTION, wxMouseEventHandler( AssetParentPanel::OnMotion ), NULL, this );
	ContentsListBox->Connect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ContentsListBox->Connect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ContentsListBox->Connect( wxEVT_RIGHT_UP, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ImportAsset->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::ImportAssetClick ), NULL, this );
	RemoveSelectedAssetButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::RemoveAssetClick ), NULL, this );
	RemoveAllAssetsButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::RemoveAllAssetsClick ), NULL, this );
	RenameAssetButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::RenameAssetClick ), NULL, this );
	AddSelectedAssetButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::AddSelectedAssetClick ), NULL, this );
	DisplayButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::OnDisplayButtonClick ), NULL, this );
}

AssetParentPanel::~AssetParentPanel()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( AssetParentPanel::OnUpdateUI ) );
	GroupListBox->Disconnect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( AssetParentPanel::MouseCheckGroupsVeto ), NULL, this );
	GroupListBox->Disconnect( wxEVT_LEFT_DOWN, wxMouseEventHandler( AssetParentPanel::MouseCheckGroupsVeto ), NULL, this );
	GroupListBox->Disconnect( wxEVT_LEFT_UP, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	GroupListBox->Disconnect( wxEVT_COMMAND_LIST_BEGIN_LABEL_EDIT, wxListEventHandler( AssetParentPanel::OnBeginEdit ), NULL, this );
	GroupListBox->Disconnect( wxEVT_COMMAND_LIST_END_LABEL_EDIT, wxListEventHandler( AssetParentPanel::OnEndEdit ), NULL, this );
	GroupListBox->Disconnect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( AssetParentPanel::OnGroupActivated ), NULL, this );
	GroupListBox->Disconnect( wxEVT_COMMAND_LIST_ITEM_FOCUSED, wxListEventHandler( AssetParentPanel::OnGroupFocusChange ), NULL, this );
	GroupListBox->Disconnect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	GroupListBox->Disconnect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	GroupListBox->Disconnect( wxEVT_MIDDLE_UP, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	GroupListBox->Disconnect( wxEVT_MOTION, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	GroupListBox->Disconnect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	GroupListBox->Disconnect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	GroupListBox->Disconnect( wxEVT_RIGHT_UP, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	AddGroupButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::NewGroupClick ), NULL, this );
	RenameGroupButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::RenameGroupClick ), NULL, this );
	RemoveGroupButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::RemoveGroupClick ), NULL, this );
	InvertGroupButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::InvertGroupClick ), NULL, this );
	NewFromParentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::NewFromParentClick ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( AssetParentPanel::MouseCheckContentsVeto ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_LEFT_DOWN, wxMouseEventHandler( AssetParentPanel::MouseCheckContentsVeto ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_LEFT_UP, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_COMMAND_LIST_BEGIN_DRAG, wxListEventHandler( AssetParentPanel::OnBeginContentsDrag ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_COMMAND_LIST_ITEM_SELECTED, wxListEventHandler( AssetParentPanel::OnContentsSelected ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_MIDDLE_UP, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_MOTION, wxMouseEventHandler( AssetParentPanel::OnMotion ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ContentsListBox->Disconnect( wxEVT_RIGHT_UP, wxMouseEventHandler( AssetParentPanel::MouseVeto ), NULL, this );
	ImportAsset->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::ImportAssetClick ), NULL, this );
	RemoveSelectedAssetButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::RemoveAssetClick ), NULL, this );
	RemoveAllAssetsButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::RemoveAllAssetsClick ), NULL, this );
	RenameAssetButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::RenameAssetClick ), NULL, this );
	AddSelectedAssetButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::AddSelectedAssetClick ), NULL, this );
	DisplayButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AssetParentPanel::OnDisplayButtonClick ), NULL, this );
	
}

RunProfilesPanel::RunProfilesPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	this->SetMinSize( wxSize( 680,400 ) );
	
	wxBoxSizer* bSizer56;
	bSizer56 = new wxBoxSizer( wxVERTICAL );
	
	m_staticline12 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer56->Add( m_staticline12, 0, wxEXPAND | wxALL, 5 );
	
	m_splitter5 = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitter5->Connect( wxEVT_IDLE, wxIdleEventHandler( RunProfilesPanel::m_splitter5OnIdle ), NULL, this );
	m_splitter5->SetMinimumPaneSize( 200 );
	
	ProfilesPanel = new wxPanel( m_splitter5, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer57;
	bSizer57 = new wxBoxSizer( wxHORIZONTAL );
	
	ProfilesListBox = new wxListCtrl( ProfilesPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_EDIT_LABELS|wxLC_NO_HEADER|wxLC_REPORT|wxLC_SINGLE_SEL );
	bSizer57->Add( ProfilesListBox, 1, wxALL|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer431;
	bSizer431 = new wxBoxSizer( wxVERTICAL );
	
	AddProfileButton = new wxButton( ProfilesPanel, wxID_ANY, wxT("Add"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer431->Add( AddProfileButton, 1, wxALL, 5 );
	
	RenameProfileButton = new wxButton( ProfilesPanel, wxID_ANY, wxT("Rename"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer431->Add( RenameProfileButton, 0, wxALL, 5 );
	
	RemoveProfileButton = new wxButton( ProfilesPanel, wxID_ANY, wxT("Remove"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer431->Add( RemoveProfileButton, 1, wxALL, 5 );
	
	DuplicateProfileButton = new wxButton( ProfilesPanel, wxID_ANY, wxT("Duplicate"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer431->Add( DuplicateProfileButton, 0, wxALL, 5 );
	
	m_staticline26 = new wxStaticLine( ProfilesPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer431->Add( m_staticline26, 0, wxEXPAND | wxALL, 5 );
	
	ImportButton = new wxButton( ProfilesPanel, wxID_ANY, wxT("Import"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer431->Add( ImportButton, 0, wxALL, 5 );
	
	ExportButton = new wxButton( ProfilesPanel, wxID_ANY, wxT("Export"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer431->Add( ExportButton, 0, wxALL, 5 );
	
	
	bSizer57->Add( bSizer431, 0, wxALIGN_LEFT, 5 );
	
	
	ProfilesPanel->SetSizer( bSizer57 );
	ProfilesPanel->Layout();
	bSizer57->Fit( ProfilesPanel );
	CommandsPanel = new wxPanel( m_splitter5, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer44;
	bSizer44 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticline15 = new wxStaticLine( CommandsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer44->Add( m_staticline15, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer37;
	bSizer37 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer40;
	bSizer40 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText34 = new wxStaticText( CommandsPanel, wxID_ANY, wxT("Total Number of Processes : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText34->Wrap( -1 );
	bSizer40->Add( m_staticText34, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	NumberProcessesStaticText = new wxStaticText( CommandsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	NumberProcessesStaticText->Wrap( -1 );
	bSizer40->Add( NumberProcessesStaticText, 1, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer37->Add( bSizer40, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer38;
	bSizer38 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText36 = new wxStaticText( CommandsPanel, wxID_ANY, wxT("Manager Command :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText36->Wrap( -1 );
	bSizer38->Add( m_staticText36, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	ManagerTextCtrl = new wxTextCtrl( CommandsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxHSCROLL|wxTE_WORDWRAP|wxALWAYS_SHOW_SB|wxHSCROLL );
	bSizer38->Add( ManagerTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	CommandErrorStaticText = new wxStaticText( CommandsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	CommandErrorStaticText->Wrap( -1 );
	CommandErrorStaticText->SetForegroundColour( wxColour( 180, 0, 0 ) );
	
	bSizer38->Add( CommandErrorStaticText, 0, wxALL|wxEXPAND, 5 );
	
	wxFlexGridSizer* fgSizer3;
	fgSizer3 = new wxFlexGridSizer( 0, 5, 0, 0 );
	fgSizer3->AddGrowableCol( 2 );
	fgSizer3->SetFlexibleDirection( wxBOTH );
	fgSizer3->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	
	fgSizer3->Add( 0, 0, 100, wxEXPAND, 5 );
	
	m_staticText65 = new wxStaticText( CommandsPanel, wxID_ANY, wxT("Gui Address :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText65->Wrap( -1 );
	fgSizer3->Add( m_staticText65, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	GuiAddressStaticText = new wxStaticText( CommandsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	GuiAddressStaticText->Wrap( -1 );
	fgSizer3->Add( GuiAddressStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_LEFT|wxALL, 5 );
	
	GuiAutoButton = new wxButton( CommandsPanel, wxID_ANY, wxT("Auto"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer3->Add( GuiAutoButton, 0, wxALL, 5 );
	
	ControllerSpecifyButton = new wxButton( CommandsPanel, wxID_ANY, wxT("Specify"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer3->Add( ControllerSpecifyButton, 0, wxALL, 5 );
	
	
	fgSizer3->Add( 0, 0, 100, wxEXPAND, 5 );
	
	m_staticText67 = new wxStaticText( CommandsPanel, wxID_ANY, wxT("Controller Address :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText67->Wrap( -1 );
	fgSizer3->Add( m_staticText67, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	ControllerAddressStaticText = new wxStaticText( CommandsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ControllerAddressStaticText->Wrap( -1 );
	fgSizer3->Add( ControllerAddressStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_LEFT|wxALL, 5 );
	
	ControllerAutoButton = new wxButton( CommandsPanel, wxID_ANY, wxT("Auto"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer3->Add( ControllerAutoButton, 0, wxALL, 5 );
	
	m_button38 = new wxButton( CommandsPanel, wxID_ANY, wxT("Specify"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer3->Add( m_button38, 0, wxALL, 5 );
	
	
	bSizer38->Add( fgSizer3, 0, wxEXPAND, 5 );
	
	m_staticText70 = new wxStaticText( CommandsPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText70->Wrap( -1 );
	bSizer38->Add( m_staticText70, 0, wxALL, 5 );
	
	
	bSizer37->Add( bSizer38, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer43;
	bSizer43 = new wxBoxSizer( wxVERTICAL );
	
	
	bSizer37->Add( bSizer43, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer41;
	bSizer41 = new wxBoxSizer( wxVERTICAL );
	
	CommandsListBox = new wxListCtrl( CommandsPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_NO_SORT_HEADER|wxLC_REPORT|wxLC_SINGLE_SEL );
	bSizer41->Add( CommandsListBox, 1, wxALL|wxEXPAND, 5 );
	
	wxGridSizer* gSizer2;
	gSizer2 = new wxGridSizer( 0, 2, 0, 0 );
	
	wxBoxSizer* bSizer42;
	bSizer42 = new wxBoxSizer( wxHORIZONTAL );
	
	AddCommandButton = new wxButton( CommandsPanel, wxID_ANY, wxT("Add"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer42->Add( AddCommandButton, 1, wxALL, 5 );
	
	EditCommandButton = new wxButton( CommandsPanel, wxID_ANY, wxT("Edit"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer42->Add( EditCommandButton, 1, wxALL, 5 );
	
	RemoveCommandButton = new wxButton( CommandsPanel, wxID_ANY, wxT("Remove"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer42->Add( RemoveCommandButton, 1, wxALL, 5 );
	
	
	gSizer2->Add( bSizer42, 0, wxALIGN_LEFT, 5 );
	
	CommandsSaveButton = new wxButton( CommandsPanel, wxID_ANY, wxT("Save"), wxDefaultPosition, wxDefaultSize, 0 );
	gSizer2->Add( CommandsSaveButton, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	
	bSizer41->Add( gSizer2, 0, wxEXPAND, 5 );
	
	
	bSizer37->Add( bSizer41, 1, wxEXPAND, 5 );
	
	
	bSizer44->Add( bSizer37, 1, wxEXPAND, 5 );
	
	
	CommandsPanel->SetSizer( bSizer44 );
	CommandsPanel->Layout();
	bSizer44->Fit( CommandsPanel );
	m_splitter5->SplitVertically( ProfilesPanel, CommandsPanel, 349 );
	bSizer56->Add( m_splitter5, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer56 );
	this->Layout();
	
	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( RunProfilesPanel::OnUpdateIU ) );
	ProfilesListBox->Connect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( RunProfilesPanel::OnProfileDClick ), NULL, this );
	ProfilesListBox->Connect( wxEVT_LEFT_DOWN, wxMouseEventHandler( RunProfilesPanel::OnProfileLeftDown ), NULL, this );
	ProfilesListBox->Connect( wxEVT_LEFT_UP, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	ProfilesListBox->Connect( wxEVT_COMMAND_LIST_END_LABEL_EDIT, wxListEventHandler( RunProfilesPanel::OnEndProfileEdit ), NULL, this );
	ProfilesListBox->Connect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( RunProfilesPanel::OnProfilesListItemActivated ), NULL, this );
	ProfilesListBox->Connect( wxEVT_COMMAND_LIST_ITEM_FOCUSED, wxListEventHandler( RunProfilesPanel::OnProfilesFocusChange ), NULL, this );
	ProfilesListBox->Connect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	ProfilesListBox->Connect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	ProfilesListBox->Connect( wxEVT_MIDDLE_UP, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	ProfilesListBox->Connect( wxEVT_MOTION, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	ProfilesListBox->Connect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	ProfilesListBox->Connect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	ProfilesListBox->Connect( wxEVT_RIGHT_UP, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	AddProfileButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::OnAddProfileClick ), NULL, this );
	RenameProfileButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::OnRenameProfileClick ), NULL, this );
	RemoveProfileButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::OnRemoveProfileClick ), NULL, this );
	DuplicateProfileButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::OnDuplicateProfileClick ), NULL, this );
	ImportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::OnImportButtonClick ), NULL, this );
	ExportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::OnExportButtonClick ), NULL, this );
	ManagerTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( RunProfilesPanel::ManagerTextChanged ), NULL, this );
	GuiAutoButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::GuiAddressAutoClick ), NULL, this );
	ControllerSpecifyButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::GuiAddressSpecifyClick ), NULL, this );
	ControllerAutoButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::ControllerAddressAutoClick ), NULL, this );
	m_button38->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::ControllerAddressSpecifyClick ), NULL, this );
	CommandsListBox->Connect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( RunProfilesPanel::OnCommandDClick ), NULL, this );
	CommandsListBox->Connect( wxEVT_LEFT_DOWN, wxMouseEventHandler( RunProfilesPanel::OnCommandLeftDown ), NULL, this );
	CommandsListBox->Connect( wxEVT_LEFT_UP, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	CommandsListBox->Connect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( RunProfilesPanel::OnCommandsActivated ), NULL, this );
	CommandsListBox->Connect( wxEVT_COMMAND_LIST_ITEM_FOCUSED, wxListEventHandler( RunProfilesPanel::OnCommandsFocusChange ), NULL, this );
	CommandsListBox->Connect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	CommandsListBox->Connect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	CommandsListBox->Connect( wxEVT_MIDDLE_UP, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	CommandsListBox->Connect( wxEVT_MOTION, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	CommandsListBox->Connect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	CommandsListBox->Connect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	CommandsListBox->Connect( wxEVT_RIGHT_UP, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	AddCommandButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::AddCommandButtonClick ), NULL, this );
	EditCommandButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::EditCommandButtonClick ), NULL, this );
	RemoveCommandButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::RemoveCommandButtonClick ), NULL, this );
	CommandsSaveButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::CommandsSaveButtonClick ), NULL, this );
}

RunProfilesPanel::~RunProfilesPanel()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( RunProfilesPanel::OnUpdateIU ) );
	ProfilesListBox->Disconnect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( RunProfilesPanel::OnProfileDClick ), NULL, this );
	ProfilesListBox->Disconnect( wxEVT_LEFT_DOWN, wxMouseEventHandler( RunProfilesPanel::OnProfileLeftDown ), NULL, this );
	ProfilesListBox->Disconnect( wxEVT_LEFT_UP, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	ProfilesListBox->Disconnect( wxEVT_COMMAND_LIST_END_LABEL_EDIT, wxListEventHandler( RunProfilesPanel::OnEndProfileEdit ), NULL, this );
	ProfilesListBox->Disconnect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( RunProfilesPanel::OnProfilesListItemActivated ), NULL, this );
	ProfilesListBox->Disconnect( wxEVT_COMMAND_LIST_ITEM_FOCUSED, wxListEventHandler( RunProfilesPanel::OnProfilesFocusChange ), NULL, this );
	ProfilesListBox->Disconnect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	ProfilesListBox->Disconnect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	ProfilesListBox->Disconnect( wxEVT_MIDDLE_UP, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	ProfilesListBox->Disconnect( wxEVT_MOTION, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	ProfilesListBox->Disconnect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	ProfilesListBox->Disconnect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	ProfilesListBox->Disconnect( wxEVT_RIGHT_UP, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	AddProfileButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::OnAddProfileClick ), NULL, this );
	RenameProfileButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::OnRenameProfileClick ), NULL, this );
	RemoveProfileButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::OnRemoveProfileClick ), NULL, this );
	DuplicateProfileButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::OnDuplicateProfileClick ), NULL, this );
	ImportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::OnImportButtonClick ), NULL, this );
	ExportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::OnExportButtonClick ), NULL, this );
	ManagerTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( RunProfilesPanel::ManagerTextChanged ), NULL, this );
	GuiAutoButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::GuiAddressAutoClick ), NULL, this );
	ControllerSpecifyButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::GuiAddressSpecifyClick ), NULL, this );
	ControllerAutoButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::ControllerAddressAutoClick ), NULL, this );
	m_button38->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::ControllerAddressSpecifyClick ), NULL, this );
	CommandsListBox->Disconnect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( RunProfilesPanel::OnCommandDClick ), NULL, this );
	CommandsListBox->Disconnect( wxEVT_LEFT_DOWN, wxMouseEventHandler( RunProfilesPanel::OnCommandLeftDown ), NULL, this );
	CommandsListBox->Disconnect( wxEVT_LEFT_UP, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	CommandsListBox->Disconnect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( RunProfilesPanel::OnCommandsActivated ), NULL, this );
	CommandsListBox->Disconnect( wxEVT_COMMAND_LIST_ITEM_FOCUSED, wxListEventHandler( RunProfilesPanel::OnCommandsFocusChange ), NULL, this );
	CommandsListBox->Disconnect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	CommandsListBox->Disconnect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	CommandsListBox->Disconnect( wxEVT_MIDDLE_UP, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	CommandsListBox->Disconnect( wxEVT_MOTION, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	CommandsListBox->Disconnect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	CommandsListBox->Disconnect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	CommandsListBox->Disconnect( wxEVT_RIGHT_UP, wxMouseEventHandler( RunProfilesPanel::MouseVeto ), NULL, this );
	AddCommandButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::AddCommandButtonClick ), NULL, this );
	EditCommandButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::EditCommandButtonClick ), NULL, this );
	RemoveCommandButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::RemoveCommandButtonClick ), NULL, this );
	CommandsSaveButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RunProfilesPanel::CommandsSaveButtonClick ), NULL, this );
	
}

AlignMoviesPanel::AlignMoviesPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : JobPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer43;
	bSizer43 = new wxBoxSizer( wxVERTICAL );
	
	m_staticline12 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline12, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer45;
	bSizer45 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer44;
	bSizer44 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText21 = new wxStaticText( this, wxID_ANY, wxT("Input Group :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21->Wrap( -1 );
	bSizer44->Add( m_staticText21, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	GroupComboBox = new MemoryComboBox( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	bSizer44->Add( GroupComboBox, 40, wxALL, 5 );
	
	
	bSizer44->Add( 0, 0, 60, wxEXPAND, 5 );
	
	
	bSizer45->Add( bSizer44, 1, wxEXPAND, 5 );
	
	ExpertToggleButton = new wxToggleButton( this, wxID_ANY, wxT("Show Expert Options"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer45->Add( ExpertToggleButton, 0, wxALL, 5 );
	
	
	bSizer43->Add( bSizer45, 0, wxEXPAND, 5 );
	
	m_staticline10 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline10, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer46;
	bSizer46 = new wxBoxSizer( wxHORIZONTAL );
	
	ExpertPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer64;
	bSizer64 = new wxBoxSizer( wxVERTICAL );
	
	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 13, 2, 0, 0 );
	fgSizer1->SetFlexibleDirection( wxBOTH );
	fgSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticText43 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Shifts"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText43->Wrap( -1 );
	m_staticText43->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, true, wxEmptyString ) );
	
	fgSizer1->Add( m_staticText43, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	m_staticText24 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Minimum Shift (Å) : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText24->Wrap( -1 );
	fgSizer1->Add( m_staticText24, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	minimum_shift_text = new wxTextCtrl( ExpertPanel, wxID_ANY, wxT("2"), wxDefaultPosition, wxDefaultSize, 0 );
	minimum_shift_text->SetToolTip( wxT("Minimum shift for first alignment round") );
	
	fgSizer1->Add( minimum_shift_text, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText40 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Maximum Shift (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText40->Wrap( -1 );
	fgSizer1->Add( m_staticText40, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	maximum_shift_text = new wxTextCtrl( ExpertPanel, wxID_ANY, wxT("80"), wxDefaultPosition, wxDefaultSize, 0 );
	maximum_shift_text->SetToolTip( wxT("Maximum shift for each alignment round") );
	
	fgSizer1->Add( maximum_shift_text, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText44 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Exposure Filter"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText44->Wrap( -1 );
	m_staticText44->SetFont( wxFont( 10, 74, 90, 92, true, wxT("Sans") ) );
	
	fgSizer1->Add( m_staticText44, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	dose_filter_checkbox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Exposure Filter Sums?"), wxDefaultPosition, wxDefaultSize, 0 );
	dose_filter_checkbox->SetValue(true); 
	dose_filter_checkbox->SetToolTip( wxT("Make a dose weighted sum") );
	
	fgSizer1->Add( dose_filter_checkbox, 1, wxALIGN_LEFT|wxALL, 5 );
	
	restore_power_checkbox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Restore Power?"), wxDefaultPosition, wxDefaultSize, 0 );
	restore_power_checkbox->SetValue(true); 
	fgSizer1->Add( restore_power_checkbox, 1, wxALIGN_RIGHT|wxALL, 5 );
	
	m_staticText45 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Convergence"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText45->Wrap( -1 );
	m_staticText45->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, true, wxEmptyString ) );
	
	fgSizer1->Add( m_staticText45, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	m_staticText46 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Termination Threshold (Å) : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText46->Wrap( -1 );
	fgSizer1->Add( m_staticText46, 0, wxALL, 5 );
	
	termination_threshold_text = new wxTextCtrl( ExpertPanel, wxID_ANY, wxT("1"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( termination_threshold_text, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText47 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Max Iterations :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText47->Wrap( -1 );
	fgSizer1->Add( m_staticText47, 0, wxALL, 5 );
	
	max_iterations_spinctrl = new wxSpinCtrl( ExpertPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 50, 20 );
	fgSizer1->Add( max_iterations_spinctrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText48 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Filter"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText48->Wrap( -1 );
	m_staticText48->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, true, wxEmptyString ) );
	
	fgSizer1->Add( m_staticText48, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	m_staticText49 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("B-Factor (Å²) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText49->Wrap( -1 );
	fgSizer1->Add( m_staticText49, 0, wxALL, 5 );
	
	bfactor_spinctrl = new wxSpinCtrl( ExpertPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 0, 5000, 1500 );
	fgSizer1->Add( bfactor_spinctrl, 0, wxALL|wxEXPAND, 5 );
	
	mask_central_cross_checkbox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Mask Central Cross?"), wxDefaultPosition, wxDefaultSize, 0 );
	mask_central_cross_checkbox->SetValue(true); 
	fgSizer1->Add( mask_central_cross_checkbox, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	m_staticText50 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Horizontal Mask (Pixels) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText50->Wrap( -1 );
	fgSizer1->Add( m_staticText50, 0, wxALL, 5 );
	
	horizontal_mask_spinctrl = new wxSpinCtrl( ExpertPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 0, 10, 1 );
	fgSizer1->Add( horizontal_mask_spinctrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText51 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Vertical Mask (Pixels) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText51->Wrap( -1 );
	fgSizer1->Add( m_staticText51, 0, wxALL, 5 );
	
	vertical_mask_spinctrl = new wxSpinCtrl( ExpertPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 0, 10, 1 );
	fgSizer1->Add( vertical_mask_spinctrl, 0, wxALL|wxEXPAND, 5 );
	
	
	bSizer64->Add( fgSizer1, 0, wxEXPAND, 5 );
	
	
	ExpertPanel->SetSizer( bSizer64 );
	ExpertPanel->Layout();
	bSizer64->Fit( ExpertPanel );
	bSizer46->Add( ExpertPanel, 0, wxEXPAND | wxALL, 5 );
	
	OutputTextPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	OutputTextPanel->Hide();
	
	wxBoxSizer* bSizer56;
	bSizer56 = new wxBoxSizer( wxVERTICAL );
	
	output_textctrl = new wxTextCtrl( OutputTextPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE|wxTE_READONLY );
	bSizer56->Add( output_textctrl, 1, wxALL|wxEXPAND, 5 );
	
	
	OutputTextPanel->SetSizer( bSizer56 );
	OutputTextPanel->Layout();
	bSizer56->Fit( OutputTextPanel );
	bSizer46->Add( OutputTextPanel, 30, wxEXPAND | wxALL, 5 );
	
	GraphPanel = new UnblurResultsPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	GraphPanel->Hide();
	
	bSizer46->Add( GraphPanel, 50, wxEXPAND | wxALL, 5 );
	
	InfoPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer61;
	bSizer61 = new wxBoxSizer( wxVERTICAL );
	
	InfoText = new wxRichTextCtrl( InfoPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_READONLY|wxHSCROLL|wxVSCROLL );
	bSizer61->Add( InfoText, 1, wxEXPAND | wxALL, 5 );
	
	
	InfoPanel->SetSizer( bSizer61 );
	InfoPanel->Layout();
	bSizer61->Fit( InfoPanel );
	bSizer46->Add( InfoPanel, 1, wxEXPAND | wxALL, 5 );
	
	
	bSizer43->Add( bSizer46, 1, wxEXPAND, 5 );
	
	m_staticline11 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline11, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer48;
	bSizer48 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer70;
	bSizer70 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer71;
	bSizer71 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer55;
	bSizer55 = new wxBoxSizer( wxHORIZONTAL );
	
	
	bSizer71->Add( bSizer55, 0, wxALIGN_CENTER_HORIZONTAL, 5 );
	
	
	bSizer70->Add( bSizer71, 1, wxALIGN_CENTER_VERTICAL, 5 );
	
	wxBoxSizer* bSizer74;
	bSizer74 = new wxBoxSizer( wxVERTICAL );
	
	
	bSizer70->Add( bSizer74, 0, wxALIGN_CENTER_VERTICAL, 5 );
	
	ProgressPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ProgressPanel->Hide();
	
	wxBoxSizer* bSizer57;
	bSizer57 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer59;
	bSizer59 = new wxBoxSizer( wxHORIZONTAL );
	
	NumberConnectedText = new wxStaticText( ProgressPanel, wxID_ANY, wxT("o"), wxDefaultPosition, wxDefaultSize, 0 );
	NumberConnectedText->Wrap( -1 );
	bSizer59->Add( NumberConnectedText, 0, wxALIGN_CENTER|wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	ProgressBar = new wxGauge( ProgressPanel, wxID_ANY, 100, wxDefaultPosition, wxDefaultSize, wxGA_HORIZONTAL );
	ProgressBar->SetValue( 0 ); 
	bSizer59->Add( ProgressBar, 100, wxALIGN_CENTER|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );
	
	TimeRemainingText = new wxStaticText( ProgressPanel, wxID_ANY, wxT("Time Remaining : ???h:??m:??s   "), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	TimeRemainingText->Wrap( -1 );
	bSizer59->Add( TimeRemainingText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	m_staticline60 = new wxStaticLine( ProgressPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer59->Add( m_staticline60, 0, wxEXPAND | wxALL, 5 );
	
	FinishButton = new wxButton( ProgressPanel, wxID_ANY, wxT("Finish"), wxDefaultPosition, wxDefaultSize, 0 );
	FinishButton->Hide();
	
	bSizer59->Add( FinishButton, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	CancelAlignmentButton = new wxButton( ProgressPanel, wxID_ANY, wxT("Terminate Job"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer59->Add( CancelAlignmentButton, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer57->Add( bSizer59, 1, wxEXPAND, 5 );
	
	
	ProgressPanel->SetSizer( bSizer57 );
	ProgressPanel->Layout();
	bSizer57->Fit( ProgressPanel );
	bSizer70->Add( ProgressPanel, 1, wxEXPAND | wxALL, 5 );
	
	
	bSizer48->Add( bSizer70, 1, wxEXPAND, 5 );
	
	StartPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer58;
	bSizer58 = new wxBoxSizer( wxHORIZONTAL );
	
	RunProfileText = new wxStaticText( StartPanel, wxID_ANY, wxT("Run Profile :"), wxDefaultPosition, wxDefaultSize, 0 );
	RunProfileText->Wrap( -1 );
	bSizer58->Add( RunProfileText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	RunProfileComboBox = new MemoryComboBox( StartPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	bSizer58->Add( RunProfileComboBox, 50, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer60;
	bSizer60 = new wxBoxSizer( wxVERTICAL );
	
	StartAlignmentButton = new wxButton( StartPanel, wxID_ANY, wxT("Start Alignment"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer60->Add( StartAlignmentButton, 0, wxALL, 5 );
	
	
	bSizer58->Add( bSizer60, 50, wxEXPAND, 5 );
	
	
	StartPanel->SetSizer( bSizer58 );
	StartPanel->Layout();
	bSizer58->Fit( StartPanel );
	bSizer48->Add( StartPanel, 1, wxEXPAND | wxALL, 5 );
	
	
	bSizer43->Add( bSizer48, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer43 );
	this->Layout();
	
	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( AlignMoviesPanel::OnUpdateUI ) );
	ExpertToggleButton->Connect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( AlignMoviesPanel::OnExpertOptionsToggle ), NULL, this );
	InfoText->Connect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( AlignMoviesPanel::OnInfoURL ), NULL, this );
	FinishButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AlignMoviesPanel::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AlignMoviesPanel::TerminateButtonClick ), NULL, this );
	StartAlignmentButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AlignMoviesPanel::StartAlignmentClick ), NULL, this );
}

AlignMoviesPanel::~AlignMoviesPanel()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( AlignMoviesPanel::OnUpdateUI ) );
	ExpertToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( AlignMoviesPanel::OnExpertOptionsToggle ), NULL, this );
	InfoText->Disconnect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( AlignMoviesPanel::OnInfoURL ), NULL, this );
	FinishButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AlignMoviesPanel::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AlignMoviesPanel::TerminateButtonClick ), NULL, this );
	StartAlignmentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AlignMoviesPanel::StartAlignmentClick ), NULL, this );
	
}

Refine3DPanel::Refine3DPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : JobPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer43;
	bSizer43 = new wxBoxSizer( wxVERTICAL );
	
	m_staticline12 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline12, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer451;
	bSizer451 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer441;
	bSizer441 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer200;
	bSizer200 = new wxBoxSizer( wxHORIZONTAL );
	
	wxFlexGridSizer* fgSizer15;
	fgSizer15 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer15->SetFlexibleDirection( wxBOTH );
	fgSizer15->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticText262 = new wxStaticText( this, wxID_ANY, wxT("Input Refinement Package :"), wxDefaultPosition, wxSize( -1,-1 ), 0 );
	m_staticText262->Wrap( -1 );
	fgSizer15->Add( m_staticText262, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	RefinementPackageComboBox = new MemoryComboBox( this, wxID_ANY, wxT("Combo!"), wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	RefinementPackageComboBox->SetMinSize( wxSize( 250,-1 ) );
	
	fgSizer15->Add( RefinementPackageComboBox, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText263 = new wxStaticText( this, wxID_ANY, wxT("Input Parameters :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText263->Wrap( -1 );
	fgSizer15->Add( m_staticText263, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	InputParametersComboBox = new MemoryComboBox( this, wxID_ANY, wxT("Combo!"), wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	fgSizer15->Add( InputParametersComboBox, 0, wxALL|wxEXPAND, 5 );
	
	
	bSizer200->Add( fgSizer15, 1, wxEXPAND, 5 );
	
	m_staticline52 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer200->Add( m_staticline52, 0, wxEXPAND | wxALL, 5 );
	
	wxGridSizer* gSizer10;
	gSizer10 = new wxGridSizer( 0, 1, 0, 0 );
	
	wxBoxSizer* bSizer215;
	bSizer215 = new wxBoxSizer( wxHORIZONTAL );
	
	LocalRefinementRadio = new wxRadioButton( this, wxID_ANY, wxT("Local Refinement"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer215->Add( LocalRefinementRadio, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	
	bSizer215->Add( 0, 0, 1, wxEXPAND, 5 );
	
	GlobalRefinementRadio = new wxRadioButton( this, wxID_ANY, wxT("Global Search"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer215->Add( GlobalRefinementRadio, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	
	gSizer10->Add( bSizer215, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer216;
	bSizer216 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText264 = new wxStaticText( this, wxID_ANY, wxT("No. of Cycles to Run :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText264->Wrap( -1 );
	bSizer216->Add( m_staticText264, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	NumberRoundsSpinCtrl = new wxSpinCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 100, 1 );
	NumberRoundsSpinCtrl->SetMinSize( wxSize( 100,-1 ) );
	
	bSizer216->Add( NumberRoundsSpinCtrl, 1, wxALL|wxEXPAND, 5 );
	
	
	gSizer10->Add( bSizer216, 1, wxEXPAND, 5 );
	
	
	bSizer200->Add( gSizer10, 0, wxEXPAND, 5 );
	
	m_staticline54 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer200->Add( m_staticline54, 0, wxEXPAND | wxALL, 5 );
	
	wxGridSizer* gSizer11;
	gSizer11 = new wxGridSizer( 2, 1, 0, 0 );
	
	wxBoxSizer* bSizer214;
	bSizer214 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText188 = new wxStaticText( this, wxID_ANY, wxT("High-Resolution Limit (Å) : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText188->Wrap( -1 );
	bSizer214->Add( m_staticText188, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	HighResolutionLimitTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("30.00"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer214->Add( HighResolutionLimitTextCtrl, 1, wxALL|wxEXPAND, 5 );
	
	
	gSizer11->Add( bSizer214, 1, wxEXPAND, 5 );
	
	
	bSizer200->Add( gSizer11, 0, wxEXPAND, 5 );
	
	
	bSizer200->Add( 0, 0, 1, wxEXPAND, 5 );
	
	ExpertToggleButton = new wxToggleButton( this, wxID_ANY, wxT("Show Expert Options"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer200->Add( ExpertToggleButton, 0, wxALIGN_BOTTOM|wxALL, 5 );
	
	
	bSizer441->Add( bSizer200, 1, wxEXPAND, 5 );
	
	
	bSizer451->Add( bSizer441, 1, wxEXPAND, 5 );
	
	
	bSizer43->Add( bSizer451, 0, wxEXPAND, 5 );
	
	m_staticline10 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline10, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer46;
	bSizer46 = new wxBoxSizer( wxHORIZONTAL );
	
	ExpertPanel = new wxScrolledWindow( this, wxID_ANY, wxDefaultPosition, wxSize( -1,-1 ), wxVSCROLL );
	ExpertPanel->SetScrollRate( 5, 5 );
	ExpertPanel->Hide();
	
	InputSizer = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer258;
	bSizer258 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText318 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Parameters To Refine"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText318->Wrap( -1 );
	m_staticText318->SetFont( wxFont( 10, 74, 90, 92, true, wxT("Sans") ) );
	
	bSizer258->Add( m_staticText318, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer258->Add( 0, 0, 1, wxEXPAND, 5 );
	
	ResetAllDefaultsButton = new wxButton( ExpertPanel, wxID_ANY, wxT("Reset All Defaults"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer258->Add( ResetAllDefaultsButton, 0, wxALL, 5 );
	
	
	bSizer258->Add( 5, 0, 0, wxEXPAND, 5 );
	
	
	InputSizer->Add( bSizer258, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer257;
	bSizer257 = new wxBoxSizer( wxHORIZONTAL );
	
	RefinePsiCheckBox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Psi"), wxDefaultPosition, wxDefaultSize, 0 );
	RefinePsiCheckBox->SetValue(true); 
	bSizer257->Add( RefinePsiCheckBox, 0, wxALL, 5 );
	
	RefineThetaCheckBox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Theta"), wxDefaultPosition, wxDefaultSize, 0 );
	RefineThetaCheckBox->SetValue(true); 
	bSizer257->Add( RefineThetaCheckBox, 0, wxALL, 5 );
	
	RefinePhiCheckBox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Phi"), wxDefaultPosition, wxDefaultSize, 0 );
	RefinePhiCheckBox->SetValue(true); 
	bSizer257->Add( RefinePhiCheckBox, 0, wxALL, 5 );
	
	RefineXShiftCheckBox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("X Shift"), wxDefaultPosition, wxDefaultSize, 0 );
	RefineXShiftCheckBox->SetValue(true); 
	bSizer257->Add( RefineXShiftCheckBox, 0, wxALL, 5 );
	
	RefineYShiftCheckBox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Y Shift"), wxDefaultPosition, wxDefaultSize, 0 );
	RefineYShiftCheckBox->SetValue(true); 
	bSizer257->Add( RefineYShiftCheckBox, 0, wxALL, 5 );
	
	
	InputSizer->Add( bSizer257, 0, wxEXPAND, 5 );
	
	
	InputSizer->Add( 0, 5, 0, wxEXPAND, 5 );
	
	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer1->SetFlexibleDirection( wxBOTH );
	fgSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticText202 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("General Refinement"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText202->Wrap( -1 );
	m_staticText202->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, true, wxEmptyString ) );
	
	fgSizer1->Add( m_staticText202, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	NoMovieFramesStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Low-Resolution Limit (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	NoMovieFramesStaticText->Wrap( -1 );
	fgSizer1->Add( NoMovieFramesStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	LowResolutionLimitTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("300.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( LowResolutionLimitTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText196 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Mask Radius (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText196->Wrap( -1 );
	fgSizer1->Add( m_staticText196, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	MaskRadiusTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( MaskRadiusTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText317 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Signed CC Resolution Limit (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText317->Wrap( -1 );
	fgSizer1->Add( m_staticText317, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	SignedCCResolutionTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( SignedCCResolutionTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText362 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Percent Used (%) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText362->Wrap( -1 );
	fgSizer1->Add( m_staticText362, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	PercentUsedTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( PercentUsedTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText201 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Global Search"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText201->Wrap( -1 );
	m_staticText201->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, true, wxEmptyString ) );
	
	fgSizer1->Add( m_staticText201, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	GlobalMaskRadiusStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Global Mask Radius (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	GlobalMaskRadiusStaticText->Wrap( -1 );
	fgSizer1->Add( GlobalMaskRadiusStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	GlobalMaskRadiusTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("30.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( GlobalMaskRadiusTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	NumberToRefineStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Number of Results to Refine :"), wxDefaultPosition, wxDefaultSize, 0 );
	NumberToRefineStaticText->Wrap( -1 );
	fgSizer1->Add( NumberToRefineStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	NumberToRefineSpinCtrl = new wxSpinCtrl( ExpertPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 0, 1000, 21 );
	fgSizer1->Add( NumberToRefineSpinCtrl, 0, wxALL, 5 );
	
	AngularStepStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Angular Search Step (°) :"), wxDefaultPosition, wxDefaultSize, 0 );
	AngularStepStaticText->Wrap( -1 );
	fgSizer1->Add( AngularStepStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	AngularStepTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("20.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( AngularStepTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	SearchRangeXStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Search Range in X (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	SearchRangeXStaticText->Wrap( -1 );
	fgSizer1->Add( SearchRangeXStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	SearchRangeXTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( SearchRangeXTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	SearchRangeYStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Search Range in Y (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	SearchRangeYStaticText->Wrap( -1 );
	fgSizer1->Add( SearchRangeYStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	SearchRangeYTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( SearchRangeYTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText200 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Classification"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText200->Wrap( -1 );
	m_staticText200->SetFont( wxFont( 10, 74, 90, 92, true, wxT("Sans") ) );
	
	fgSizer1->Add( m_staticText200, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	MinPhaseShiftStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("High-Resolution Limit (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	MinPhaseShiftStaticText->Wrap( -1 );
	fgSizer1->Add( MinPhaseShiftStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	ClassificationHighResLimitTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("10.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( ClassificationHighResLimitTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	PhaseShiftStepStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Focused Classification?"), wxDefaultPosition, wxDefaultSize, 0 );
	PhaseShiftStepStaticText->Wrap( -1 );
	fgSizer1->Add( PhaseShiftStepStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	wxBoxSizer* bSizer263;
	bSizer263 = new wxBoxSizer( wxHORIZONTAL );
	
	SphereClassificatonYesRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer263->Add( SphereClassificatonYesRadio, 0, wxALL, 5 );
	
	SphereClassificatonNoRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer263->Add( SphereClassificatonNoRadio, 0, wxALL, 5 );
	
	
	fgSizer1->Add( bSizer263, 1, wxEXPAND, 5 );
	
	SphereXStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tSphere X Co-ordinate (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	SphereXStaticText->Wrap( -1 );
	SphereXStaticText->Enable( false );
	
	fgSizer1->Add( SphereXStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	SphereXTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.00"), wxDefaultPosition, wxDefaultSize, 0 );
	SphereXTextCtrl->Enable( false );
	
	fgSizer1->Add( SphereXTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	SphereYStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tSphere Y Co-ordinate (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	SphereYStaticText->Wrap( -1 );
	SphereYStaticText->Enable( false );
	
	fgSizer1->Add( SphereYStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	SphereYTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.00"), wxDefaultPosition, wxDefaultSize, 0 );
	SphereYTextCtrl->Enable( false );
	
	fgSizer1->Add( SphereYTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	SphereZStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tSphere Z Co-ordinate (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	SphereZStaticText->Wrap( -1 );
	SphereZStaticText->Enable( false );
	
	fgSizer1->Add( SphereZStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_LEFT|wxALL, 5 );
	
	SphereZTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.00"), wxDefaultPosition, wxDefaultSize, 0 );
	SphereZTextCtrl->Enable( false );
	
	fgSizer1->Add( SphereZTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	SphereRadiusStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tSphere Radius (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	SphereRadiusStaticText->Wrap( -1 );
	SphereRadiusStaticText->Enable( false );
	
	fgSizer1->Add( SphereRadiusStaticText, 0, wxALIGN_LEFT|wxALL, 5 );
	
	SphereRadiusTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.00"), wxDefaultPosition, wxDefaultSize, 0 );
	SphereRadiusTextCtrl->Enable( false );
	
	fgSizer1->Add( SphereRadiusTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText323 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("CTF Refinement"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText323->Wrap( -1 );
	m_staticText323->SetFont( wxFont( 10, 74, 90, 92, true, wxT("Sans") ) );
	
	fgSizer1->Add( m_staticText323, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	m_staticText324 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Refine CTF?"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText324->Wrap( -1 );
	fgSizer1->Add( m_staticText324, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	wxBoxSizer* bSizer264;
	bSizer264 = new wxBoxSizer( wxHORIZONTAL );
	
	RefineCTFYesRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer264->Add( RefineCTFYesRadio, 0, wxALL, 5 );
	
	RefineCTFNoRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer264->Add( RefineCTFNoRadio, 0, wxALL, 5 );
	
	
	fgSizer1->Add( bSizer264, 1, wxEXPAND, 5 );
	
	DefocusSearchRangeStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tDefocus Search Range (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	DefocusSearchRangeStaticText->Wrap( -1 );
	DefocusSearchRangeStaticText->Enable( false );
	
	fgSizer1->Add( DefocusSearchRangeStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	DefocusSearchRangeTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("500.00"), wxDefaultPosition, wxDefaultSize, 0 );
	DefocusSearchRangeTextCtrl->Enable( false );
	
	fgSizer1->Add( DefocusSearchRangeTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	DefocusSearchStepStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tDefocus Search Step (Å) : "), wxDefaultPosition, wxDefaultSize, 0 );
	DefocusSearchStepStaticText->Wrap( -1 );
	DefocusSearchStepStaticText->Enable( false );
	
	fgSizer1->Add( DefocusSearchStepStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	DefocusSearchStepTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, 0 );
	DefocusSearchStepTextCtrl->Enable( false );
	
	fgSizer1->Add( DefocusSearchStepTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText329 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Reconstruction"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText329->Wrap( -1 );
	m_staticText329->SetFont( wxFont( 10, 74, 90, 92, true, wxT("Sans") ) );
	
	fgSizer1->Add( m_staticText329, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	m_staticText330 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Inner Mask Radius (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText330->Wrap( -1 );
	fgSizer1->Add( m_staticText330, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	ReconstructionInnerRadiusTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( ReconstructionInnerRadiusTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText331 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Outer Mask Radius (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText331->Wrap( -1 );
	fgSizer1->Add( m_staticText331, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	ReconstructionOuterRadiusTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( ReconstructionOuterRadiusTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText332 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Score to Weight Constant (Å²) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText332->Wrap( -1 );
	fgSizer1->Add( m_staticText332, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	ScoreToWeightConstantTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("5.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( ScoreToWeightConstantTextCtrl, 0, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );
	
	m_staticText335 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Adjust Score for Defocus?"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText335->Wrap( -1 );
	fgSizer1->Add( m_staticText335, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	wxBoxSizer* bSizer265;
	bSizer265 = new wxBoxSizer( wxHORIZONTAL );
	
	AdjustScoreForDefocusYesRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer265->Add( AdjustScoreForDefocusYesRadio, 0, wxALL, 5 );
	
	AdjustScoreForDefocusNoRadio = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer265->Add( AdjustScoreForDefocusNoRadio, 0, wxALL, 5 );
	
	
	fgSizer1->Add( bSizer265, 1, wxEXPAND, 5 );
	
	m_staticText333 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Score Threshold :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText333->Wrap( -1 );
	fgSizer1->Add( m_staticText333, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	ReconstructioScoreThreshold = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("-1.0"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( ReconstructioScoreThreshold, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText334 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Resolution Limit (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText334->Wrap( -1 );
	fgSizer1->Add( m_staticText334, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	ReconstructionResolutionLimitTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( ReconstructionResolutionLimitTextCtrl, 0, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );
	
	m_staticText336 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Autocrop Images?"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText336->Wrap( -1 );
	fgSizer1->Add( m_staticText336, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	wxBoxSizer* bSizer266;
	bSizer266 = new wxBoxSizer( wxHORIZONTAL );
	
	AutoCropYesRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer266->Add( AutoCropYesRadioButton, 0, wxALL, 5 );
	
	AutoCropNoRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer266->Add( AutoCropNoRadioButton, 0, wxALL, 5 );
	
	
	fgSizer1->Add( bSizer266, 1, wxEXPAND, 5 );
	
	m_staticText363 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Apply Likelihood Blurring?"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText363->Wrap( -1 );
	fgSizer1->Add( m_staticText363, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	wxBoxSizer* bSizer2661;
	bSizer2661 = new wxBoxSizer( wxHORIZONTAL );
	
	ApplyBlurringYesRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Yes"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
	bSizer2661->Add( ApplyBlurringYesRadioButton, 0, wxALL, 5 );
	
	ApplyBlurringNoRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("No"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer2661->Add( ApplyBlurringNoRadioButton, 0, wxALL, 5 );
	
	
	fgSizer1->Add( bSizer2661, 1, wxEXPAND, 5 );
	
	m_staticText364 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("\tSmoothing Factor :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText364->Wrap( -1 );
	fgSizer1->Add( m_staticText364, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	SmoothingFactorTextCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("1.00"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( SmoothingFactorTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	
	InputSizer->Add( fgSizer1, 1, wxEXPAND, 5 );
	
	
	ExpertPanel->SetSizer( InputSizer );
	ExpertPanel->Layout();
	InputSizer->Fit( ExpertPanel );
	bSizer46->Add( ExpertPanel, 0, wxALL|wxEXPAND, 5 );
	
	OutputTextPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	OutputTextPanel->Hide();
	
	wxBoxSizer* bSizer56;
	bSizer56 = new wxBoxSizer( wxVERTICAL );
	
	output_textctrl = new wxTextCtrl( OutputTextPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE|wxTE_READONLY );
	bSizer56->Add( output_textctrl, 1, wxALL|wxEXPAND, 5 );
	
	
	OutputTextPanel->SetSizer( bSizer56 );
	OutputTextPanel->Layout();
	bSizer56->Fit( OutputTextPanel );
	bSizer46->Add( OutputTextPanel, 30, wxEXPAND | wxALL, 5 );
	
	InfoPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer61;
	bSizer61 = new wxBoxSizer( wxVERTICAL );
	
	InfoText = new wxRichTextCtrl( InfoPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_READONLY|wxHSCROLL|wxVSCROLL );
	bSizer61->Add( InfoText, 1, wxEXPAND | wxALL, 5 );
	
	
	InfoPanel->SetSizer( bSizer61 );
	InfoPanel->Layout();
	bSizer61->Fit( InfoPanel );
	bSizer46->Add( InfoPanel, 1, wxEXPAND | wxALL, 5 );
	
	FSCResultsPanel = new MyFSCPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	FSCResultsPanel->Hide();
	
	bSizer46->Add( FSCResultsPanel, 50, wxEXPAND | wxALL, 5 );
	
	AngularPlotPanel = new AngularDistributionPlotPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	AngularPlotPanel->Hide();
	
	bSizer46->Add( AngularPlotPanel, 50, wxEXPAND | wxALL, 5 );
	
	
	bSizer43->Add( bSizer46, 1, wxEXPAND, 5 );
	
	m_staticline11 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline11, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer48;
	bSizer48 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer70;
	bSizer70 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer71;
	bSizer71 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer55;
	bSizer55 = new wxBoxSizer( wxHORIZONTAL );
	
	
	bSizer71->Add( bSizer55, 0, wxALIGN_CENTER_HORIZONTAL, 5 );
	
	
	bSizer70->Add( bSizer71, 1, wxALIGN_CENTER_VERTICAL, 5 );
	
	wxBoxSizer* bSizer74;
	bSizer74 = new wxBoxSizer( wxVERTICAL );
	
	
	bSizer70->Add( bSizer74, 0, wxALIGN_CENTER_VERTICAL, 5 );
	
	ProgressPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ProgressPanel->Hide();
	
	wxBoxSizer* bSizer57;
	bSizer57 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer59;
	bSizer59 = new wxBoxSizer( wxHORIZONTAL );
	
	NumberConnectedText = new wxStaticText( ProgressPanel, wxID_ANY, wxT("o"), wxDefaultPosition, wxDefaultSize, 0 );
	NumberConnectedText->Wrap( -1 );
	bSizer59->Add( NumberConnectedText, 0, wxALIGN_CENTER|wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	ProgressBar = new wxGauge( ProgressPanel, wxID_ANY, 100, wxDefaultPosition, wxDefaultSize, wxGA_HORIZONTAL );
	ProgressBar->SetValue( 0 ); 
	bSizer59->Add( ProgressBar, 100, wxALIGN_CENTER|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );
	
	TimeRemainingText = new wxStaticText( ProgressPanel, wxID_ANY, wxT("Time Remaining : ???h:??m:??s   "), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	TimeRemainingText->Wrap( -1 );
	bSizer59->Add( TimeRemainingText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	m_staticline60 = new wxStaticLine( ProgressPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer59->Add( m_staticline60, 0, wxEXPAND | wxALL, 5 );
	
	FinishButton = new wxButton( ProgressPanel, wxID_ANY, wxT("Finish"), wxDefaultPosition, wxDefaultSize, 0 );
	FinishButton->Hide();
	
	bSizer59->Add( FinishButton, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	CancelAlignmentButton = new wxButton( ProgressPanel, wxID_ANY, wxT("Terminate Job"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer59->Add( CancelAlignmentButton, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer57->Add( bSizer59, 1, wxEXPAND, 5 );
	
	
	ProgressPanel->SetSizer( bSizer57 );
	ProgressPanel->Layout();
	bSizer57->Fit( ProgressPanel );
	bSizer70->Add( ProgressPanel, 1, wxEXPAND | wxALL, 5 );
	
	
	bSizer48->Add( bSizer70, 1, wxEXPAND, 5 );
	
	StartPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer58;
	bSizer58 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer268;
	bSizer268 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer267;
	bSizer267 = new wxBoxSizer( wxHORIZONTAL );
	
	RunProfileText = new wxStaticText( StartPanel, wxID_ANY, wxT("Refinement Run Profile :"), wxDefaultPosition, wxDefaultSize, 0 );
	RunProfileText->Wrap( -1 );
	bSizer267->Add( RunProfileText, 20, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	RefinementRunProfileComboBox = new MemoryComboBox( StartPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	bSizer267->Add( RefinementRunProfileComboBox, 50, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );
	
	
	bSizer268->Add( bSizer267, 50, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer2671;
	bSizer2671 = new wxBoxSizer( wxHORIZONTAL );
	
	RunProfileText1 = new wxStaticText( StartPanel, wxID_ANY, wxT("Reconstruction Run Profile :"), wxDefaultPosition, wxDefaultSize, 0 );
	RunProfileText1->Wrap( -1 );
	bSizer2671->Add( RunProfileText1, 20, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	ReconstructionRunProfileComboBox = new MemoryComboBox( StartPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	bSizer2671->Add( ReconstructionRunProfileComboBox, 50, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );
	
	
	bSizer268->Add( bSizer2671, 0, wxEXPAND, 5 );
	
	
	bSizer58->Add( bSizer268, 50, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer60;
	bSizer60 = new wxBoxSizer( wxVERTICAL );
	
	
	bSizer60->Add( 0, 0, 1, wxEXPAND, 5 );
	
	StartRefinementButton = new wxButton( StartPanel, wxID_ANY, wxT("Start Refinement"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer60->Add( StartRefinementButton, 0, wxALL, 5 );
	
	
	bSizer58->Add( bSizer60, 50, wxEXPAND, 5 );
	
	
	StartPanel->SetSizer( bSizer58 );
	StartPanel->Layout();
	bSizer58->Fit( StartPanel );
	bSizer48->Add( StartPanel, 1, wxEXPAND | wxALL, 5 );
	
	
	bSizer43->Add( bSizer48, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer43 );
	this->Layout();
	
	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( Refine3DPanel::OnUpdateUI ) );
	RefinementPackageComboBox->Connect( wxEVT_COMMAND_COMBOBOX_SELECTED, wxCommandEventHandler( Refine3DPanel::OnRefinementPackageComboBox ), NULL, this );
	InputParametersComboBox->Connect( wxEVT_COMMAND_COMBOBOX_SELECTED, wxCommandEventHandler( Refine3DPanel::OnInputParametersComboBox ), NULL, this );
	HighResolutionLimitTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( Refine3DPanel::OnHighResLimitChange ), NULL, this );
	ExpertToggleButton->Connect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( Refine3DPanel::OnExpertOptionsToggle ), NULL, this );
	ResetAllDefaultsButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine3DPanel::ResetAllDefaultsClick ), NULL, this );
	InfoText->Connect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( Refine3DPanel::OnInfoURL ), NULL, this );
	FinishButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine3DPanel::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine3DPanel::TerminateButtonClick ), NULL, this );
	StartRefinementButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine3DPanel::StartRefinementClick ), NULL, this );
}

Refine3DPanel::~Refine3DPanel()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( Refine3DPanel::OnUpdateUI ) );
	RefinementPackageComboBox->Disconnect( wxEVT_COMMAND_COMBOBOX_SELECTED, wxCommandEventHandler( Refine3DPanel::OnRefinementPackageComboBox ), NULL, this );
	InputParametersComboBox->Disconnect( wxEVT_COMMAND_COMBOBOX_SELECTED, wxCommandEventHandler( Refine3DPanel::OnInputParametersComboBox ), NULL, this );
	HighResolutionLimitTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( Refine3DPanel::OnHighResLimitChange ), NULL, this );
	ExpertToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( Refine3DPanel::OnExpertOptionsToggle ), NULL, this );
	ResetAllDefaultsButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine3DPanel::ResetAllDefaultsClick ), NULL, this );
	InfoText->Disconnect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( Refine3DPanel::OnInfoURL ), NULL, this );
	FinishButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine3DPanel::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine3DPanel::TerminateButtonClick ), NULL, this );
	StartRefinementButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( Refine3DPanel::StartRefinementClick ), NULL, this );
	
}

FindCTFPanel::FindCTFPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : JobPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer43;
	bSizer43 = new wxBoxSizer( wxVERTICAL );
	
	m_staticline12 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline12, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer45;
	bSizer45 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer44;
	bSizer44 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText21 = new wxStaticText( this, wxID_ANY, wxT("Input Group :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21->Wrap( -1 );
	bSizer44->Add( m_staticText21, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	GroupComboBox = new MemoryComboBox( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	bSizer44->Add( GroupComboBox, 40, wxALL, 5 );
	
	
	bSizer44->Add( 0, 0, 60, wxEXPAND, 5 );
	
	
	bSizer45->Add( bSizer44, 1, wxEXPAND, 5 );
	
	ExpertToggleButton = new wxToggleButton( this, wxID_ANY, wxT("Show Expert Options"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer45->Add( ExpertToggleButton, 0, wxALL, 5 );
	
	
	bSizer43->Add( bSizer45, 0, wxEXPAND, 5 );
	
	m_staticline10 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline10, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer46;
	bSizer46 = new wxBoxSizer( wxHORIZONTAL );
	
	ExpertPanel = new wxScrolledWindow( this, wxID_ANY, wxDefaultPosition, wxSize( -1,-1 ), wxVSCROLL );
	ExpertPanel->SetScrollRate( 5, 5 );
	ExpertPanel->Hide();
	
	InputSizer = new wxBoxSizer( wxVERTICAL );
	
	wxFlexGridSizer* fgSizer1;
	fgSizer1 = new wxFlexGridSizer( 0, 2, 0, 0 );
	fgSizer1->SetFlexibleDirection( wxBOTH );
	fgSizer1->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticText202 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Search"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText202->Wrap( -1 );
	m_staticText202->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, true, wxEmptyString ) );
	
	fgSizer1->Add( m_staticText202, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	m_staticText186 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Estimate Using :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText186->Wrap( -1 );
	fgSizer1->Add( m_staticText186, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	wxBoxSizer* bSizer123;
	bSizer123 = new wxBoxSizer( wxHORIZONTAL );
	
	MovieRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Movies"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer123->Add( MovieRadioButton, 0, wxALL, 5 );
	
	ImageRadioButton = new wxRadioButton( ExpertPanel, wxID_ANY, wxT("Images"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer123->Add( ImageRadioButton, 0, wxALL, 5 );
	
	
	fgSizer1->Add( bSizer123, 1, wxEXPAND, 5 );
	
	NoMovieFramesStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("No. Movie Frames to Average :"), wxDefaultPosition, wxDefaultSize, 0 );
	NoMovieFramesStaticText->Wrap( -1 );
	fgSizer1->Add( NoMovieFramesStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	NoFramesToAverageSpinCtrl = new wxSpinCtrl( ExpertPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 999999999, 3 );
	fgSizer1->Add( NoFramesToAverageSpinCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText188 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Box Size (px) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText188->Wrap( -1 );
	fgSizer1->Add( m_staticText188, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	BoxSizeSpinCtrl = new wxSpinCtrl( ExpertPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 128, 99999999, 512 );
	fgSizer1->Add( BoxSizeSpinCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText196 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Amplitude Contrast :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText196->Wrap( -1 );
	fgSizer1->Add( m_staticText196, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	AmplitudeContrastNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.07"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( AmplitudeContrastNumericCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText201 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Search Limits"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText201->Wrap( -1 );
	m_staticText201->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, true, wxEmptyString ) );
	
	fgSizer1->Add( m_staticText201, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	m_staticText189 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Min. Resolution of Fit (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText189->Wrap( -1 );
	fgSizer1->Add( m_staticText189, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	MinResNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("30.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( MinResNumericCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText190 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Max. Resolution of Fit (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText190->Wrap( -1 );
	fgSizer1->Add( m_staticText190, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	MaxResNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("5.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( MaxResNumericCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText191 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Low Defocus For Search (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText191->Wrap( -1 );
	fgSizer1->Add( m_staticText191, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	LowDefocusNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("5000.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( LowDefocusNumericCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText192 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("High Defocus For Search (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText192->Wrap( -1 );
	fgSizer1->Add( m_staticText192, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	HighDefocusNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("50000.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( HighDefocusNumericCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText194 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Defocus Search Step (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText194->Wrap( -1 );
	fgSizer1->Add( m_staticText194, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	DefocusStepNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("500.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( DefocusStepNumericCtrl, 0, wxALL|wxEXPAND, 5 );
	
	LargeAstigmatismExpectedCheckBox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Very large astigmatism expected?"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( LargeAstigmatismExpectedCheckBox, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	RestrainAstigmatismCheckBox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Restrain Astigmatism?"), wxDefaultPosition, wxDefaultSize, 0 );
	RestrainAstigmatismCheckBox->SetValue(true); 
	fgSizer1->Add( RestrainAstigmatismCheckBox, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	ToleratedAstigmatismStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Tolerated Astigmatism (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	ToleratedAstigmatismStaticText->Wrap( -1 );
	fgSizer1->Add( ToleratedAstigmatismStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	ToleratedAstigmatismNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("100.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	fgSizer1->Add( ToleratedAstigmatismNumericCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText200 = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Phase Plates"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText200->Wrap( -1 );
	m_staticText200->SetFont( wxFont( 10, 74, 90, 92, true, wxT("Sans") ) );
	
	fgSizer1->Add( m_staticText200, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	AdditionalPhaseShiftCheckBox = new wxCheckBox( ExpertPanel, wxID_ANY, wxT("Find Additional Phase Shift?"), wxDefaultPosition, wxDefaultSize, 0 );
	fgSizer1->Add( AdditionalPhaseShiftCheckBox, 0, wxALL, 5 );
	
	
	fgSizer1->Add( 0, 0, 1, wxEXPAND, 5 );
	
	MinPhaseShiftStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Min. Phase Shift (rad) :"), wxDefaultPosition, wxDefaultSize, 0 );
	MinPhaseShiftStaticText->Wrap( -1 );
	MinPhaseShiftStaticText->Enable( false );
	
	fgSizer1->Add( MinPhaseShiftStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	MinPhaseShiftNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.00"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	MinPhaseShiftNumericCtrl->Enable( false );
	
	fgSizer1->Add( MinPhaseShiftNumericCtrl, 0, wxALL|wxEXPAND, 5 );
	
	MaxPhaseShiftStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Max. Phase Shift (rad) :"), wxDefaultPosition, wxDefaultSize, 0 );
	MaxPhaseShiftStaticText->Wrap( -1 );
	MaxPhaseShiftStaticText->Enable( false );
	
	fgSizer1->Add( MaxPhaseShiftStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	MaxPhaseShiftNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("3.15"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	MaxPhaseShiftNumericCtrl->Enable( false );
	
	fgSizer1->Add( MaxPhaseShiftNumericCtrl, 0, wxALL|wxEXPAND, 5 );
	
	PhaseShiftStepStaticText = new wxStaticText( ExpertPanel, wxID_ANY, wxT("Phase Shift Search Step (rad) :"), wxDefaultPosition, wxDefaultSize, 0 );
	PhaseShiftStepStaticText->Wrap( -1 );
	PhaseShiftStepStaticText->Enable( false );
	
	fgSizer1->Add( PhaseShiftStepStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	PhaseShiftStepNumericCtrl = new NumericTextCtrl( ExpertPanel, wxID_ANY, wxT("0.20"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	PhaseShiftStepNumericCtrl->Enable( false );
	
	fgSizer1->Add( PhaseShiftStepNumericCtrl, 0, wxALL|wxEXPAND, 5 );
	
	
	InputSizer->Add( fgSizer1, 1, wxEXPAND, 5 );
	
	
	ExpertPanel->SetSizer( InputSizer );
	ExpertPanel->Layout();
	InputSizer->Fit( ExpertPanel );
	bSizer46->Add( ExpertPanel, 0, wxALL|wxEXPAND, 5 );
	
	OutputTextPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	OutputTextPanel->Hide();
	
	wxBoxSizer* bSizer56;
	bSizer56 = new wxBoxSizer( wxVERTICAL );
	
	output_textctrl = new wxTextCtrl( OutputTextPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE|wxTE_READONLY );
	bSizer56->Add( output_textctrl, 1, wxALL|wxEXPAND, 5 );
	
	
	OutputTextPanel->SetSizer( bSizer56 );
	OutputTextPanel->Layout();
	bSizer56->Fit( OutputTextPanel );
	bSizer46->Add( OutputTextPanel, 30, wxEXPAND | wxALL, 5 );
	
	InfoPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer61;
	bSizer61 = new wxBoxSizer( wxVERTICAL );
	
	InfoText = new wxRichTextCtrl( InfoPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_READONLY|wxHSCROLL|wxVSCROLL );
	bSizer61->Add( InfoText, 1, wxEXPAND | wxALL, 5 );
	
	
	InfoPanel->SetSizer( bSizer61 );
	InfoPanel->Layout();
	bSizer61->Fit( InfoPanel );
	bSizer46->Add( InfoPanel, 1, wxEXPAND | wxALL, 5 );
	
	CTFResultsPanel = new ShowCTFResultsPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	CTFResultsPanel->Hide();
	
	bSizer46->Add( CTFResultsPanel, 50, wxEXPAND | wxALL, 5 );
	
	
	bSizer43->Add( bSizer46, 1, wxEXPAND, 5 );
	
	m_staticline11 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer43->Add( m_staticline11, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer48;
	bSizer48 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer70;
	bSizer70 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer71;
	bSizer71 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer55;
	bSizer55 = new wxBoxSizer( wxHORIZONTAL );
	
	
	bSizer71->Add( bSizer55, 0, wxALIGN_CENTER_HORIZONTAL, 5 );
	
	
	bSizer70->Add( bSizer71, 1, wxALIGN_CENTER_VERTICAL, 5 );
	
	wxBoxSizer* bSizer74;
	bSizer74 = new wxBoxSizer( wxVERTICAL );
	
	
	bSizer70->Add( bSizer74, 0, wxALIGN_CENTER_VERTICAL, 5 );
	
	ProgressPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	ProgressPanel->Hide();
	
	wxBoxSizer* bSizer57;
	bSizer57 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer59;
	bSizer59 = new wxBoxSizer( wxHORIZONTAL );
	
	NumberConnectedText = new wxStaticText( ProgressPanel, wxID_ANY, wxT("o"), wxDefaultPosition, wxDefaultSize, 0 );
	NumberConnectedText->Wrap( -1 );
	bSizer59->Add( NumberConnectedText, 0, wxALIGN_CENTER|wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	ProgressBar = new wxGauge( ProgressPanel, wxID_ANY, 100, wxDefaultPosition, wxDefaultSize, wxGA_HORIZONTAL );
	ProgressBar->SetValue( 0 ); 
	bSizer59->Add( ProgressBar, 100, wxALIGN_CENTER|wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );
	
	TimeRemainingText = new wxStaticText( ProgressPanel, wxID_ANY, wxT("Time Remaining : ???h:??m:??s   "), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	TimeRemainingText->Wrap( -1 );
	bSizer59->Add( TimeRemainingText, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	m_staticline60 = new wxStaticLine( ProgressPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL|wxLI_VERTICAL );
	bSizer59->Add( m_staticline60, 0, wxEXPAND | wxALL, 5 );
	
	FinishButton = new wxButton( ProgressPanel, wxID_ANY, wxT("Finish"), wxDefaultPosition, wxDefaultSize, 0 );
	FinishButton->Hide();
	
	bSizer59->Add( FinishButton, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	CancelAlignmentButton = new wxButton( ProgressPanel, wxID_ANY, wxT("Terminate Job"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer59->Add( CancelAlignmentButton, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer57->Add( bSizer59, 1, wxEXPAND, 5 );
	
	
	ProgressPanel->SetSizer( bSizer57 );
	ProgressPanel->Layout();
	bSizer57->Fit( ProgressPanel );
	bSizer70->Add( ProgressPanel, 1, wxEXPAND | wxALL, 5 );
	
	
	bSizer48->Add( bSizer70, 1, wxEXPAND, 5 );
	
	StartPanel = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer58;
	bSizer58 = new wxBoxSizer( wxHORIZONTAL );
	
	RunProfileText = new wxStaticText( StartPanel, wxID_ANY, wxT("Run Profile :"), wxDefaultPosition, wxDefaultSize, 0 );
	RunProfileText->Wrap( -1 );
	bSizer58->Add( RunProfileText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	RunProfileComboBox = new MemoryComboBox( StartPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	bSizer58->Add( RunProfileComboBox, 50, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	wxBoxSizer* bSizer60;
	bSizer60 = new wxBoxSizer( wxVERTICAL );
	
	StartEstimationButton = new wxButton( StartPanel, wxID_ANY, wxT("Start Estimation"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer60->Add( StartEstimationButton, 0, wxALL, 5 );
	
	
	bSizer58->Add( bSizer60, 50, wxEXPAND, 5 );
	
	
	StartPanel->SetSizer( bSizer58 );
	StartPanel->Layout();
	bSizer58->Fit( StartPanel );
	bSizer48->Add( StartPanel, 1, wxEXPAND | wxALL, 5 );
	
	
	bSizer43->Add( bSizer48, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer43 );
	this->Layout();
	
	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( FindCTFPanel::OnUpdateUI ) );
	ExpertToggleButton->Connect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( FindCTFPanel::OnExpertOptionsToggle ), NULL, this );
	MovieRadioButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( FindCTFPanel::OnMovieRadioButton ), NULL, this );
	ImageRadioButton->Connect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( FindCTFPanel::OnImageRadioButton ), NULL, this );
	LargeAstigmatismExpectedCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnLargeAstigmatismExpectedCheckBox ), NULL, this );
	RestrainAstigmatismCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnRestrainAstigmatismCheckBox ), NULL, this );
	AdditionalPhaseShiftCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnFindAdditionalPhaseCheckBox ), NULL, this );
	InfoText->Connect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( FindCTFPanel::OnInfoURL ), NULL, this );
	FinishButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFPanel::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFPanel::TerminateButtonClick ), NULL, this );
	StartEstimationButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFPanel::StartEstimationClick ), NULL, this );
}

FindCTFPanel::~FindCTFPanel()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( FindCTFPanel::OnUpdateUI ) );
	ExpertToggleButton->Disconnect( wxEVT_COMMAND_TOGGLEBUTTON_CLICKED, wxCommandEventHandler( FindCTFPanel::OnExpertOptionsToggle ), NULL, this );
	MovieRadioButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( FindCTFPanel::OnMovieRadioButton ), NULL, this );
	ImageRadioButton->Disconnect( wxEVT_COMMAND_RADIOBUTTON_SELECTED, wxCommandEventHandler( FindCTFPanel::OnImageRadioButton ), NULL, this );
	LargeAstigmatismExpectedCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnLargeAstigmatismExpectedCheckBox ), NULL, this );
	RestrainAstigmatismCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnRestrainAstigmatismCheckBox ), NULL, this );
	AdditionalPhaseShiftCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FindCTFPanel::OnFindAdditionalPhaseCheckBox ), NULL, this );
	InfoText->Disconnect( wxEVT_COMMAND_TEXT_URL, wxTextUrlEventHandler( FindCTFPanel::OnInfoURL ), NULL, this );
	FinishButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFPanel::FinishButtonClick ), NULL, this );
	CancelAlignmentButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFPanel::TerminateButtonClick ), NULL, this );
	StartEstimationButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FindCTFPanel::StartEstimationClick ), NULL, this );
	
}

NewProjectWizard::NewProjectWizard( wxWindow* parent, wxWindowID id, const wxString& title, const wxBitmap& bitmap, const wxPoint& pos, long style ) 
{
	this->Create( parent, id, title, bitmap, pos, style );
	this->SetSizeHints( wxSize( 500,350 ), wxDefaultSize );
	
	wxWizardPageSimple* m_wizPage1 = new wxWizardPageSimple( this );
	m_pages.Add( m_wizPage1 );
	
	wxBoxSizer* bSizer47;
	bSizer47 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText41 = new wxStaticText( m_wizPage1, wxID_ANY, wxT("Project Name :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText41->Wrap( -1 );
	bSizer47->Add( m_staticText41, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	ProjectNameTextCtrl = new wxTextCtrl( m_wizPage1, wxID_ANY, wxT("New_Project"), wxDefaultPosition, wxSize( -1,-1 ), 0 );
	ProjectNameTextCtrl->SetMinSize( wxSize( 500,-1 ) );
	
	bSizer47->Add( ProjectNameTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticText42 = new wxStaticText( m_wizPage1, wxID_ANY, wxT("Project Parent Directory :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText42->Wrap( -1 );
	bSizer47->Add( m_staticText42, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	wxBoxSizer* bSizer50;
	bSizer50 = new wxBoxSizer( wxHORIZONTAL );
	
	ParentDirTextCtrl = new wxTextCtrl( m_wizPage1, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	bSizer50->Add( ParentDirTextCtrl, 1, wxALL, 5 );
	
	m_button24 = new wxButton( m_wizPage1, wxID_ANY, wxT("Browse..."), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer50->Add( m_button24, 0, wxALL, 5 );
	
	
	bSizer47->Add( bSizer50, 0, wxEXPAND, 5 );
	
	m_staticText45 = new wxStaticText( m_wizPage1, wxID_ANY, wxT("Resulting project path :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText45->Wrap( -1 );
	bSizer47->Add( m_staticText45, 0, wxALL, 5 );
	
	ProjectPathTextCtrl = new wxTextCtrl( m_wizPage1, wxID_ANY, wxT("/tmp/New_Project"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer47->Add( ProjectPathTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	ErrorText = new wxStaticText( m_wizPage1, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	ErrorText->Wrap( -1 );
	ErrorText->SetForegroundColour( wxColour( 255, 0, 0 ) );
	
	bSizer47->Add( ErrorText, 0, wxALL, 5 );
	
	
	m_wizPage1->SetSizer( bSizer47 );
	m_wizPage1->Layout();
	bSizer47->Fit( m_wizPage1 );
	
	this->Centre( wxBOTH );
	
	for ( unsigned int i = 1; i < m_pages.GetCount(); i++ )
	{
		m_pages.Item( i )->SetPrev( m_pages.Item( i - 1 ) );
		m_pages.Item( i - 1 )->SetNext( m_pages.Item( i ) );
	}
	
	// Connect Events
	this->Connect( wxID_ANY, wxEVT_WIZARD_FINISHED, wxWizardEventHandler( NewProjectWizard::OnFinished ) );
	ProjectNameTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( NewProjectWizard::OnProjectTextChange ), NULL, this );
	ParentDirTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( NewProjectWizard::OnParentDirChange ), NULL, this );
	m_button24->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( NewProjectWizard::OnBrowseButtonClick ), NULL, this );
	ProjectPathTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( NewProjectWizard::OnProjectPathChange ), NULL, this );
}

NewProjectWizard::~NewProjectWizard()
{
	// Disconnect Events
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_FINISHED, wxWizardEventHandler( NewProjectWizard::OnFinished ) );
	ProjectNameTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( NewProjectWizard::OnProjectTextChange ), NULL, this );
	ParentDirTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( NewProjectWizard::OnParentDirChange ), NULL, this );
	m_button24->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( NewProjectWizard::OnBrowseButtonClick ), NULL, this );
	ProjectPathTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_UPDATED, wxCommandEventHandler( NewProjectWizard::OnProjectPathChange ), NULL, this );
	
	m_pages.Clear();
}

AddRunCommandDialog::AddRunCommandDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxBoxSizer* bSizer45;
	bSizer45 = new wxBoxSizer( wxVERTICAL );
	
	wxFlexGridSizer* fgSizer2;
	fgSizer2 = new wxFlexGridSizer( 3, 2, 0, 0 );
	fgSizer2->AddGrowableCol( 1 );
	fgSizer2->SetFlexibleDirection( wxBOTH );
	fgSizer2->SetNonFlexibleGrowMode( wxFLEX_GROWMODE_SPECIFIED );
	
	m_staticText45 = new wxStaticText( this, wxID_ANY, wxT("Command :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText45->Wrap( -1 );
	fgSizer2->Add( m_staticText45, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	CommandTextCtrl = new wxTextCtrl( this, wxID_ANY, wxT("$command"), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
	CommandTextCtrl->SetMinSize( wxSize( 300,-1 ) );
	
	fgSizer2->Add( CommandTextCtrl, 1, wxALL|wxEXPAND, 5 );
	
	m_staticText46 = new wxStaticText( this, wxID_ANY, wxT("No. Copies :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText46->Wrap( -1 );
	fgSizer2->Add( m_staticText46, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	NumberCopiesSpinCtrl = new wxSpinCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 999999, 1 );
	fgSizer2->Add( NumberCopiesSpinCtrl, 1, wxALL|wxEXPAND, 5 );
	
	m_staticText58 = new wxStaticText( this, wxID_ANY, wxT("Delay (ms) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText58->Wrap( -1 );
	fgSizer2->Add( m_staticText58, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	DelayTimeSpinCtrl = new wxSpinCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 0, 10000, 100 );
	fgSizer2->Add( DelayTimeSpinCtrl, 0, wxALL|wxEXPAND, 5 );
	
	
	bSizer45->Add( fgSizer2, 0, wxEXPAND, 5 );
	
	ErrorStaticText = new wxStaticText( this, wxID_ANY, wxT("Oops! - Command must contain \"$command\""), wxDefaultPosition, wxDefaultSize, 0 );
	ErrorStaticText->Wrap( -1 );
	ErrorStaticText->SetForegroundColour( wxColour( 180, 0, 0 ) );
	ErrorStaticText->Hide();
	
	bSizer45->Add( ErrorStaticText, 0, wxALIGN_CENTER|wxALL, 5 );
	
	m_staticline14 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer45->Add( m_staticline14, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer47;
	bSizer47 = new wxBoxSizer( wxHORIZONTAL );
	
	OKButton = new wxButton( this, wxID_ANY, wxT("OK"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer47->Add( OKButton, 0, wxALL, 5 );
	
	CancelButton = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer47->Add( CancelButton, 0, wxALL, 5 );
	
	
	bSizer45->Add( bSizer47, 0, wxALIGN_CENTER, 5 );
	
	
	this->SetSizer( bSizer45 );
	this->Layout();
	bSizer45->Fit( this );
	
	this->Centre( wxBOTH );
	
	// Connect Events
	CommandTextCtrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( AddRunCommandDialog::OnEnter ), NULL, this );
	OKButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AddRunCommandDialog::OnOKClick ), NULL, this );
	CancelButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AddRunCommandDialog::OnCancelClick ), NULL, this );
}

AddRunCommandDialog::~AddRunCommandDialog()
{
	// Disconnect Events
	CommandTextCtrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( AddRunCommandDialog::OnEnter ), NULL, this );
	OKButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AddRunCommandDialog::OnOKClick ), NULL, this );
	CancelButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( AddRunCommandDialog::OnCancelClick ), NULL, this );
	
}

RenameDialog::RenameDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( -1,-1 ), wxDefaultSize );
	
	MainBoxSizer = new wxBoxSizer( wxVERTICAL );
	
	m_staticText246 = new wxStaticText( this, wxID_ANY, wxT("Set Asset Names :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText246->Wrap( -1 );
	MainBoxSizer->Add( m_staticText246, 0, wxALL, 5 );
	
	m_staticline18 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainBoxSizer->Add( m_staticline18, 0, wxEXPAND | wxALL, 5 );
	
	RenameScrollPanel = new wxScrolledWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxVSCROLL );
	RenameScrollPanel->SetScrollRate( 5, 5 );
	RenameBoxSizer = new wxBoxSizer( wxVERTICAL );
	
	
	RenameScrollPanel->SetSizer( RenameBoxSizer );
	RenameScrollPanel->Layout();
	RenameBoxSizer->Fit( RenameScrollPanel );
	MainBoxSizer->Add( RenameScrollPanel, 1, wxEXPAND, 5 );
	
	m_staticline19 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainBoxSizer->Add( m_staticline19, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer90;
	bSizer90 = new wxBoxSizer( wxHORIZONTAL );
	
	
	bSizer90->Add( 0, 0, 1, wxEXPAND, 5 );
	
	CancelButton = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer90->Add( CancelButton, 0, wxALL, 5 );
	
	RenameButton = new wxButton( this, wxID_ANY, wxT("Set Name"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer90->Add( RenameButton, 0, wxALL, 5 );
	
	
	bSizer90->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	MainBoxSizer->Add( bSizer90, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( MainBoxSizer );
	this->Layout();
	MainBoxSizer->Fit( this );
	
	this->Centre( wxBOTH );
	
	// Connect Events
	CancelButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RenameDialog::OnCancelClick ), NULL, this );
	RenameButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RenameDialog::OnRenameClick ), NULL, this );
}

RenameDialog::~RenameDialog()
{
	// Disconnect Events
	CancelButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RenameDialog::OnCancelClick ), NULL, this );
	RenameButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RenameDialog::OnRenameClick ), NULL, this );
	
}

NewRefinementPackageWizard::NewRefinementPackageWizard( wxWindow* parent, wxWindowID id, const wxString& title, const wxBitmap& bitmap, const wxPoint& pos, long style ) 
{
	this->Create( parent, id, title, bitmap, pos, style );
	this->SetSizeHints( wxSize( -1,-1 ), wxDefaultSize );
	
	
	this->Centre( wxBOTH );
	
	
	// Connect Events
	this->Connect( wxID_ANY, wxEVT_WIZARD_FINISHED, wxWizardEventHandler( NewRefinementPackageWizard::OnFinished ) );
	this->Connect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGED, wxWizardEventHandler( NewRefinementPackageWizard::PageChanged ) );
	this->Connect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGING, wxWizardEventHandler( NewRefinementPackageWizard::PageChanging ) );
}

NewRefinementPackageWizard::~NewRefinementPackageWizard()
{
	// Disconnect Events
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_FINISHED, wxWizardEventHandler( NewRefinementPackageWizard::OnFinished ) );
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGED, wxWizardEventHandler( NewRefinementPackageWizard::PageChanged ) );
	this->Disconnect( wxID_ANY, wxEVT_WIZARD_PAGE_CHANGING, wxWizardEventHandler( NewRefinementPackageWizard::PageChanging ) );
	
	m_pages.Clear();
}

VolumeChooserDialog::VolumeChooserDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( -1,-1 ), wxDefaultSize );
	
	MainBoxSizer = new wxBoxSizer( wxVERTICAL );
	
	m_staticText246 = new wxStaticText( this, wxID_ANY, wxT("Select New Volume :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText246->Wrap( -1 );
	MainBoxSizer->Add( m_staticText246, 0, wxALL, 5 );
	
	m_staticline18 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainBoxSizer->Add( m_staticline18, 0, wxEXPAND | wxALL, 5 );
	
	ComboBox = new wxComboBox( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	ComboBox->SetSelection( 0 );
	MainBoxSizer->Add( ComboBox, 0, wxALL|wxEXPAND, 5 );
	
	m_staticline19 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainBoxSizer->Add( m_staticline19, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer90;
	bSizer90 = new wxBoxSizer( wxHORIZONTAL );
	
	
	bSizer90->Add( 0, 0, 1, wxEXPAND, 5 );
	
	CancelButton = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer90->Add( CancelButton, 0, wxALL, 5 );
	
	SetButton = new wxButton( this, wxID_ANY, wxT("Set Reference"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer90->Add( SetButton, 0, wxALL, 5 );
	
	
	bSizer90->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	MainBoxSizer->Add( bSizer90, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( MainBoxSizer );
	this->Layout();
	MainBoxSizer->Fit( this );
	
	this->Centre( wxBOTH );
	
	// Connect Events
	CancelButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeChooserDialog::OnCancelClick ), NULL, this );
	SetButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeChooserDialog::OnRenameClick ), NULL, this );
}

VolumeChooserDialog::~VolumeChooserDialog()
{
	// Disconnect Events
	CancelButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeChooserDialog::OnCancelClick ), NULL, this );
	SetButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( VolumeChooserDialog::OnRenameClick ), NULL, this );
	
}

FilterDialog::FilterDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( -1,-1 ), wxDefaultSize );
	
	MainBoxSizer = new wxBoxSizer( wxVERTICAL );
	
	m_staticText64 = new wxStaticText( this, wxID_ANY, wxT("Filter By :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText64->Wrap( -1 );
	m_staticText64->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	MainBoxSizer->Add( m_staticText64, 0, wxALL, 5 );
	
	m_staticline18 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainBoxSizer->Add( m_staticline18, 0, wxEXPAND | wxALL, 5 );
	
	FilterScrollPanel = new wxScrolledWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSUNKEN_BORDER|wxVSCROLL );
	FilterScrollPanel->SetScrollRate( 5, 5 );
	FilterBoxSizer = new wxBoxSizer( wxVERTICAL );
	
	
	FilterScrollPanel->SetSizer( FilterBoxSizer );
	FilterScrollPanel->Layout();
	FilterBoxSizer->Fit( FilterScrollPanel );
	MainBoxSizer->Add( FilterScrollPanel, 1, wxALL|wxEXPAND, 5 );
	
	m_staticText81 = new wxStaticText( this, wxID_ANY, wxT("\nSort By :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText81->Wrap( -1 );
	m_staticText81->SetFont( wxFont( wxNORMAL_FONT->GetPointSize(), 70, 90, 92, false, wxEmptyString ) );
	
	MainBoxSizer->Add( m_staticText81, 0, wxALL, 5 );
	
	m_staticline19 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainBoxSizer->Add( m_staticline19, 0, wxEXPAND | wxALL, 5 );
	
	SortScrollPanel = new wxScrolledWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxVSCROLL );
	SortScrollPanel->SetScrollRate( 5, 5 );
	SortSizer = new wxGridSizer( 0, 3, 0, 0 );
	
	
	SortScrollPanel->SetSizer( SortSizer );
	SortScrollPanel->Layout();
	SortSizer->Fit( SortScrollPanel );
	MainBoxSizer->Add( SortScrollPanel, 0, wxEXPAND | wxALL, 5 );
	
	m_staticline21 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainBoxSizer->Add( m_staticline21, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer90;
	bSizer90 = new wxBoxSizer( wxHORIZONTAL );
	
	
	bSizer90->Add( 0, 0, 1, wxEXPAND, 5 );
	
	CancelButton = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer90->Add( CancelButton, 0, wxALL, 5 );
	
	FilterButton = new wxButton( this, wxID_ANY, wxT("Filter/Sort"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer90->Add( FilterButton, 0, wxALL, 5 );
	
	
	bSizer90->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	MainBoxSizer->Add( bSizer90, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( MainBoxSizer );
	this->Layout();
	MainBoxSizer->Fit( this );
	
	this->Centre( wxBOTH );
	
	// Connect Events
	CancelButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FilterDialog::OnCancelClick ), NULL, this );
	FilterButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FilterDialog::OnFilterClick ), NULL, this );
}

FilterDialog::~FilterDialog()
{
	// Disconnect Events
	CancelButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FilterDialog::OnCancelClick ), NULL, this );
	FilterButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FilterDialog::OnFilterClick ), NULL, this );
	
}

ParticlePositionExportDialog::ParticlePositionExportDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxBoxSizer* bSizer133;
	bSizer133 = new wxBoxSizer( wxVERTICAL );
	
	m_panel38 = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer135;
	bSizer135 = new wxBoxSizer( wxVERTICAL );
	
	wxStaticBoxSizer* sbSizer3;
	sbSizer3 = new wxStaticBoxSizer( new wxStaticBox( m_panel38, wxID_ANY, wxT("Export from these images") ), wxVERTICAL );
	
	GroupComboBox = new wxComboBox( m_panel38, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	sbSizer3->Add( GroupComboBox, 0, wxALL|wxEXPAND, 5 );
	
	
	bSizer135->Add( sbSizer3, 0, wxEXPAND, 25 );
	
	wxStaticBoxSizer* sbSizer4;
	sbSizer4 = new wxStaticBoxSizer( new wxStaticBox( m_panel38, wxID_ANY, wxT("Destination directory") ), wxVERTICAL );
	
	DestinationDirectoryPickerCtrl = new wxDirPickerCtrl( m_panel38, wxID_ANY, wxEmptyString, wxT("Select a folder"), wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE );
	sbSizer4->Add( DestinationDirectoryPickerCtrl, 0, wxALL|wxEXPAND, 5 );
	
	
	bSizer135->Add( sbSizer4, 0, wxEXPAND, 5 );
	
	
	bSizer135->Add( 0, 0, 0, wxEXPAND, 5 );
	
	WarningText = new wxStaticText( m_panel38, wxID_ANY, wxT("Warning: running jobs \nmay affect exported coordinates"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	WarningText->Wrap( -1 );
	WarningText->SetForegroundColour( wxColour( 180, 0, 0 ) );
	WarningText->Hide();
	
	bSizer135->Add( WarningText, 0, wxALL, 5 );
	
	wxBoxSizer* bSizer137;
	bSizer137 = new wxBoxSizer( wxHORIZONTAL );
	
	
	bSizer137->Add( 0, 0, 1, wxEXPAND, 5 );
	
	CancelButton = new wxButton( m_panel38, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer137->Add( CancelButton, 0, wxALL, 5 );
	
	ExportButton = new wxButton( m_panel38, wxID_ANY, wxT("Export"), wxDefaultPosition, wxDefaultSize, 0 );
	ExportButton->SetDefault(); 
	ExportButton->Enable( false );
	
	bSizer137->Add( ExportButton, 0, wxALL, 5 );
	
	
	bSizer135->Add( bSizer137, 0, wxEXPAND, 5 );
	
	
	m_panel38->SetSizer( bSizer135 );
	m_panel38->Layout();
	bSizer135->Fit( m_panel38 );
	bSizer133->Add( m_panel38, 0, wxEXPAND | wxALL, 5 );
	
	
	this->SetSizer( bSizer133 );
	this->Layout();
	bSizer133->Fit( this );
	
	this->Centre( wxBOTH );
	
	// Connect Events
	DestinationDirectoryPickerCtrl->Connect( wxEVT_COMMAND_DIRPICKER_CHANGED, wxFileDirPickerEventHandler( ParticlePositionExportDialog::OnDirChanged ), NULL, this );
	CancelButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ParticlePositionExportDialog::OnCancelButtonClick ), NULL, this );
	ExportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ParticlePositionExportDialog::OnExportButtonClick ), NULL, this );
}

ParticlePositionExportDialog::~ParticlePositionExportDialog()
{
	// Disconnect Events
	DestinationDirectoryPickerCtrl->Disconnect( wxEVT_COMMAND_DIRPICKER_CHANGED, wxFileDirPickerEventHandler( ParticlePositionExportDialog::OnDirChanged ), NULL, this );
	CancelButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ParticlePositionExportDialog::OnCancelButtonClick ), NULL, this );
	ExportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ParticlePositionExportDialog::OnExportButtonClick ), NULL, this );
	
}

FrealignExportDialog::FrealignExportDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxBoxSizer* bSizer133;
	bSizer133 = new wxBoxSizer( wxVERTICAL );
	
	m_panel38 = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer135;
	bSizer135 = new wxBoxSizer( wxVERTICAL );
	
	wxStaticBoxSizer* sbSizer3;
	sbSizer3 = new wxStaticBoxSizer( new wxStaticBox( m_panel38, wxID_ANY, wxT("Export from these images") ), wxVERTICAL );
	
	GroupComboBox = new wxComboBox( m_panel38, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	sbSizer3->Add( GroupComboBox, 0, wxALL|wxEXPAND, 5 );
	
	
	bSizer135->Add( sbSizer3, 0, wxEXPAND, 25 );
	
	wxBoxSizer* bSizer141;
	bSizer141 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer142;
	bSizer142 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText202 = new wxStaticText( m_panel38, wxID_ANY, wxT("Downsampling factor : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText202->Wrap( -1 );
	bSizer142->Add( m_staticText202, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	DownsamplingFactorSpinCtrl = new wxSpinCtrl( m_panel38, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 10, 1 );
	bSizer142->Add( DownsamplingFactorSpinCtrl, 0, wxALL, 5 );
	
	
	bSizer141->Add( bSizer142, 0, wxEXPAND, 5 );
	
	
	bSizer135->Add( bSizer141, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer145;
	bSizer145 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer143;
	bSizer143 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText203 = new wxStaticText( m_panel38, wxID_ANY, wxT("Box size after downsampling : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText203->Wrap( -1 );
	bSizer143->Add( m_staticText203, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	BoxSizeSpinCtrl = new wxSpinCtrl( m_panel38, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 999999999, 512 );
	bSizer143->Add( BoxSizeSpinCtrl, 0, wxALL, 5 );
	
	
	bSizer145->Add( bSizer143, 1, wxEXPAND, 5 );
	
	
	bSizer135->Add( bSizer145, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer140;
	bSizer140 = new wxBoxSizer( wxVERTICAL );
	
	NormalizeCheckBox = new wxCheckBox( m_panel38, wxID_ANY, wxT("Normalize images"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer140->Add( NormalizeCheckBox, 0, wxALL, 5 );
	
	
	bSizer135->Add( bSizer140, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer213;
	bSizer213 = new wxBoxSizer( wxVERTICAL );
	
	FlipCTFCheckBox = new wxCheckBox( m_panel38, wxID_ANY, wxT("Flip CTF"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer213->Add( FlipCTFCheckBox, 0, wxALL, 5 );
	
	
	bSizer135->Add( bSizer213, 1, wxEXPAND, 5 );
	
	wxStaticBoxSizer* sbSizer4;
	sbSizer4 = new wxStaticBoxSizer( new wxStaticBox( m_panel38, wxID_ANY, wxT("Output image stack") ), wxVERTICAL );
	
	OutputImageStackPicker = new wxFilePickerCtrl( m_panel38, wxID_ANY, wxEmptyString, wxT("Select a file"), wxT("MRC files (*.mrc, *.mrcs)|*.mrc;*.mrcs"), wxDefaultPosition, wxDefaultSize, wxFLP_OVERWRITE_PROMPT|wxFLP_SAVE|wxFLP_USE_TEXTCTRL );
	sbSizer4->Add( OutputImageStackPicker, 0, wxALL|wxEXPAND, 5 );
	
	
	bSizer135->Add( sbSizer4, 0, wxEXPAND, 5 );
	
	
	bSizer135->Add( 0, 0, 0, wxEXPAND, 5 );
	
	WarningText = new wxStaticText( m_panel38, wxID_ANY, wxT("Warning: running jobs \nmay affect exported coordinates"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	WarningText->Wrap( -1 );
	WarningText->SetForegroundColour( wxColour( 180, 0, 0 ) );
	WarningText->Hide();
	
	bSizer135->Add( WarningText, 0, wxALL, 5 );
	
	wxBoxSizer* bSizer137;
	bSizer137 = new wxBoxSizer( wxHORIZONTAL );
	
	
	bSizer137->Add( 0, 0, 1, wxEXPAND, 5 );
	
	CancelButton = new wxButton( m_panel38, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer137->Add( CancelButton, 0, wxALL, 5 );
	
	ExportButton = new wxButton( m_panel38, wxID_ANY, wxT("Export"), wxDefaultPosition, wxDefaultSize, 0 );
	ExportButton->SetDefault(); 
	ExportButton->Enable( false );
	
	bSizer137->Add( ExportButton, 0, wxALL, 5 );
	
	
	bSizer135->Add( bSizer137, 0, wxEXPAND, 5 );
	
	
	m_panel38->SetSizer( bSizer135 );
	m_panel38->Layout();
	bSizer135->Fit( m_panel38 );
	bSizer133->Add( m_panel38, 0, wxEXPAND | wxALL, 5 );
	
	
	this->SetSizer( bSizer133 );
	this->Layout();
	bSizer133->Fit( this );
	
	this->Centre( wxBOTH );
	
	// Connect Events
	FlipCTFCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FrealignExportDialog::OnFlipCTFCheckBox ), NULL, this );
	OutputImageStackPicker->Connect( wxEVT_COMMAND_FILEPICKER_CHANGED, wxFileDirPickerEventHandler( FrealignExportDialog::OnOutputImageStackFileChanged ), NULL, this );
	CancelButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FrealignExportDialog::OnCancelButtonClick ), NULL, this );
	ExportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FrealignExportDialog::OnExportButtonClick ), NULL, this );
}

FrealignExportDialog::~FrealignExportDialog()
{
	// Disconnect Events
	FlipCTFCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( FrealignExportDialog::OnFlipCTFCheckBox ), NULL, this );
	OutputImageStackPicker->Disconnect( wxEVT_COMMAND_FILEPICKER_CHANGED, wxFileDirPickerEventHandler( FrealignExportDialog::OnOutputImageStackFileChanged ), NULL, this );
	CancelButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FrealignExportDialog::OnCancelButtonClick ), NULL, this );
	ExportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( FrealignExportDialog::OnExportButtonClick ), NULL, this );
	
}

RelionExportDialog::RelionExportDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxBoxSizer* bSizer133;
	bSizer133 = new wxBoxSizer( wxVERTICAL );
	
	m_panel38 = new wxPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer135;
	bSizer135 = new wxBoxSizer( wxVERTICAL );
	
	wxStaticBoxSizer* sbSizer3;
	sbSizer3 = new wxStaticBoxSizer( new wxStaticBox( m_panel38, wxID_ANY, wxT("Export from these images") ), wxVERTICAL );
	
	GroupComboBox = new wxComboBox( m_panel38, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	sbSizer3->Add( GroupComboBox, 0, wxALL|wxEXPAND, 5 );
	
	
	bSizer135->Add( sbSizer3, 0, wxEXPAND, 25 );
	
	wxBoxSizer* bSizer141;
	bSizer141 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer142;
	bSizer142 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText202 = new wxStaticText( m_panel38, wxID_ANY, wxT("Downsampling factor : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText202->Wrap( -1 );
	bSizer142->Add( m_staticText202, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	DownsamplingFactorSpinCtrl = new wxSpinCtrl( m_panel38, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 10, 1 );
	bSizer142->Add( DownsamplingFactorSpinCtrl, 0, wxALL|wxEXPAND, 5 );
	
	
	bSizer141->Add( bSizer142, 0, wxEXPAND, 5 );
	
	
	bSizer135->Add( bSizer141, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer145;
	bSizer145 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer143;
	bSizer143 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText203 = new wxStaticText( m_panel38, wxID_ANY, wxT("Box size after downsampling : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText203->Wrap( -1 );
	bSizer143->Add( m_staticText203, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	BoxSizeSpinCtrl = new wxSpinCtrl( m_panel38, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 999999999, 512 );
	bSizer143->Add( BoxSizeSpinCtrl, 0, wxALL|wxEXPAND, 5 );
	
	
	bSizer145->Add( bSizer143, 1, wxEXPAND, 5 );
	
	
	bSizer135->Add( bSizer145, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer140;
	bSizer140 = new wxBoxSizer( wxVERTICAL );
	
	NormalizeCheckBox = new wxCheckBox( m_panel38, wxID_ANY, wxT("Normalize images"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer140->Add( NormalizeCheckBox, 0, wxALL, 5 );
	
	
	bSizer135->Add( bSizer140, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer1451;
	bSizer1451 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer1431;
	bSizer1431 = new wxBoxSizer( wxHORIZONTAL );
	
	particleRadiusStaticText = new wxStaticText( m_panel38, wxID_ANY, wxT("Particle radius (A) : "), wxDefaultPosition, wxDefaultSize, 0 );
	particleRadiusStaticText->Wrap( -1 );
	particleRadiusStaticText->Enable( false );
	
	bSizer1431->Add( particleRadiusStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	particleRadiusTextCtrl = new wxTextCtrl( m_panel38, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	particleRadiusTextCtrl->Enable( false );
	
	bSizer1431->Add( particleRadiusTextCtrl, 0, wxALL|wxEXPAND, 5 );
	
	
	bSizer1451->Add( bSizer1431, 1, wxEXPAND, 5 );
	
	
	bSizer135->Add( bSizer1451, 1, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer213;
	bSizer213 = new wxBoxSizer( wxVERTICAL );
	
	FlipCTFCheckBox = new wxCheckBox( m_panel38, wxID_ANY, wxT("Flip CTF"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer213->Add( FlipCTFCheckBox, 0, wxALL, 5 );
	
	
	bSizer135->Add( bSizer213, 1, wxEXPAND, 5 );
	
	wxStaticBoxSizer* sbSizer4;
	sbSizer4 = new wxStaticBoxSizer( new wxStaticBox( m_panel38, wxID_ANY, wxT("Output image stack") ), wxVERTICAL );
	
	wxBoxSizer* bSizer229;
	bSizer229 = new wxBoxSizer( wxHORIZONTAL );
	
	OutputImageStackPicker = new wxFilePickerCtrl( m_panel38, wxID_ANY, wxEmptyString, wxT("Select a file"), wxT("MRC files (*.mrcs)|*.mrcs"), wxDefaultPosition, wxDefaultSize, wxFLP_OVERWRITE_PROMPT|wxFLP_SAVE );
	bSizer229->Add( OutputImageStackPicker, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	FileNameStaticText = new wxStaticText( m_panel38, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	FileNameStaticText->Wrap( -1 );
	bSizer229->Add( FileNameStaticText, 0, wxALIGN_CENTER_VERTICAL|wxALL|wxEXPAND, 5 );
	
	
	sbSizer4->Add( bSizer229, 1, wxEXPAND, 5 );
	
	
	bSizer135->Add( sbSizer4, 0, wxEXPAND, 5 );
	
	
	bSizer135->Add( 0, 0, 0, wxEXPAND, 5 );
	
	WarningText = new wxStaticText( m_panel38, wxID_ANY, wxT("Warning: running jobs \nmay affect exported coordinates"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
	WarningText->Wrap( -1 );
	WarningText->SetForegroundColour( wxColour( 180, 0, 0 ) );
	WarningText->Hide();
	
	bSizer135->Add( WarningText, 0, wxALL, 5 );
	
	wxBoxSizer* bSizer137;
	bSizer137 = new wxBoxSizer( wxHORIZONTAL );
	
	
	bSizer137->Add( 0, 0, 1, wxEXPAND, 5 );
	
	CancelButton = new wxButton( m_panel38, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer137->Add( CancelButton, 0, wxALL, 5 );
	
	ExportButton = new wxButton( m_panel38, wxID_ANY, wxT("Export"), wxDefaultPosition, wxDefaultSize, 0 );
	ExportButton->SetDefault(); 
	ExportButton->Enable( false );
	
	bSizer137->Add( ExportButton, 0, wxALL, 5 );
	
	
	bSizer135->Add( bSizer137, 0, wxEXPAND, 5 );
	
	
	m_panel38->SetSizer( bSizer135 );
	m_panel38->Layout();
	bSizer135->Fit( m_panel38 );
	bSizer133->Add( m_panel38, 0, wxEXPAND | wxALL, 5 );
	
	
	this->SetSizer( bSizer133 );
	this->Layout();
	bSizer133->Fit( this );
	
	this->Centre( wxBOTH );
	
	// Connect Events
	NormalizeCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( RelionExportDialog::OnNormalizeCheckBox ), NULL, this );
	FlipCTFCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( RelionExportDialog::OnFlipCTFCheckBox ), NULL, this );
	OutputImageStackPicker->Connect( wxEVT_COMMAND_FILEPICKER_CHANGED, wxFileDirPickerEventHandler( RelionExportDialog::OnOutputImageStackFileChanged ), NULL, this );
	CancelButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RelionExportDialog::OnCancelButtonClick ), NULL, this );
	ExportButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RelionExportDialog::OnExportButtonClick ), NULL, this );
}

RelionExportDialog::~RelionExportDialog()
{
	// Disconnect Events
	NormalizeCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( RelionExportDialog::OnNormalizeCheckBox ), NULL, this );
	FlipCTFCheckBox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( RelionExportDialog::OnFlipCTFCheckBox ), NULL, this );
	OutputImageStackPicker->Disconnect( wxEVT_COMMAND_FILEPICKER_CHANGED, wxFileDirPickerEventHandler( RelionExportDialog::OnOutputImageStackFileChanged ), NULL, this );
	CancelButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RelionExportDialog::OnCancelButtonClick ), NULL, this );
	ExportButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RelionExportDialog::OnExportButtonClick ), NULL, this );
	
}

RefinementPackageAssetPanel::RefinementPackageAssetPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer187;
	bSizer187 = new wxBoxSizer( wxVERTICAL );
	
	m_staticline52 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer187->Add( m_staticline52, 0, wxEXPAND | wxALL, 5 );
	
	m_splitter11 = new wxSplitterWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_3D );
	m_splitter11->Connect( wxEVT_IDLE, wxIdleEventHandler( RefinementPackageAssetPanel::m_splitter11OnIdle ), NULL, this );
	
	m_panel50 = new wxPanel( m_splitter11, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer188;
	bSizer188 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText313 = new wxStaticText( m_panel50, wxID_ANY, wxT("Refinement Packages :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText313->Wrap( -1 );
	bSizer188->Add( m_staticText313, 0, wxALL, 5 );
	
	wxBoxSizer* bSizer145;
	bSizer145 = new wxBoxSizer( wxHORIZONTAL );
	
	wxBoxSizer* bSizer193;
	bSizer193 = new wxBoxSizer( wxVERTICAL );
	
	CreateButton = new wxButton( m_panel50, wxID_ANY, wxT("Create"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer193->Add( CreateButton, 0, wxALL, 5 );
	
	RenameButton = new wxButton( m_panel50, wxID_ANY, wxT("Rename"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer193->Add( RenameButton, 0, wxALL, 5 );
	
	DeleteButton = new wxButton( m_panel50, wxID_ANY, wxT("Delete"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer193->Add( DeleteButton, 0, wxALL, 5 );
	
	
	bSizer145->Add( bSizer193, 0, wxEXPAND, 5 );
	
	RefinementPackageListCtrl = new RefinementPackageListControl( m_panel50, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_EDIT_LABELS|wxLC_NO_HEADER|wxLC_REPORT|wxLC_SINGLE_SEL|wxLC_VIRTUAL );
	bSizer145->Add( RefinementPackageListCtrl, 1, wxALL|wxEXPAND, 5 );
	
	
	bSizer188->Add( bSizer145, 1, wxEXPAND, 5 );
	
	
	m_panel50->SetSizer( bSizer188 );
	m_panel50->Layout();
	bSizer188->Fit( m_panel50 );
	m_panel51 = new wxPanel( m_splitter11, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	wxBoxSizer* bSizer191;
	bSizer191 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText314 = new wxStaticText( m_panel51, wxID_ANY, wxT("Contained Particles :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText314->Wrap( -1 );
	bSizer191->Add( m_staticText314, 0, wxALL, 5 );
	
	ContainedParticlesListCtrl = new ContainedParticleListControl( m_panel51, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_REPORT|wxLC_SINGLE_SEL|wxLC_VIRTUAL );
	bSizer191->Add( ContainedParticlesListCtrl, 1, wxALL|wxEXPAND, 5 );
	
	wxBoxSizer* bSizer199;
	bSizer199 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText230 = new wxStaticText( m_panel51, wxID_ANY, wxT("Active 3D References :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText230->Wrap( -1 );
	bSizer199->Add( m_staticText230, 0, wxALIGN_BOTTOM|wxALL, 5 );
	
	
	bSizer199->Add( 0, 0, 1, wxEXPAND, 5 );
	
	DisplayStackButton = new wxButton( m_panel51, wxID_ANY, wxT("Display Stack"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer199->Add( DisplayStackButton, 0, wxALL, 5 );
	
	
	bSizer191->Add( bSizer199, 0, wxEXPAND, 5 );
	
	Active3DReferencesListCtrl = new ReferenceVolumesListControl( m_panel51, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_REPORT|wxLC_SINGLE_SEL|wxLC_VIRTUAL );
	bSizer191->Add( Active3DReferencesListCtrl, 0, wxALL|wxEXPAND, 5 );
	
	
	m_panel51->SetSizer( bSizer191 );
	m_panel51->Layout();
	bSizer191->Fit( m_panel51 );
	m_splitter11->SplitVertically( m_panel50, m_panel51, 600 );
	bSizer187->Add( m_splitter11, 1, wxEXPAND, 5 );
	
	m_staticline53 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer187->Add( m_staticline53, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer192;
	bSizer192 = new wxBoxSizer( wxHORIZONTAL );
	
	wxGridSizer* gSizer12;
	gSizer12 = new wxGridSizer( 0, 6, 0, 0 );
	
	m_staticText319 = new wxStaticText( this, wxID_ANY, wxT("Stack Filename :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText319->Wrap( -1 );
	gSizer12->Add( m_staticText319, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	StackFileNameText = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	StackFileNameText->Wrap( -1 );
	gSizer12->Add( StackFileNameText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	gSizer12->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	gSizer12->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	gSizer12->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	gSizer12->Add( 0, 0, 1, wxEXPAND, 5 );
	
	m_staticText210 = new wxStaticText( this, wxID_ANY, wxT("Stack Box Size :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText210->Wrap( -1 );
	gSizer12->Add( m_staticText210, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	StackBoxSizeText = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	StackBoxSizeText->Wrap( -1 );
	gSizer12->Add( StackBoxSizeText, 0, wxALL, 5 );
	
	m_staticText315 = new wxStaticText( this, wxID_ANY, wxT("No. Classes :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText315->Wrap( -1 );
	gSizer12->Add( m_staticText315, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	NumberofClassesText = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	NumberofClassesText->Wrap( -1 );
	gSizer12->Add( NumberofClassesText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	m_staticText279 = new wxStaticText( this, wxID_ANY, wxT("Symmetry :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText279->Wrap( -1 );
	gSizer12->Add( m_staticText279, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	SymmetryText = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	SymmetryText->Wrap( -1 );
	gSizer12->Add( SymmetryText, 0, wxALL, 5 );
	
	m_staticText281 = new wxStaticText( this, wxID_ANY, wxT("M.W. :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText281->Wrap( -1 );
	gSizer12->Add( m_staticText281, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	MolecularWeightText = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	MolecularWeightText->Wrap( -1 );
	gSizer12->Add( MolecularWeightText, 0, wxALL, 5 );
	
	m_staticText283 = new wxStaticText( this, wxID_ANY, wxT("Largest Dimension :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText283->Wrap( -1 );
	gSizer12->Add( m_staticText283, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	LargestDimensionText = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	LargestDimensionText->Wrap( -1 );
	gSizer12->Add( LargestDimensionText, 0, wxALL, 5 );
	
	m_staticText317 = new wxStaticText( this, wxID_ANY, wxT("No. Refinements :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText317->Wrap( -1 );
	gSizer12->Add( m_staticText317, 0, wxALIGN_CENTER_VERTICAL|wxALIGN_RIGHT|wxALL, 5 );
	
	NumberofRefinementsText = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	NumberofRefinementsText->Wrap( -1 );
	gSizer12->Add( NumberofRefinementsText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	m_staticText212 = new wxStaticText( this, wxID_ANY, wxT("Last Refinement ID :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText212->Wrap( -1 );
	gSizer12->Add( m_staticText212, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	LastRefinementIDText = new wxStaticText( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
	LastRefinementIDText->Wrap( -1 );
	gSizer12->Add( LastRefinementIDText, 0, wxALL, 5 );
	
	
	bSizer192->Add( gSizer12, 1, wxEXPAND, 5 );
	
	
	bSizer187->Add( bSizer192, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer187 );
	this->Layout();
	
	// Connect Events
	this->Connect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( RefinementPackageAssetPanel::OnUpdateUI ) );
	CreateButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnCreateClick ), NULL, this );
	RenameButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnRenameClick ), NULL, this );
	DeleteButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnDeleteClick ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseCheckPackagesVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_LEFT_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseCheckPackagesVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_LEFT_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_COMMAND_LIST_BEGIN_LABEL_EDIT, wxListEventHandler( RefinementPackageAssetPanel::OnBeginEdit ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_COMMAND_LIST_END_LABEL_EDIT, wxListEventHandler( RefinementPackageAssetPanel::OnEndEdit ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( RefinementPackageAssetPanel::OnPackageActivated ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_COMMAND_LIST_ITEM_FOCUSED, wxListEventHandler( RefinementPackageAssetPanel::OnPackageFocusChange ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_MIDDLE_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_MOTION, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Connect( wxEVT_RIGHT_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseCheckParticlesVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_LEFT_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseCheckParticlesVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_LEFT_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_MIDDLE_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_MOTION, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Connect( wxEVT_RIGHT_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	DisplayStackButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnDisplayStackButton ), NULL, this );
	Active3DReferencesListCtrl->Connect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( RefinementPackageAssetPanel::OnVolumeListItemActivated ), NULL, this );
}

RefinementPackageAssetPanel::~RefinementPackageAssetPanel()
{
	// Disconnect Events
	this->Disconnect( wxEVT_UPDATE_UI, wxUpdateUIEventHandler( RefinementPackageAssetPanel::OnUpdateUI ) );
	CreateButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnCreateClick ), NULL, this );
	RenameButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnRenameClick ), NULL, this );
	DeleteButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnDeleteClick ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseCheckPackagesVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_LEFT_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseCheckPackagesVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_LEFT_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_COMMAND_LIST_BEGIN_LABEL_EDIT, wxListEventHandler( RefinementPackageAssetPanel::OnBeginEdit ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_COMMAND_LIST_END_LABEL_EDIT, wxListEventHandler( RefinementPackageAssetPanel::OnEndEdit ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( RefinementPackageAssetPanel::OnPackageActivated ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_COMMAND_LIST_ITEM_FOCUSED, wxListEventHandler( RefinementPackageAssetPanel::OnPackageFocusChange ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_MIDDLE_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_MOTION, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	RefinementPackageListCtrl->Disconnect( wxEVT_RIGHT_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_LEFT_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseCheckParticlesVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_LEFT_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseCheckParticlesVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_LEFT_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_MIDDLE_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_MIDDLE_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_MIDDLE_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_MOTION, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_RIGHT_DCLICK, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	ContainedParticlesListCtrl->Disconnect( wxEVT_RIGHT_UP, wxMouseEventHandler( RefinementPackageAssetPanel::MouseVeto ), NULL, this );
	DisplayStackButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( RefinementPackageAssetPanel::OnDisplayStackButton ), NULL, this );
	Active3DReferencesListCtrl->Disconnect( wxEVT_COMMAND_LIST_ITEM_ACTIVATED, wxListEventHandler( RefinementPackageAssetPanel::OnVolumeListItemActivated ), NULL, this );
	
}

TemplateWizardPanel::TemplateWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer153;
	bSizer153 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer147;
	bSizer147 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText214 = new wxStaticText( this, wxID_ANY, wxT("Template Refinement Package :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText214->Wrap( -1 );
	bSizer147->Add( m_staticText214, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	GroupComboBox = new wxComboBox( this, wxID_ANY, wxT("Combo!"), wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	bSizer147->Add( GroupComboBox, 1, wxALL, 5 );
	
	
	bSizer153->Add( bSizer147, 0, wxEXPAND, 5 );
	
	
	bSizer153->Add( 0, 0, 1, wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Select \"New Refinement Package\" to create a package from scratch.  Select \"Create From 2D Class Average Selection\" to create a new package based on one or more selections of 2D class averages. Alternatively, an existing refinement package can be used as a template.  Template based refinement packages will have the same particle stack as their template, but you will be able to change the other parameters, you may wish to do this to change the number of classes or symmetry for example."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer153->Add( InfoText, 0, wxALL|wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer153 );
	this->Layout();
}

TemplateWizardPanel::~TemplateWizardPanel()
{
}

InputParameterWizardPanel::InputParameterWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer153;
	bSizer153 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer147;
	bSizer147 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText214 = new wxStaticText( this, wxID_ANY, wxT("Input Parameters :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText214->Wrap( -1 );
	bSizer147->Add( m_staticText214, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	GroupComboBox = new wxComboBox( this, wxID_ANY, wxT("Combo!"), wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	bSizer147->Add( GroupComboBox, 1, wxALL, 5 );
	
	
	bSizer153->Add( bSizer147, 0, wxEXPAND, 5 );
	
	
	bSizer153->Add( 0, 0, 1, wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Please select the parameters to use to create the new refinement package."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer153->Add( InfoText, 0, wxALL|wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer153 );
	this->Layout();
}

InputParameterWizardPanel::~InputParameterWizardPanel()
{
}

ClassSelectionWizardPanel::ClassSelectionWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer153;
	bSizer153 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText416 = new wxStaticText( this, wxID_ANY, wxT("Select class average selections to use :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText416->Wrap( -1 );
	bSizer153->Add( m_staticText416, 0, wxALL, 5 );
	
	wxBoxSizer* bSizer278;
	bSizer278 = new wxBoxSizer( wxHORIZONTAL );
	
	SelectionListCtrl = new wxListCtrl( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLC_EDIT_LABELS|wxLC_REPORT );
	bSizer278->Add( SelectionListCtrl, 1, wxALL|wxEXPAND, 5 );
	
	
	bSizer153->Add( bSizer278, 1, wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("If you have performed any 2D classifications, then you can use one or more selections of class averages (selected from the classification results panel) to copy over ONLY the particles in the selected class averages to the new refinement package.  This may be done to clean the dataset for example."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer153->Add( InfoText, 0, wxALL|wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer153 );
	this->Layout();
}

ClassSelectionWizardPanel::~ClassSelectionWizardPanel()
{
}

SymmetryWizardPanel::SymmetryWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer153;
	bSizer153 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer147;
	bSizer147 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText214 = new wxStaticText( this, wxID_ANY, wxT("Wanted Pointgroup Symmetry :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText214->Wrap( -1 );
	bSizer147->Add( m_staticText214, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	SymmetryComboBox = new wxComboBox( this, wxID_ANY, wxT("C1"), wxDefaultPosition, wxDefaultSize, 0, NULL, 0 ); 
	bSizer147->Add( SymmetryComboBox, 1, wxALL, 5 );
	
	
	bSizer153->Add( bSizer147, 0, wxEXPAND, 5 );
	
	
	bSizer153->Add( 0, 0, 1, wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Please select the pointgroup symmetry to use for the refinement.  Whilst a number of symmetries can be selected from the menu for convenience, you may enter any valid pointgroup symmetry."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer153->Add( InfoText, 0, wxALL|wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer153 );
	this->Layout();
}

SymmetryWizardPanel::~SymmetryWizardPanel()
{
}

MolecularWeightWizardPanel::MolecularWeightWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer153;
	bSizer153 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer147;
	bSizer147 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText214 = new wxStaticText( this, wxID_ANY, wxT("Estimated Molecular Weight (kDa) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText214->Wrap( -1 );
	bSizer147->Add( m_staticText214, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	MolecularWeightTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("300.00"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer147->Add( MolecularWeightTextCtrl, 1, wxALL, 5 );
	
	
	bSizer153->Add( bSizer147, 0, wxEXPAND, 5 );
	
	
	bSizer153->Add( 0, 0, 1, wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Please enter the estimated molecular weight for the refinement.  In general, this should be the molecular weight of the coherent parts (e.g. micelle would not be included)."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer153->Add( InfoText, 0, wxALL|wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer153 );
	this->Layout();
}

MolecularWeightWizardPanel::~MolecularWeightWizardPanel()
{
}

LargestDimensionWizardPanel::LargestDimensionWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer153;
	bSizer153 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer147;
	bSizer147 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText214 = new wxStaticText( this, wxID_ANY, wxT("Estimated Largest Dimension (Å) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText214->Wrap( -1 );
	bSizer147->Add( m_staticText214, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	LargestDimensionTextCtrl = new NumericTextCtrl( this, wxID_ANY, wxT("150.00"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer147->Add( LargestDimensionTextCtrl, 1, wxALL, 5 );
	
	
	bSizer153->Add( bSizer147, 0, wxEXPAND, 5 );
	
	
	bSizer153->Add( 0, 0, 1, wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Please enter the estimated largest dimension of the particle. This will be used for autosizing masks and shift limits, it need not be exact, and it is better to err on the large size."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer153->Add( InfoText, 0, wxALL|wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer153 );
	this->Layout();
}

LargestDimensionWizardPanel::~LargestDimensionWizardPanel()
{
}

ParticleGroupWizardPanel::ParticleGroupWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer1531;
	bSizer1531 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer1471;
	bSizer1471 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText2141 = new wxStaticText( this, wxID_ANY, wxT("Particle Position Group :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText2141->Wrap( -1 );
	bSizer1471->Add( m_staticText2141, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	ParticlePositionsGroupComboBox = new wxComboBox( this, wxID_ANY, wxT("Combo!"), wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	bSizer1471->Add( ParticlePositionsGroupComboBox, 1, wxALL, 5 );
	
	
	bSizer1531->Add( bSizer1471, 0, wxEXPAND, 5 );
	
	
	bSizer1531->Add( 0, 0, 1, wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Please select a group of particle positions to use for the refinement package.  The particles will be \"cut\" from the active image assets using the active CTF estimation parameters.  From this point on the particle images and defocus parameters will be fixed, and changing the active results will not change the refinement package."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer1531->Add( InfoText, 0, wxALL|wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer1531 );
	this->Layout();
}

ParticleGroupWizardPanel::~ParticleGroupWizardPanel()
{
}

BoxSizeWizardPanel::BoxSizeWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer15311;
	bSizer15311 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer14711;
	bSizer14711 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText21411 = new wxStaticText( this, wxID_ANY, wxT("Box size for particles (pixels) :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText21411->Wrap( -1 );
	bSizer14711->Add( m_staticText21411, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	BoxSizeSpinCtrl = new wxSpinCtrl( this, wxID_ANY, wxT("512"), wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 2048, 0 );
	bSizer14711->Add( BoxSizeSpinCtrl, 1, wxALL, 5 );
	
	
	bSizer15311->Add( bSizer14711, 0, wxEXPAND, 5 );
	
	
	bSizer15311->Add( 0, 0, 1, wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Select the box size that should be used to extract the particles.  This box size should be big enough to not only contain the full particle, but also to take into account the delocalisation of information due to the CTF, as well as any aliasing of the CTF."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer15311->Add( InfoText, 0, wxALL|wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer15311 );
	this->Layout();
}

BoxSizeWizardPanel::~BoxSizeWizardPanel()
{
}

NumberofClassesWizardPanel::NumberofClassesWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer153111;
	bSizer153111 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer147111;
	bSizer147111 = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText214111 = new wxStaticText( this, wxID_ANY, wxT("Number of classes for refinement :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText214111->Wrap( -1 );
	bSizer147111->Add( m_staticText214111, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	NumberOfClassesSpinCtrl = new wxSpinCtrl( this, wxID_ANY, wxT("1"), wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 100, 0 );
	bSizer147111->Add( NumberOfClassesSpinCtrl, 1, wxALL, 5 );
	
	
	bSizer153111->Add( bSizer147111, 0, wxEXPAND, 5 );
	
	
	bSizer153111->Add( 0, 0, 1, wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Select the number of classes to use during the refinement.  This number is fixed for a refinment package, but can be changed by creating a new refinement package and using an old refinement package as a template.  In general you will start with 1 class, and split into more classes later in the refinement.If this package is based on a previous package, you will be given options about how to split/merge classes in the following steps."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	bSizer153111->Add( InfoText, 0, wxALL|wxEXPAND, 10 );
	
	
	this->SetSizer( bSizer153111 );
	this->Layout();
}

NumberofClassesWizardPanel::~NumberofClassesWizardPanel()
{
}

InitialReferenceSelectWizardPanel::InitialReferenceSelectWizardPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	MainSizer = new wxBoxSizer( wxVERTICAL );
	
	TitleText = new wxStaticText( this, wxID_ANY, wxT("Initial class references :-"), wxDefaultPosition, wxDefaultSize, 0 );
	TitleText->Wrap( -1 );
	MainSizer->Add( TitleText, 0, wxALL, 5 );
	
	ScrollWindow = new wxScrolledWindow( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxHSCROLL|wxVSCROLL );
	ScrollWindow->SetScrollRate( 5, 5 );
	GridSizer = new wxGridSizer( 0, 2, 0, 0 );
	
	
	ScrollWindow->SetSizer( GridSizer );
	ScrollWindow->Layout();
	GridSizer->Fit( ScrollWindow );
	MainSizer->Add( ScrollWindow, 1, wxALL|wxEXPAND, 5 );
	
	InfoText = new AutoWrapStaticText( this, wxID_ANY, wxT("Select the volume asset to use for the initial reference for each of the classes.  If \"Generate from params.\" is selected, a new volume asset will be calculated from the starting parameters at the start of the next refinement. In general, you will provide a reference for a new refinement package, and generate one when using a template."), wxDefaultPosition, wxDefaultSize, 0 );
	InfoText->Wrap( -1 );
	MainSizer->Add( InfoText, 0, wxALL|wxEXPAND, 10 );
	
	
	this->SetSizer( MainSizer );
	this->Layout();
}

InitialReferenceSelectWizardPanel::~InitialReferenceSelectWizardPanel()
{
}

ClassesSetupWizardPanel ::ClassesSetupWizardPanel ( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer160;
	bSizer160 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText232 = new wxStaticText( this, wxID_ANY, wxT("How should classes be created :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText232->Wrap( -1 );
	bSizer160->Add( m_staticText232, 0, wxALL, 5 );
	
	
	this->SetSizer( bSizer160 );
	this->Layout();
}

ClassesSetupWizardPanel ::~ClassesSetupWizardPanel ()
{
}

FSCPanel::FSCPanel( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer200;
	bSizer200 = new wxBoxSizer( wxVERTICAL );
	
	TitleSizer = new wxBoxSizer( wxHORIZONTAL );
	
	m_staticText280 = new wxStaticText( this, wxID_ANY, wxT("Select Class :"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText280->Wrap( -1 );
	TitleSizer->Add( m_staticText280, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	ClassComboBox = new wxComboBox( this, wxID_ANY, wxT("Combo!"), wxDefaultPosition, wxDefaultSize, 0, NULL, wxCB_READONLY ); 
	TitleSizer->Add( ClassComboBox, 0, wxALL, 5 );
	
	
	TitleSizer->Add( 0, 0, 1, wxEXPAND, 5 );
	
	EstimatedResolutionLabel = new wxStaticText( this, wxID_ANY, wxT("Estimated Resolution : "), wxDefaultPosition, wxDefaultSize, 0 );
	EstimatedResolutionLabel->Wrap( -1 );
	TitleSizer->Add( EstimatedResolutionLabel, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	EstimatedResolutionText = new wxStaticText( this, wxID_ANY, wxT("00.00  ‎Å"), wxDefaultPosition, wxDefaultSize, 0 );
	EstimatedResolutionText->Wrap( -1 );
	TitleSizer->Add( EstimatedResolutionText, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer200->Add( TitleSizer, 0, wxEXPAND, 5 );
	
	m_staticline52 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer200->Add( m_staticline52, 0, wxEXPAND | wxALL, 5 );
	
	PlotPanel = new PlotFSCPanel( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	bSizer200->Add( PlotPanel, 1, wxEXPAND | wxALL, 5 );
	
	
	this->SetSizer( bSizer200 );
	this->Layout();
	
	// Connect Events
	ClassComboBox->Connect( wxEVT_COMMAND_COMBOBOX_SELECTED, wxCommandEventHandler( FSCPanel::OnClassComboBoxChange ), NULL, this );
}

FSCPanel::~FSCPanel()
{
	// Disconnect Events
	ClassComboBox->Disconnect( wxEVT_COMMAND_COMBOBOX_SELECTED, wxCommandEventHandler( FSCPanel::OnClassComboBoxChange ), NULL, this );
	
}

DisplayPanelParent::DisplayPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	MainSizer = new wxBoxSizer( wxVERTICAL );
	
	Toolbar = new wxToolBar( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTB_FLAT|wxTB_HORIZONTAL ); 
	Toolbar->Realize(); 
	
	MainSizer->Add( Toolbar, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( MainSizer );
	this->Layout();
}

DisplayPanelParent::~DisplayPanelParent()
{
}

DisplayManualDialogParent::DisplayManualDialogParent( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	MainSizer = new wxBoxSizer( wxVERTICAL );
	
	
	MainSizer->Add( 400, 200, 0, wxFIXED_MINSIZE, 5 );
	
	m_staticline58 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainSizer->Add( m_staticline58, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer262;
	bSizer262 = new wxBoxSizer( wxHORIZONTAL );
	
	
	bSizer262->Add( 0, 0, 1, wxEXPAND, 5 );
	
	m_staticText315 = new wxStaticText( this, wxID_ANY, wxT("Min : "), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText315->Wrap( -1 );
	bSizer262->Add( m_staticText315, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	minimum_text_ctrl = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_CENTRE|wxTE_PROCESS_ENTER );
	bSizer262->Add( minimum_text_ctrl, 0, wxALL, 5 );
	
	m_staticText316 = new wxStaticText( this, wxID_ANY, wxT("/"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText316->Wrap( -1 );
	bSizer262->Add( m_staticText316, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	maximum_text_ctrl = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_CENTRE|wxTE_PROCESS_ENTER );
	bSizer262->Add( maximum_text_ctrl, 0, wxALL, 5 );
	
	m_staticText317 = new wxStaticText( this, wxID_ANY, wxT(": Max"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText317->Wrap( -1 );
	bSizer262->Add( m_staticText317, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer262->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	MainSizer->Add( bSizer262, 0, wxEXPAND, 5 );
	
	wxBoxSizer* bSizer265;
	bSizer265 = new wxBoxSizer( wxHORIZONTAL );
	
	
	bSizer265->Add( 0, 0, 1, wxEXPAND, 5 );
	
	Toolbar = new wxToolBar( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTB_HORIZONTAL ); 
	Toolbar->Realize(); 
	
	bSizer265->Add( Toolbar, 0, 0, 5 );
	
	
	bSizer265->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	MainSizer->Add( bSizer265, 0, wxEXPAND, 5 );
	
	m_staticline61 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainSizer->Add( m_staticline61, 0, wxEXPAND | wxALL, 5 );
	
	wxGridSizer* gSizer13;
	gSizer13 = new wxGridSizer( 0, 2, 0, 0 );
	
	histogram_checkbox = new wxCheckBox( this, wxID_ANY, wxT("Use Entire File For Histogram"), wxDefaultPosition, wxDefaultSize, 0 );
	gSizer13->Add( histogram_checkbox, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5 );
	
	live_checkbox = new wxCheckBox( this, wxID_ANY, wxT("Live Update of Display"), wxDefaultPosition, wxDefaultSize, 0 );
	live_checkbox->SetValue(true); 
	gSizer13->Add( live_checkbox, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5 );
	
	
	MainSizer->Add( gSizer13, 0, wxEXPAND, 5 );
	
	m_staticline63 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	MainSizer->Add( m_staticline63, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer264;
	bSizer264 = new wxBoxSizer( wxHORIZONTAL );
	
	
	bSizer264->Add( 0, 0, 1, wxEXPAND, 5 );
	
	m_button94 = new wxButton( this, wxID_ANY, wxT("OK"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer264->Add( m_button94, 0, wxALL, 5 );
	
	m_button95 = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer264->Add( m_button95, 0, wxALL, 5 );
	
	
	bSizer264->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	MainSizer->Add( bSizer264, 0, wxEXPAND, 5 );
	
	
	this->SetSizer( MainSizer );
	this->Layout();
	MainSizer->Fit( this );
	
	this->Centre( wxBOTH );
	
	// Connect Events
	this->Connect( wxEVT_CLOSE_WINDOW, wxCloseEventHandler( DisplayManualDialogParent::OnClose ) );
	this->Connect( wxEVT_LEFT_DOWN, wxMouseEventHandler( DisplayManualDialogParent::OnLeftDown ) );
	this->Connect( wxEVT_MOTION, wxMouseEventHandler( DisplayManualDialogParent::OnMotion ) );
	this->Connect( wxEVT_PAINT, wxPaintEventHandler( DisplayManualDialogParent::OnPaint ) );
	this->Connect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( DisplayManualDialogParent::OnRightDown ) );
	minimum_text_ctrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( DisplayManualDialogParent::OnLowChange ), NULL, this );
	maximum_text_ctrl->Connect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( DisplayManualDialogParent::OnHighChange ), NULL, this );
	histogram_checkbox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( DisplayManualDialogParent::OnHistogramCheck ), NULL, this );
	live_checkbox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( DisplayManualDialogParent::OnRealtimeCheck ), NULL, this );
	m_button94->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( DisplayManualDialogParent::OnButtonOK ), NULL, this );
	m_button95->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( DisplayManualDialogParent::OnButtonCancel ), NULL, this );
}

DisplayManualDialogParent::~DisplayManualDialogParent()
{
	// Disconnect Events
	this->Disconnect( wxEVT_CLOSE_WINDOW, wxCloseEventHandler( DisplayManualDialogParent::OnClose ) );
	this->Disconnect( wxEVT_LEFT_DOWN, wxMouseEventHandler( DisplayManualDialogParent::OnLeftDown ) );
	this->Disconnect( wxEVT_MOTION, wxMouseEventHandler( DisplayManualDialogParent::OnMotion ) );
	this->Disconnect( wxEVT_PAINT, wxPaintEventHandler( DisplayManualDialogParent::OnPaint ) );
	this->Disconnect( wxEVT_RIGHT_DOWN, wxMouseEventHandler( DisplayManualDialogParent::OnRightDown ) );
	minimum_text_ctrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( DisplayManualDialogParent::OnLowChange ), NULL, this );
	maximum_text_ctrl->Disconnect( wxEVT_COMMAND_TEXT_ENTER, wxCommandEventHandler( DisplayManualDialogParent::OnHighChange ), NULL, this );
	histogram_checkbox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( DisplayManualDialogParent::OnHistogramCheck ), NULL, this );
	live_checkbox->Disconnect( wxEVT_COMMAND_CHECKBOX_CLICKED, wxCommandEventHandler( DisplayManualDialogParent::OnRealtimeCheck ), NULL, this );
	m_button94->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( DisplayManualDialogParent::OnButtonOK ), NULL, this );
	m_button95->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( DisplayManualDialogParent::OnButtonCancel ), NULL, this );
	
}

ClassificationPlotPanelParent::ClassificationPlotPanelParent( wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style ) : wxPanel( parent, id, pos, size, style )
{
	wxBoxSizer* bSizer278;
	bSizer278 = new wxBoxSizer( wxVERTICAL );
	
	my_notebook = new wxAuiNotebook( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0 );
	MobilityPanel = new wxPanel( my_notebook, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	my_notebook->AddPage( MobilityPanel, wxT("Image Mobility"), true, wxNullBitmap );
	LikelihoodPanel = new wxPanel( my_notebook, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	my_notebook->AddPage( LikelihoodPanel, wxT("Likelihood"), false, wxNullBitmap );
	SigmaPanel = new wxPanel( my_notebook, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL );
	my_notebook->AddPage( SigmaPanel, wxT("Sigma"), false, wxNullBitmap );
	
	bSizer278->Add( my_notebook, 1, wxEXPAND | wxALL, 5 );
	
	
	this->SetSizer( bSizer278 );
	this->Layout();
}

ClassificationPlotPanelParent::~ClassificationPlotPanelParent()
{
}

ClassumSelectionCopyFromDialogParent::ClassumSelectionCopyFromDialogParent( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxDefaultSize, wxDefaultSize );
	
	wxBoxSizer* bSizer279;
	bSizer279 = new wxBoxSizer( wxVERTICAL );
	
	m_staticText414 = new wxStaticText( this, wxID_ANY, wxT("Please select a valid (same no. classums) selection :-"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText414->Wrap( -1 );
	bSizer279->Add( m_staticText414, 0, wxALL, 5 );
	
	m_staticline72 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer279->Add( m_staticline72, 0, wxEXPAND | wxALL, 5 );
	
	SelectionListCtrl = new wxListCtrl( this, wxID_ANY, wxDefaultPosition, wxSize( -1,100 ), wxLC_REPORT|wxLC_SINGLE_SEL );
	bSizer279->Add( SelectionListCtrl, 0, wxALL|wxEXPAND, 5 );
	
	m_staticline71 = new wxStaticLine( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
	bSizer279->Add( m_staticline71, 0, wxEXPAND | wxALL, 5 );
	
	wxBoxSizer* bSizer280;
	bSizer280 = new wxBoxSizer( wxHORIZONTAL );
	
	
	bSizer280->Add( 0, 0, 1, wxEXPAND, 5 );
	
	OkButton = new wxButton( this, wxID_ANY, wxT("OK"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer280->Add( OkButton, 0, wxALL, 5 );
	
	CancelButton = new wxButton( this, wxID_ANY, wxT("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer280->Add( CancelButton, 0, wxALL, 5 );
	
	
	bSizer280->Add( 0, 0, 1, wxEXPAND, 5 );
	
	
	bSizer279->Add( bSizer280, 1, wxEXPAND, 5 );
	
	
	this->SetSizer( bSizer279 );
	this->Layout();
	bSizer279->Fit( this );
	
	this->Centre( wxBOTH );
	
	// Connect Events
	OkButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ClassumSelectionCopyFromDialogParent::OnOKButtonClick ), NULL, this );
	CancelButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ClassumSelectionCopyFromDialogParent::OnCancelButtonClick ), NULL, this );
}

ClassumSelectionCopyFromDialogParent::~ClassumSelectionCopyFromDialogParent()
{
	// Disconnect Events
	OkButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ClassumSelectionCopyFromDialogParent::OnOKButtonClick ), NULL, this );
	CancelButton->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ClassumSelectionCopyFromDialogParent::OnCancelButtonClick ), NULL, this );
	
}

ErrorDialog::ErrorDialog( wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos, const wxSize& size, long style ) : wxDialog( parent, id, title, pos, size, style )
{
	this->SetSizeHints( wxSize( -1,500 ), wxDefaultSize );
	
	wxBoxSizer* bSizer35;
	bSizer35 = new wxBoxSizer( wxVERTICAL );
	
	wxBoxSizer* bSizer36;
	bSizer36 = new wxBoxSizer( wxHORIZONTAL );
	
	m_bitmap1 = new wxStaticBitmap( this, wxID_ANY, wxArtProvider::GetBitmap( wxART_ERROR, wxART_OTHER ), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer36->Add( m_bitmap1, 0, wxALL, 5 );
	
	m_staticText25 = new wxStaticText( this, wxID_ANY, wxT(": ERROR"), wxDefaultPosition, wxDefaultSize, 0 );
	m_staticText25->Wrap( -1 );
	bSizer36->Add( m_staticText25, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
	
	
	bSizer35->Add( bSizer36, 0, wxEXPAND, 5 );
	
	ErrorText = new wxTextCtrl( this, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE|wxTE_READONLY );
	bSizer35->Add( ErrorText, 100, wxALL|wxEXPAND, 5 );
	
	m_button23 = new wxButton( this, wxID_ANY, wxT("OK"), wxDefaultPosition, wxDefaultSize, 0 );
	bSizer35->Add( m_button23, 0, wxALIGN_RIGHT|wxALL, 5 );
	
	
	this->SetSizer( bSizer35 );
	this->Layout();
	
	this->Centre( wxBOTH );
	
	// Connect Events
	m_button23->Connect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ErrorDialog::OnClickOK ), NULL, this );
}

ErrorDialog::~ErrorDialog()
{
	// Disconnect Events
	m_button23->Disconnect( wxEVT_COMMAND_BUTTON_CLICKED, wxCommandEventHandler( ErrorDialog::OnClickOK ), NULL, this );
	
}
