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

class MyParticlePositionAssetPanel : public MyAssetPanelParent {

  protected:
    void DirtyGroups( ) { main_frame->DirtyParticlePositionGroups( ); };

  public:
    MyParticlePositionAssetPanel(wxWindow* parent);
    ~MyParticlePositionAssetPanel( );

    void ImportAssetClick(wxCommandEvent& event);
    void NewFromParentClick(wxCommandEvent& event);

    void EnableNewFromParentButton( );

    void RemoveAssetFromDatabase(long wanted_asset);
    void RemoveFromGroupInDatabase(int wanted_group_id, int wanted_asset_id);
    void InsertGroupMemberToDatabase(int wanted_group, int wanted_asset);
    void InsertArrayofGroupMembersToDatabase(long wanted_group, wxArrayLong* wanted_array, OneSecondProgressDialog* progress_dialog = NULL);
    void RemoveAllFromDatabase( );
    void RemoveAllGroupMembersFromDatabase(int wanted_group_id);
    void AddGroupToDatabase(int wanted_group_id, const char* wanted_group_name, int wanted_list_id);
    void RemoveGroupFromDatabase(int wanted_group_id);
    void RenameGroupInDatabase(int wanted_group_id, const char* wanted_name);
    void ImportAllFromDatabase( );
    void FillAssetSpecificContentsList( );
    void UpdateInfo( );

    void DisplaySelectedItems( ){ };

    void RemoveParticlePositionAssetsWithGivenParentImageID(long parent_image_id);

    void     RenameAsset(long wanted_asset, wxString wanted_name){ };
    wxString ReturnItemText(long item, long column) const;

    int  ShowDeleteMessageDialog( );
    int  ShowDeleteAllMessageDialog( );
    void CompletelyRemoveAsset(long wanted_asset);
    void DoAfterDeletionCleanup( );

    ParticlePositionAsset* ReturnAssetPointer(long wanted_asset);
};
