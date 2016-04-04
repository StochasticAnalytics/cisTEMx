class MyImageAssetPanel : public MyAssetParentPanel
{

	protected:

		void DirtyGroups() {main_frame->DirtyImageGroups();};

	public:

		MyImageAssetPanel( wxWindow* parent );
		~MyImageAssetPanel();

		void ImportAssetClick( wxCommandEvent& event );

		void RemoveAssetFromDatabase(long wanted_asset);
		void RemoveFromGroupInDatabase(int wanted_group_id, int wanted_asset_id);
		void InsertGroupMemberToDatabase(int wanted_group, int wanted_asset);
		void RemoveAllFromDatabase();
		void RemoveAllGroupMembersFromDatabase(int wanted_group_id);
		void AddGroupToDatabase(int wanted_group_id, const char * wanted_group_name, int wanted_list_id);
		void RemoveGroupFromDatabase(int wanted_group_id);
		void RenameGroupInDatabase(int wanted_group_id, const char *wanted_name);
		void ImportAllFromDatabase();
		void FillAssetSpecificContentsList();
		void UpdateInfo();

		bool IsFileAnAsset(wxFileName file_to_check);
		ImageAsset* ReturnAssetPointer(long wanted_asset);

};

