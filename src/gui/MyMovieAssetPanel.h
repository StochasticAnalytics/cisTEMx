class MyMovieAssetPanel : public MyAssetParentPanel
{

	private:


	public:

		MyMovieAssetPanel( wxWindow* parent );
		~MyMovieAssetPanel();

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

		double ReturnAssetPixelSize(long wanted_asset);
		double ReturnAssetAccelerationVoltage(long wanted_asset);
		double ReturnAssetDosePerFrame(long wanted_asset);
		double ReturnAssetPreExposureAmount(long wanted_asset);
		int ReturnAssetID(long wanted_asset);

};

