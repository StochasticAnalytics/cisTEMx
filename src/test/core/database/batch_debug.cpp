#include "database_test_helpers.h"

using namespace DatabaseTestHelpers;

TEST_CASE("Database::BatchSelect debug WHY N-1 rows", "[database][debug]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    REQUIRE(CreateTestDatabase(db, test_db_path) == true);
    REQUIRE(CreateStandardTestTable(db) == true);
    PopulateTestTable(db, "TEST_TABLE", 3);

    // Verify we actually have 3 rows
    int actual_count = db.ReturnSingleIntFromSelectCommand("SELECT COUNT(*) FROM TEST_TABLE");
    REQUIRE(actual_count == 3);

    SECTION("test with pre-count pattern like production") {
        // Pattern from database.cpp:442-458
        int  expected_rows = actual_count;
        bool more_data     = db.BeginBatchSelect("SELECT ID, NAME, VALUE, COUNT FROM TEST_TABLE ORDER BY ID");

        REQUIRE(more_data == true); // Should have first row

        int rows_retrieved = 0;
        for ( int i = 0; i < expected_rows; i++ ) {
            REQUIRE(more_data == true); // Should always have data in counted loop

            int      id;
            wxString name;
            double   value;
            int      count;

            more_data = db.GetFromBatchSelect("itri", &id, &name, &value, &count);
            rows_retrieved++;

            // Verify data
            REQUIRE(id == (i + 1));
            REQUIRE(name == wxString::Format("item_%d", id));
        }

        db.EndBatchSelect( );
        REQUIRE(rows_retrieved == 3);
    }

    db.Close( );
}
