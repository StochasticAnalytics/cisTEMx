/*
 * Copyright (c) 2025, Stochastic Analytics, LLC
 *
 * This file is part of cisTEMx.
 *
 * For academic and non-profit use, this file is licensed under the
 * Mozilla Public License Version 2.0. See license_details/LICENSE-MPL-2.0.txt.
 *
 * Commercial use requires a separate license from Stochastic Analytics, LLC.
 * Contact: <commercial-license@clearnoise.org>
 */

#include "database_test_helpers.h"

using namespace DatabaseTestHelpers;

/**
 * Unit tests for Database lifecycle operations.
 *
 * Tests cover:
 * - CreateNewDatabase: Database file creation
 * - Open: Opening existing databases
 * - Close: Resource cleanup and state management
 * - CopyDatabaseFile: Backup functionality
 *
 * These are fundamental operations that all other database functionality depends on.
 */

// =============================================================================
// CreateNewDatabase Tests
// =============================================================================

TEST_CASE("Database::CreateNewDatabase creates valid database file", "[database][crud][lifecycle][.broken]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;

    SECTION("creates database file on filesystem") {
        REQUIRE(CreateTestDatabase(db, test_db_path) == true);
        REQUIRE(DatabaseFileExists(test_db_path) == true);
    }

    SECTION("sets is_open flag to true") {
        CreateTestDatabase(db, test_db_path);
        REQUIRE(db.is_open == true);
    }

    SECTION("stores database filename correctly") {
        CreateTestDatabase(db, test_db_path);
        REQUIRE(db.ReturnFilename( ) == test_db_path.GetFullPath( ));
    }

    SECTION("database is usable after creation") {
        CreateTestDatabase(db, test_db_path);
        // Should be able to create a table
        REQUIRE(CreateStandardTestTable(db) == true);
        REQUIRE(db.DoesTableExist("TEST_TABLE") == true);
    }

    if ( db.is_open )
        db.Close( );
}

TEST_CASE("Database::CreateNewDatabase handles edge cases", "[database][crud][lifecycle][.broken]") {
    SECTION("fails if directory doesn't exist") {
        Database   db;
        wxFileName invalid_path("/nonexistent/impossible/directory/test.db");
        REQUIRE(CreateTestDatabase(db, invalid_path) == false);
        REQUIRE(db.is_open == false);
    }

    SECTION("fails gracefully with empty path") {
        Database   db;
        wxFileName empty_path;
        REQUIRE(CreateTestDatabase(db, empty_path) == false);
        REQUIRE(db.is_open == false);
    }
}

// =============================================================================
// Open Tests
// =============================================================================

TEST_CASE("Database::Open opens existing database", "[database][crud][lifecycle][.broken]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    // Create a database with some content
    {
        Database db;
        CreateTestDatabase(db, test_db_path);
        CreateStandardTestTable(db);
        PopulateTestTable(db, "TEST_TABLE", 5);
        db.Close( );
    }

    SECTION("successfully opens existing database") {
        Database db;
        REQUIRE(db.Open(test_db_path) == true);
        REQUIRE(db.is_open == true);
        db.Close( );
    }

    SECTION("sets is_open flag correctly") {
        Database db;
        db.Open(test_db_path);
        REQUIRE(db.is_open == true);
        db.Close( );
    }

    SECTION("stores filename correctly") {
        Database db;
        db.Open(test_db_path);
        REQUIRE(db.ReturnFilename( ) == test_db_path.GetFullPath( ));
        db.Close( );
    }

    SECTION("can access existing data after open") {
        Database db;
        db.Open(test_db_path);

        REQUIRE(db.DoesTableExist("TEST_TABLE") == true);
        REQUIRE(TableHasExpectedRowCount(db, "TEST_TABLE", 5) == true);

        // Verify data integrity
        wxString name = db.ReturnSingleStringFromSelectCommand(
                "SELECT NAME FROM TEST_TABLE WHERE ID=1");
        REQUIRE(name == "item_1");

        db.Close( );
    }
}

TEST_CASE("Database::Open handles errors gracefully", "[database][crud][lifecycle][.broken]") {
    SECTION("fails for non-existent database") {
        Database   db;
        wxFileName fake_path(wxFileName::GetTempDir( ) + "/nonexistent_db_12345.db");
        REQUIRE(db.Open(fake_path) == false);
        REQUIRE(db.is_open == false);
    }

    SECTION("fails with empty path") {
        Database   db;
        wxFileName empty_path;
        REQUIRE(db.Open(empty_path) == false);
        REQUIRE(db.is_open == false);
    }
}

TEST_CASE("Database::Open with disable_locking option", "[database][crud][lifecycle][.broken]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db1;
    CreateTestDatabase(db1, test_db_path);
    db1.Close( );

    SECTION("opens with locking disabled when requested") {
        Database db;
        REQUIRE(db.Open(test_db_path, true) == true); // disable_locking = true
        REQUIRE(db.is_open == true);
        db.Close( );
    }

    SECTION("default behavior enables locking") {
        Database db;
        REQUIRE(db.Open(test_db_path) == true); // disable_locking = false (default)
        REQUIRE(db.is_open == true);
        db.Close( );
    }
}

// =============================================================================
// Close Tests
// =============================================================================

TEST_CASE("Database::Close cleans up resources", "[database][crud][lifecycle][.broken]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    SECTION("sets is_open to false") {
        Database db;
        CreateTestDatabase(db, test_db_path);
        REQUIRE(db.is_open == true);

        db.Close( );
        REQUIRE(db.is_open == false);
    }

    SECTION("can be called multiple times safely") {
        Database db;
        CreateTestDatabase(db, test_db_path);

        db.Close( );
        REQUIRE(db.is_open == false);

        // Second close should not crash
        db.Close( );
        REQUIRE(db.is_open == false);
    }

    SECTION("can be called on never-opened database") {
        Database db;
        REQUIRE(db.is_open == false);

        // Should not crash
        db.Close( );
        REQUIRE(db.is_open == false);
    }
}

TEST_CASE("Database::Close with remove_lock parameter", "[database][crud][lifecycle][.broken]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    SECTION("default behavior removes lock") {
        Database db;
        CreateTestDatabase(db, test_db_path);
        db.Close( ); // remove_lock = true (default)
        REQUIRE(db.is_open == false);
    }

    SECTION("remove_lock=false preserves lock") {
        Database db;
        CreateTestDatabase(db, test_db_path);
        db.Close(false); // remove_lock = false
        REQUIRE(db.is_open == false);
    }
}

TEST_CASE("Database lifecycle: create-open-close cycle", "[database][crud][lifecycle][.broken]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    SECTION("data persists across open-close cycles") {
        // Create and populate
        {
            Database db;
            CreateTestDatabase(db, test_db_path);
            CreateStandardTestTable(db);
            db.InsertOrReplace("TEST_TABLE", "ptri",
                               "ID", "NAME", "VALUE", "COUNT",
                               42, "persistent", 3.14, 100);
            db.Close( );
        }

        // Open and verify
        {
            Database db;
            db.Open(test_db_path);

            int id = db.ReturnSingleIntFromSelectCommand(
                    "SELECT ID FROM TEST_TABLE WHERE NAME='persistent'");
            REQUIRE(id == 42);

            double value = db.ReturnSingleDoubleFromSelectCommand(
                    "SELECT VALUE FROM TEST_TABLE WHERE ID=42");
            REQUIRE(value == Approx(3.14));

            db.Close( );
        }

        // Reopen and modify
        {
            Database db;
            db.Open(test_db_path);

            db.InsertOrReplace("TEST_TABLE", "ptri",
                               "ID", "NAME", "VALUE", "COUNT",
                               42, "modified", 2.71, 200);
            db.Close( );
        }

        // Final verification
        {
            Database db;
            db.Open(test_db_path);

            wxString name = db.ReturnSingleStringFromSelectCommand(
                    "SELECT NAME FROM TEST_TABLE WHERE ID=42");
            REQUIRE(name == "modified");

            db.Close( );
        }
    }
}

// =============================================================================
// CopyDatabaseFile Tests
// =============================================================================

TEST_CASE("Database::CopyDatabaseFile creates backup", "[database][crud][lifecycle][.broken]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    wxFileName           backup_path(wxFileName::GetTempDir( ) + "/cistem_backup_" +
                                     wxString::Format("%ld.db", wxGetUTCTime( )));
    DatabaseCleanupGuard cleanup1(test_db_path);
    DatabaseCleanupGuard cleanup2(backup_path);

    // Create source database with data
    Database db;
    CreateTestDatabase(db, test_db_path);
    CreateStandardTestTable(db);
    PopulateTestTable(db, "TEST_TABLE", 10);

    SECTION("creates backup file") {
        REQUIRE(db.CopyDatabaseFile(backup_path) == true);
        REQUIRE(DatabaseFileExists(backup_path) == true);
    }

    SECTION("backup contains same data as original") {
        db.CopyDatabaseFile(backup_path);
        db.Close( );

        // Verify original
        Database db_original;
        db_original.Open(test_db_path);
        int original_count = db_original.ReturnSingleIntFromSelectCommand(
                "SELECT COUNT(*) FROM TEST_TABLE");
        db_original.Close( );

        // Verify backup
        Database db_backup;
        db_backup.Open(backup_path);
        int backup_count = db_backup.ReturnSingleIntFromSelectCommand(
                "SELECT COUNT(*) FROM TEST_TABLE");
        REQUIRE(backup_count == original_count);
        REQUIRE(backup_count == 10);
        db_backup.Close( );
    }

    SECTION("backup is independent of original") {
        db.CopyDatabaseFile(backup_path);
        db.Close( );

        // Modify original
        Database db_original;
        db_original.Open(test_db_path);
        db_original.InsertOrReplace("TEST_TABLE", "ptri",
                                    "ID", "NAME", "VALUE", "COUNT",
                                    99, "new_data", 9.99, 999);
        db_original.Close( );

        // Backup should not have new data
        Database db_backup;
        db_backup.Open(backup_path);
        REQUIRE(TableHasExpectedRowCount(db_backup, "TEST_TABLE", 10) == true);
        REQUIRE(RowExists(db_backup, "TEST_TABLE", "ID=99") == false);
        db_backup.Close( );
    }

    if ( db.is_open )
        db.Close( );
}

TEST_CASE("Database::CopyDatabaseFile handles overwrite", "[database][crud][lifecycle][.broken]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    wxFileName           backup_path(wxFileName::GetTempDir( ) + "/cistem_backup_" +
                                     wxString::Format("%ld.db", wxGetUTCTime( )));
    DatabaseCleanupGuard cleanup1(test_db_path);
    DatabaseCleanupGuard cleanup2(backup_path);

    SECTION("overwrites existing backup file") {
        // Create original database
        Database db;
        CreateTestDatabase(db, test_db_path);
        CreateStandardTestTable(db);
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           1, "first", 1.0, 1);

        // First backup
        db.CopyDatabaseFile(backup_path);

        // Modify original
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           2, "second", 2.0, 2);

        // Second backup (should overwrite)
        REQUIRE(db.CopyDatabaseFile(backup_path) == true);

        db.Close( );

        // Verify backup has latest data
        Database db_backup;
        db_backup.Open(backup_path);
        REQUIRE(TableHasExpectedRowCount(db_backup, "TEST_TABLE", 2) == true);
        REQUIRE(RowExists(db_backup, "TEST_TABLE", "ID=2 AND NAME='second'") == true);
        db_backup.Close( );
    }
}

TEST_CASE("Database::CopyDatabaseFile error cases", "[database][crud][lifecycle][.broken]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    CreateTestDatabase(db, test_db_path);

    SECTION("fails with invalid destination path") {
        wxFileName invalid_path("/nonexistent/directory/backup.db");
        REQUIRE(db.CopyDatabaseFile(invalid_path) == false);
    }

    db.Close( );
}
