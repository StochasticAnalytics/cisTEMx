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
 * Unit tests for Database table operations.
 *
 * Tests cover:
 * - CreateTable: Table creation with format strings and vector overload
 * - DeleteTable: Table removal
 * - DoesTableExist: Table existence checks
 * - DoesColumnExist: Column existence checks
 * - AddColumnToTable: Schema modification
 *
 * These operations manage database schema and are critical for
 * ensuring correct table structure.
 */

// =============================================================================
// CreateTable Tests (Variadic Version)
// =============================================================================

TEST_CASE("Database::CreateTable with format strings", "[database][crud][tables][.broken]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    REQUIRE(CreateTestDatabase(db, test_db_path) == true);

    SECTION("creates table with primary key ('p')") {
        REQUIRE(db.CreateTable("TEST_PK", "p", "ID") == true);
        REQUIRE(db.DoesTableExist("TEST_PK") == true);
        REQUIRE(db.DoesColumnExist("TEST_PK", "ID") == true);
    }

    SECTION("creates table with text column ('t')") {
        REQUIRE(db.CreateTable("TEST_TEXT", "pt", "ID", "NAME") == true);
        REQUIRE(db.DoesColumnExist("TEST_TEXT", "NAME") == true);

        // Verify can insert text
        db.InsertOrReplace("TEST_TEXT", "pt", "ID", "NAME", 1, "test_string");
        wxString name = db.ReturnSingleStringFromSelectCommand(
                "SELECT NAME FROM TEST_TEXT WHERE ID=1");
        REQUIRE(name == "test_string");
    }

    SECTION("creates table with real column ('r')") {
        REQUIRE(db.CreateTable("TEST_REAL", "pr", "ID", "VALUE") == true);
        REQUIRE(db.DoesColumnExist("TEST_REAL", "VALUE") == true);

        // Verify can insert real
        db.InsertOrReplace("TEST_REAL", "pr", "ID", "VALUE", 1, 3.14159);
        double value = db.ReturnSingleDoubleFromSelectCommand(
                "SELECT VALUE FROM TEST_REAL WHERE ID=1");
        REQUIRE(value == Approx(3.14159));
    }

    SECTION("creates table with integer column ('i')") {
        REQUIRE(db.CreateTable("TEST_INT", "pi", "ID", "COUNT") == true);
        REQUIRE(db.DoesColumnExist("TEST_INT", "COUNT") == true);

        // Verify can insert integer
        db.InsertOrReplace("TEST_INT", "pi", "ID", "COUNT", 1, 42);
        int count = db.ReturnSingleIntFromSelectCommand(
                "SELECT COUNT FROM TEST_INT WHERE ID=1");
        REQUIRE(count == 42);
    }

    SECTION("creates table with long/integer column ('l')") {
        REQUIRE(db.CreateTable("TEST_LONG", "pl", "ID", "BIG_VALUE") == true);
        REQUIRE(db.DoesColumnExist("TEST_LONG", "BIG_VALUE") == true);

        // Verify can insert long
        long big_value = 9999999L;
        db.InsertOrReplace("TEST_LONG", "pl", "ID", "BIG_VALUE", 1, big_value);
        long retrieved = db.ReturnSingleLongFromSelectCommand(
                "SELECT BIG_VALUE FROM TEST_LONG WHERE ID=1");
        REQUIRE(retrieved == big_value);
    }

    SECTION("creates table with mixed column types") {
        REQUIRE(db.CreateTable("MIXED", "ptri",
                               "ID", "NAME", "VALUE", "COUNT") == true);

        REQUIRE(db.DoesColumnExist("MIXED", "ID") == true);
        REQUIRE(db.DoesColumnExist("MIXED", "NAME") == true);
        REQUIRE(db.DoesColumnExist("MIXED", "VALUE") == true);
        REQUIRE(db.DoesColumnExist("MIXED", "COUNT") == true);

        // Verify mixed data insertion
        db.InsertOrReplace("MIXED", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           1, "test", 2.5, 100);
        REQUIRE(TableHasExpectedRowCount(db, "MIXED", 1) == true);
    }

    SECTION("handles uppercase 'P' for primary key") {
        REQUIRE(db.CreateTable("TEST_UPPER_PK", "P", "ID") == true);
        REQUIRE(db.DoesTableExist("TEST_UPPER_PK") == true);
    }

    db.Close( );
}

// =============================================================================
// CreateTable Tests (Vector Version)
// =============================================================================

TEST_CASE("Database::CreateTable with vector columns", "[database][crud][tables][.broken]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    REQUIRE(CreateTestDatabase(db, test_db_path) == true);

    SECTION("creates table with vector of column names") {
        std::vector<wxString> columns = {"ID", "NAME", "VALUE", "COUNT"};
        REQUIRE(db.CreateTable("VECTOR_TABLE", "ptri", columns) == true);
        REQUIRE(db.DoesTableExist("VECTOR_TABLE") == true);

        // Verify all columns exist
        for ( const auto& col : columns ) {
            REQUIRE(db.DoesColumnExist("VECTOR_TABLE", col) == true);
        }
    }

    db.Close( );
}

// =============================================================================
// DeleteTable Tests
// =============================================================================

TEST_CASE("Database::DeleteTable removes table", "[database][crud][tables][.broken]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    REQUIRE(CreateTestDatabase(db, test_db_path) == true);

    SECTION("deletes existing table") {
        // Create table
        REQUIRE(db.CreateTable("TO_DELETE", "pt", "ID", "NAME") == true);
        REQUIRE(db.DoesTableExist("TO_DELETE") == true);

        // Delete table (return value not checked - production code never checks it)
        db.DeleteTable("TO_DELETE");
        REQUIRE(db.DoesTableExist("TO_DELETE") == false);
    }

    SECTION("deleted table data is gone") {
        // Create and populate
        db.CreateTable("DATA_TABLE", "pi", "ID", "VALUE");
        db.InsertOrReplace("DATA_TABLE", "pi", "ID", "VALUE", 1, 100);
        REQUIRE(TableHasExpectedRowCount(db, "DATA_TABLE", 1) == true);

        // Delete
        db.DeleteTable("DATA_TABLE");

        // Recreate with same name
        db.CreateTable("DATA_TABLE", "pi", "ID", "VALUE");
        REQUIRE(TableHasExpectedRowCount(db, "DATA_TABLE", 0) == true);
    }

    SECTION("can recreate deleted table") {
        db.CreateTable("RECREATE", "p", "ID");
        db.DeleteTable("RECREATE");

        // Should be able to create again
        REQUIRE(db.CreateTable("RECREATE", "p", "ID") == true);
        REQUIRE(db.DoesTableExist("RECREATE") == true);
    }

    db.Close( );
}

TEST_CASE("Database::DeleteTable handles non-existent tables", "[database][crud][tables][.broken]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    REQUIRE(CreateTestDatabase(db, test_db_path) == true);

    SECTION("fails gracefully for non-existent table") {
        // Should not crash, may return false or true depending on implementation
        db.DeleteTable("NONEXISTENT_TABLE");
        REQUIRE(db.DoesTableExist("NONEXISTENT_TABLE") == false);
    }

    db.Close( );
}

// =============================================================================
// DoesTableExist Tests
// =============================================================================

TEST_CASE("Database::DoesTableExist checks table existence", "[database][crud][tables][.broken]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    REQUIRE(CreateTestDatabase(db, test_db_path) == true); // Verify database creation succeeds

    SECTION("returns true for existing table") {
        db.CreateTable("TEST_TABLE", "p", "ID");
        REQUIRE(db.DoesTableExist("TEST_TABLE") == true);
    }

    SECTION("returns false for non-existent table") {
        REQUIRE(db.DoesTableExist("DOES_NOT_EXIST") == false);
    }

    SECTION("handles empty table name") {
        REQUIRE(db.DoesTableExist("") == false);
    }

    SECTION("detects table after deletion") {
        db.CreateTable("TEMP", "p", "ID");
        REQUIRE(db.DoesTableExist("TEMP") == true);

        db.DeleteTable("TEMP");
        REQUIRE(db.DoesTableExist("TEMP") == false);
    }

    db.Close( );
}

// =============================================================================
// DoesColumnExist Tests
// =============================================================================

TEST_CASE("Database::DoesColumnExist checks column existence", "[database][crud][tables][.broken]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    REQUIRE(CreateTestDatabase(db, test_db_path) == true);
    db.CreateTable("TEST_COLS", "ptri", "ID", "NAME", "VALUE", "COUNT");

    SECTION("returns true for existing column") {
        REQUIRE(db.DoesColumnExist("TEST_COLS", "ID") == true);
        REQUIRE(db.DoesColumnExist("TEST_COLS", "NAME") == true);
        REQUIRE(db.DoesColumnExist("TEST_COLS", "VALUE") == true);
        REQUIRE(db.DoesColumnExist("TEST_COLS", "COUNT") == true);
    }

    SECTION("returns false for non-existent column") {
        REQUIRE(db.DoesColumnExist("TEST_COLS", "NONEXISTENT") == false);
    }

    SECTION("returns false if table doesn't exist") {
        REQUIRE(db.DoesColumnExist("FAKE_TABLE", "ID") == false);
    }

    SECTION("handles empty column name") {
        REQUIRE(db.DoesColumnExist("TEST_COLS", "") == false);
    }

    db.Close( );
}

// =============================================================================
// AddColumnToTable Tests
// =============================================================================

TEST_CASE("Database::AddColumnToTable modifies schema", "[database][crud][tables][.broken]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    REQUIRE(CreateTestDatabase(db, test_db_path) == true);
    REQUIRE(db.CreateTable("EVOLVING_TABLE", "pt", "ID", "FIELD_NAME") == true);

    SECTION("adds new TEXT column") {
        REQUIRE(db.AddColumnToTable("EVOLVING_TABLE", "DESCRIPTION", "t", "NULL") == true);
        REQUIRE(db.DoesColumnExist("EVOLVING_TABLE", "DESCRIPTION") == true);
    }

    SECTION("adds new INTEGER column") {
        REQUIRE(db.AddColumnToTable("EVOLVING_TABLE", "AGE", "i", "0") == true);
        REQUIRE(db.DoesColumnExist("EVOLVING_TABLE", "AGE") == true);
    }

    SECTION("adds new REAL column") {
        REQUIRE(db.AddColumnToTable("EVOLVING_TABLE", "SCORE", "r", "0.0") == true);
        REQUIRE(db.DoesColumnExist("EVOLVING_TABLE", "SCORE") == true);
    }

    SECTION("new column has default value for existing rows") {
        // Insert data before adding column
        db.InsertOrReplace("EVOLVING_TABLE", "pt", "ID", "FIELD_NAME", 1, "existing");

        // Add column with default
        db.AddColumnToTable("EVOLVING_TABLE", "STATUS", "t", "'pending'");

        // Verify existing row has default value
        wxString status = db.ReturnSingleStringFromSelectCommand(
                "SELECT STATUS FROM EVOLVING_TABLE WHERE ID=1");
        REQUIRE(status == "pending");
    }

    SECTION("new rows can set column value") {
        db.AddColumnToTable("EVOLVING_TABLE", "EXTRA", "i", "0");

        // Can't use InsertOrReplace with old format - need ExecuteSQL
        db.ExecuteSQL("INSERT INTO EVOLVING_TABLE (ID, FIELD_NAME, EXTRA) VALUES (2, 'new', 42)");

        int extra = db.ReturnSingleIntFromSelectCommand(
                "SELECT EXTRA FROM EVOLVING_TABLE WHERE ID=2");
        REQUIRE(extra == 42);
    }

    db.Close( );
}

TEST_CASE("Database::AddColumnToTable preserves existing data", "[database][crud][tables][.broken]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    REQUIRE(CreateTestDatabase(db, test_db_path) == true);

    SECTION("existing data remains intact after adding column") {
        // Create and populate table
        db.CreateTable("STABLE", "pti", "ID", "NAME", "VALUE");
        db.InsertOrReplace("STABLE", "pti", "ID", "NAME", "VALUE", 1, "first", 100);
        db.InsertOrReplace("STABLE", "pti", "ID", "NAME", "VALUE", 2, "second", 200);
        db.InsertOrReplace("STABLE", "pti", "ID", "NAME", "VALUE", 3, "third", 300);

        REQUIRE(TableHasExpectedRowCount(db, "STABLE", 3) == true);

        // Add new column
        db.AddColumnToTable("STABLE", "NEW_FIELD", "t", "'default'");

        // Verify row count unchanged
        REQUIRE(TableHasExpectedRowCount(db, "STABLE", 3) == true);

        // Verify original data intact
        wxString name = db.ReturnSingleStringFromSelectCommand(
                "SELECT NAME FROM STABLE WHERE ID=2");
        REQUIRE(name == "second");

        int value = db.ReturnSingleIntFromSelectCommand(
                "SELECT VALUE FROM STABLE WHERE ID=3");
        REQUIRE(value == 300);
    }

    db.Close( );
}

// =============================================================================
// Integration: Table Lifecycle
// =============================================================================

TEST_CASE("Table lifecycle: create-populate-modify-delete", "[database][crud][tables][.broken]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    REQUIRE(CreateTestDatabase(db, test_db_path) == true);

    SECTION("complete table lifecycle") {
        // Create (use format that matches PopulateTestTable)
        REQUIRE(db.CreateTable("LIFECYCLE", "ptri", "ID", "NAME", "VALUE", "COUNT") == true);
        REQUIRE(db.DoesTableExist("LIFECYCLE") == true);

        // Populate
        PopulateTestTable(db, "LIFECYCLE", 5);
        REQUIRE(TableHasExpectedRowCount(db, "LIFECYCLE", 5) == true);

        // Modify schema
        db.AddColumnToTable("LIFECYCLE", "TIMESTAMP", "i", "0");
        REQUIRE(db.DoesColumnExist("LIFECYCLE", "TIMESTAMP") == true);

        // Verify data persists
        REQUIRE(TableHasExpectedRowCount(db, "LIFECYCLE", 5) == true);

        // Delete
        db.DeleteTable("LIFECYCLE");
        REQUIRE(db.DoesTableExist("LIFECYCLE") == false);
    }

    db.Close( );
}
