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
 * Unit tests for Database insert and execute operations.
 *
 * Tests cover:
 * - InsertOrReplace: UPSERT semantics (INSERT vs UPDATE)
 * - ExecuteSQL: Raw SQL execution for various operations
 *
 * These are the core write operations that modify database content.
 */

// =============================================================================
// InsertOrReplace Tests (INSERT Semantics)
// =============================================================================

TEST_CASE("Database::InsertOrReplace inserts new rows", "[database][crud][insert]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    CreateTestDatabase(db, test_db_path);
    CreateStandardTestTable(db);

    SECTION("inserts single row with all types") {
        REQUIRE(db.InsertOrReplace("TEST_TABLE", "ptri",
                                   "ID", "NAME", "VALUE", "COUNT",
                                   1, "first", 1.5, 10) == true);

        REQUIRE(TableHasExpectedRowCount(db, "TEST_TABLE", 1) == true);
        REQUIRE(RowExists(db, "TEST_TABLE", "ID=1") == true);
    }

    SECTION("inserts multiple rows with unique primary keys") {
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           1, "first", 1.0, 10);
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           2, "second", 2.0, 20);
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           3, "third", 3.0, 30);

        REQUIRE(TableHasExpectedRowCount(db, "TEST_TABLE", 3) == true);
    }

    SECTION("inserts with integer type") {
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           42, "answer", 0.0, 1000);

        int count = db.ReturnSingleIntFromSelectCommand(
                "SELECT COUNT FROM TEST_TABLE WHERE ID=42");
        REQUIRE(count == 1000);
    }

    SECTION("inserts with long type") {
        long big_id = 999999L;
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           big_id, "big", 0.0, 0);

        REQUIRE(RowExists(db, "TEST_TABLE", "ID=999999") == true);
    }

    SECTION("inserts with double/real type") {
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           1, "pi", 3.14159265, 0);

        double value = db.ReturnSingleDoubleFromSelectCommand(
                "SELECT VALUE FROM TEST_TABLE WHERE ID=1");
        REQUIRE(value == Approx(3.14159265));
    }

    SECTION("inserts with wxString type") {
        wxString name = "test_string_with_spaces";
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           1, name.ToUTF8( ).data( ), 0.0, 0);

        wxString retrieved = db.ReturnSingleStringFromSelectCommand(
                "SELECT NAME FROM TEST_TABLE WHERE ID=1");
        REQUIRE(retrieved == name);
    }

    db.Close( );
}

// =============================================================================
// InsertOrReplace Tests (REPLACE/UPDATE Semantics)
// =============================================================================

TEST_CASE("Database::InsertOrReplace replaces existing rows", "[database][crud][insert]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    CreateTestDatabase(db, test_db_path);
    CreateStandardTestTable(db);

    SECTION("replaces row with same primary key") {
        // Insert initial row
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           1, "original", 1.0, 10);

        // Replace with same ID
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           1, "updated", 2.0, 20);

        // Should still have only 1 row
        REQUIRE(TableHasExpectedRowCount(db, "TEST_TABLE", 1) == true);

        // Verify updated values
        wxString name = db.ReturnSingleStringFromSelectCommand(
                "SELECT NAME FROM TEST_TABLE WHERE ID=1");
        REQUIRE(name == "updated");

        double value = db.ReturnSingleDoubleFromSelectCommand(
                "SELECT VALUE FROM TEST_TABLE WHERE ID=1");
        REQUIRE(value == Approx(2.0));
    }

    SECTION("update semantics - overwrites all columns") {
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           5, "first", 1.0, 100);

        // Update with different values
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           5, "second", 9.99, 999);

        // All columns should be updated
        wxString name = db.ReturnSingleStringFromSelectCommand(
                "SELECT NAME FROM TEST_TABLE WHERE ID=5");
        REQUIRE(name == "second");

        double value = db.ReturnSingleDoubleFromSelectCommand(
                "SELECT VALUE FROM TEST_TABLE WHERE ID=5");
        REQUIRE(value == Approx(9.99));

        int count = db.ReturnSingleIntFromSelectCommand(
                "SELECT COUNT FROM TEST_TABLE WHERE ID=5");
        REQUIRE(count == 999);
    }

    SECTION("multiple updates to same row") {
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           1, "v1", 1.0, 1);
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           1, "v2", 2.0, 2);
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           1, "v3", 3.0, 3);

        // Still only 1 row
        REQUIRE(TableHasExpectedRowCount(db, "TEST_TABLE", 1) == true);

        // Has final values
        wxString name = db.ReturnSingleStringFromSelectCommand(
                "SELECT NAME FROM TEST_TABLE WHERE ID=1");
        REQUIRE(name == "v3");
    }

    db.Close( );
}

// =============================================================================
// InsertOrReplace Type Conversion Tests
// =============================================================================

TEST_CASE("Database::InsertOrReplace type conversion", "[database][crud][insert]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    CreateTestDatabase(db, test_db_path);
    CreateStandardTestTable(db);

    SECTION("handles zero values") {
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           0, "zero", 0.0, 0);

        REQUIRE(RowExists(db, "TEST_TABLE", "ID=0") == true);

        int count = db.ReturnSingleIntFromSelectCommand(
                "SELECT COUNT FROM TEST_TABLE WHERE ID=0");
        REQUIRE(count == 0);
    }

    SECTION("handles negative values") {
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           -1, "negative", -99.5, -100);

        double value = db.ReturnSingleDoubleFromSelectCommand(
                "SELECT VALUE FROM TEST_TABLE WHERE ID=-1");
        REQUIRE(value == Approx(-99.5));

        int count = db.ReturnSingleIntFromSelectCommand(
                "SELECT COUNT FROM TEST_TABLE WHERE ID=-1");
        REQUIRE(count == -100);
    }

    SECTION("handles empty string") {
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           1, "", 0.0, 0);

        wxString name = db.ReturnSingleStringFromSelectCommand(
                "SELECT NAME FROM TEST_TABLE WHERE ID=1");
        REQUIRE(name == "");
    }

    SECTION("handles special characters in string") {
        wxString special = "test'with\"quotes";
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           1, special.ToUTF8( ).data( ), 0.0, 0);

        wxString retrieved = db.ReturnSingleStringFromSelectCommand(
                "SELECT NAME FROM TEST_TABLE WHERE ID=1");
        REQUIRE(retrieved == special);
    }

    SECTION("handles very large double values") {
        double large_value = 1.7976931348623157e+308; // Near max double
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           1, "large", large_value, 0);

        double retrieved = db.ReturnSingleDoubleFromSelectCommand(
                "SELECT VALUE FROM TEST_TABLE WHERE ID=1");
        REQUIRE(retrieved == Approx(large_value));
    }

    SECTION("handles very small double values") {
        double small_value = 2.2250738585072014e-308; // Near min positive double
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           1, "small", small_value, 0);

        double retrieved = db.ReturnSingleDoubleFromSelectCommand(
                "SELECT VALUE FROM TEST_TABLE WHERE ID=1");
        // Use cisTEMx function - handles near-zero case properly (unlike Catch2 Approx)
        REQUIRE(RelativeErrorIsLessThanEpsilon(small_value, retrieved, false));
    }

    db.Close( );
}

// =============================================================================
// ExecuteSQL Tests
// =============================================================================

TEST_CASE("Database::ExecuteSQL executes raw SQL", "[database][crud][insert]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    CreateTestDatabase(db, test_db_path);
    CreateStandardTestTable(db);

    SECTION("executes INSERT statement") {
        db.ExecuteSQL("INSERT INTO TEST_TABLE (ID, NAME, VALUE, COUNT) VALUES (1, 'raw', 1.0, 10)");
        REQUIRE(TableHasExpectedRowCount(db, "TEST_TABLE", 1) == true);
        REQUIRE(RowExists(db, "TEST_TABLE", "ID=1 AND NAME='raw'") == true);
    }

    SECTION("executes UPDATE statement") {
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           1, "original", 1.0, 10);

        db.ExecuteSQL("UPDATE TEST_TABLE SET NAME='modified' WHERE ID=1");

        wxString name = db.ReturnSingleStringFromSelectCommand(
                "SELECT NAME FROM TEST_TABLE WHERE ID=1");
        REQUIRE(name == "modified");
    }

    SECTION("executes DELETE statement") {
        PopulateTestTable(db, "TEST_TABLE", 3);
        REQUIRE(TableHasExpectedRowCount(db, "TEST_TABLE", 3) == true);

        db.ExecuteSQL("DELETE FROM TEST_TABLE WHERE ID=2");

        REQUIRE(TableHasExpectedRowCount(db, "TEST_TABLE", 2) == true);
        REQUIRE(RowExists(db, "TEST_TABLE", "ID=2") == false);
    }

    SECTION("executes CREATE TABLE statement") {
        db.ExecuteSQL("CREATE TABLE CUSTOM_TABLE (ID INTEGER PRIMARY KEY, DATA TEXT)");
        REQUIRE(db.DoesTableExist("CUSTOM_TABLE") == true);
    }

    SECTION("executes DROP TABLE statement") {
        db.CreateTable("TEMP_TABLE", "p", "ID");
        REQUIRE(db.DoesTableExist("TEMP_TABLE") == true);

        db.ExecuteSQL("DROP TABLE TEMP_TABLE");
        REQUIRE(db.DoesTableExist("TEMP_TABLE") == false);
    }

    SECTION("executes ALTER TABLE statement") {
        db.ExecuteSQL("ALTER TABLE TEST_TABLE ADD COLUMN NEW_FIELD TEXT");
        REQUIRE(db.DoesColumnExist("TEST_TABLE", "NEW_FIELD") == true);
    }

    SECTION("executes multiple statements sequentially") {
        db.ExecuteSQL("INSERT INTO TEST_TABLE (ID, NAME, VALUE, COUNT) VALUES (1, 'a', 1.0, 1)");
        db.ExecuteSQL("INSERT INTO TEST_TABLE (ID, NAME, VALUE, COUNT) VALUES (2, 'b', 2.0, 2)");
        db.ExecuteSQL("INSERT INTO TEST_TABLE (ID, NAME, VALUE, COUNT) VALUES (3, 'c', 3.0, 3)");

        REQUIRE(TableHasExpectedRowCount(db, "TEST_TABLE", 3) == true);
    }

    db.Close( );
}

TEST_CASE("Database::ExecuteSQL handles complex queries", "[database][crud][insert]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    CreateTestDatabase(db, test_db_path);
    CreateStandardTestTable(db);
    PopulateTestTable(db, "TEST_TABLE", 5);

    SECTION("executes UPDATE with WHERE clause") {
        db.ExecuteSQL("UPDATE TEST_TABLE SET COUNT=999 WHERE ID > 3");

        int count_id4 = db.ReturnSingleIntFromSelectCommand(
                "SELECT COUNT FROM TEST_TABLE WHERE ID=4");
        REQUIRE(count_id4 == 999);

        int count_id5 = db.ReturnSingleIntFromSelectCommand(
                "SELECT COUNT FROM TEST_TABLE WHERE ID=5");
        REQUIRE(count_id5 == 999);

        // ID 1-3 should be unchanged
        int count_id1 = db.ReturnSingleIntFromSelectCommand(
                "SELECT COUNT FROM TEST_TABLE WHERE ID=1");
        REQUIRE(count_id1 == 10);
    }

    SECTION("executes DELETE with complex WHERE") {
        db.ExecuteSQL("DELETE FROM TEST_TABLE WHERE ID IN (2, 4)");

        REQUIRE(TableHasExpectedRowCount(db, "TEST_TABLE", 3) == true);
        REQUIRE(RowExists(db, "TEST_TABLE", "ID=2") == false);
        REQUIRE(RowExists(db, "TEST_TABLE", "ID=4") == false);
        REQUIRE(RowExists(db, "TEST_TABLE", "ID=1") == true);
        REQUIRE(RowExists(db, "TEST_TABLE", "ID=3") == true);
        REQUIRE(RowExists(db, "TEST_TABLE", "ID=5") == true);
    }

    db.Close( );
}

// ExecuteSQL error handling tests removed - they attempt to trigger DEBUG_ABORT
// by passing invalid SQL. ExecuteSQL has DEBUG_ABORT for non-LOCKED errors,
// indicating these are programmer errors that calling code should prevent.
