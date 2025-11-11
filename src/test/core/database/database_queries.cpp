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
 * Unit tests for Database query operations.
 *
 * Tests cover:
 * - ReturnSingle{Int,Long,Double,String}FromSelectCommand: Single value queries
 * - Return{Int,Long,String}ArrayFromSelectCommand: Multiple value queries
 *
 * These operations retrieve data from the database.
 */

// =============================================================================
// Single Value Query Tests
// =============================================================================

TEST_CASE("Database::ReturnSingleIntFromSelectCommand", "[database][crud][queries]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    CreateTestDatabase(db, test_db_path);
    CreateStandardTestTable(db);
    PopulateTestTable(db, "TEST_TABLE", 5);

    SECTION("returns integer from query") {
        int count = db.ReturnSingleIntFromSelectCommand(
                "SELECT COUNT FROM TEST_TABLE WHERE ID=1");
        REQUIRE(count == 10);
    }

    SECTION("returns count aggregate") {
        int row_count = db.ReturnSingleIntFromSelectCommand(
                "SELECT COUNT(*) FROM TEST_TABLE");
        REQUIRE(row_count == 5);
    }

    SECTION("returns zero for empty result") {
        int result = db.ReturnSingleIntFromSelectCommand(
                "SELECT COUNT FROM TEST_TABLE WHERE ID=999");
        REQUIRE(result == 0);
    }

    SECTION("returns ID value") {
        int id = db.ReturnSingleIntFromSelectCommand(
                "SELECT ID FROM TEST_TABLE WHERE NAME='item_3'");
        REQUIRE(id == 3);
    }

    db.Close( );
}

TEST_CASE("Database::ReturnSingleLongFromSelectCommand", "[database][crud][queries]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    CreateTestDatabase(db, test_db_path);
    CreateStandardTestTable(db);

    SECTION("returns long value") {
        long big_id = 9999999L;
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           big_id, "big", 0.0, 0);

        long retrieved = db.ReturnSingleLongFromSelectCommand(
                "SELECT ID FROM TEST_TABLE WHERE NAME='big'");
        REQUIRE(retrieved == big_id);
    }

    SECTION("returns zero for empty result") {
        long result = db.ReturnSingleLongFromSelectCommand(
                "SELECT ID FROM TEST_TABLE WHERE ID=999");
        REQUIRE(result == 0);
    }

    SECTION("returns count as long") {
        PopulateTestTable(db, "TEST_TABLE", 10);
        long count = db.ReturnSingleLongFromSelectCommand(
                "SELECT COUNT(*) FROM TEST_TABLE");
        REQUIRE(count == 10);
    }

    db.Close( );
}

TEST_CASE("Database::ReturnSingleDoubleFromSelectCommand", "[database][crud][queries]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    CreateTestDatabase(db, test_db_path);
    CreateStandardTestTable(db);

    SECTION("returns double value") {
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           1, "pi", 3.14159, 0);

        double value = db.ReturnSingleDoubleFromSelectCommand(
                "SELECT VALUE FROM TEST_TABLE WHERE ID=1");
        REQUIRE(value == Approx(3.14159));
    }

    SECTION("returns zero for empty result") {
        double result = db.ReturnSingleDoubleFromSelectCommand(
                "SELECT VALUE FROM TEST_TABLE WHERE ID=999");
        REQUIRE(result == Approx(0.0));
    }

    SECTION("returns negative double") {
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           2, "neg", -99.5, 0);

        double value = db.ReturnSingleDoubleFromSelectCommand(
                "SELECT VALUE FROM TEST_TABLE WHERE ID=2");
        REQUIRE(value == Approx(-99.5));
    }

    SECTION("returns very small double") {
        double small = 1.23e-10;
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           3, "small", small, 0);

        double value = db.ReturnSingleDoubleFromSelectCommand(
                "SELECT VALUE FROM TEST_TABLE WHERE ID=3");
        // Use cisTEMx function - handles near-zero case (Catch2 Approx fails for zero)
        REQUIRE(RelativeErrorIsLessThanEpsilon(small, value, false));
    }

    db.Close( );
}

TEST_CASE("Database::ReturnSingleStringFromSelectCommand", "[database][crud][queries]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    CreateTestDatabase(db, test_db_path);
    CreateStandardTestTable(db);
    PopulateTestTable(db, "TEST_TABLE", 3);

    SECTION("returns string value") {
        wxString name = db.ReturnSingleStringFromSelectCommand(
                "SELECT NAME FROM TEST_TABLE WHERE ID=1");
        REQUIRE(name == "item_1");
    }

    SECTION("returns empty string for empty result") {
        wxString result = db.ReturnSingleStringFromSelectCommand(
                "SELECT NAME FROM TEST_TABLE WHERE ID=999");
        REQUIRE(result == "");
    }

    SECTION("returns string with special characters") {
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           10, "test'with\"quotes", 0.0, 0);

        wxString name = db.ReturnSingleStringFromSelectCommand(
                "SELECT NAME FROM TEST_TABLE WHERE ID=10");
        REQUIRE(name == "test'with\"quotes");
    }

    SECTION("returns empty string field") {
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           20, "", 0.0, 0);

        wxString name = db.ReturnSingleStringFromSelectCommand(
                "SELECT NAME FROM TEST_TABLE WHERE ID=20");
        REQUIRE(name == "");
    }

    db.Close( );
}

// =============================================================================
// Array Query Tests
// =============================================================================

TEST_CASE("Database::ReturnIntArrayFromSelectCommand", "[database][crud][queries]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    CreateTestDatabase(db, test_db_path);
    CreateStandardTestTable(db);
    PopulateTestTable(db, "TEST_TABLE", 5);

    SECTION("returns array of integers") {
        wxArrayInt counts = db.ReturnIntArrayFromSelectCommand(
                "SELECT COUNT FROM TEST_TABLE ORDER BY ID");
        REQUIRE(counts.GetCount( ) == 5);
        REQUIRE(counts[0] == 10);
        REQUIRE(counts[1] == 20);
        REQUIRE(counts[4] == 50);
    }

    SECTION("returns array of IDs") {
        wxArrayInt ids = db.ReturnIntArrayFromSelectCommand(
                "SELECT ID FROM TEST_TABLE ORDER BY ID");
        REQUIRE(ids.GetCount( ) == 5);
        REQUIRE(ids[0] == 1);
        REQUIRE(ids[4] == 5);
    }

    SECTION("returns empty array for no results") {
        wxArrayInt result = db.ReturnIntArrayFromSelectCommand(
                "SELECT ID FROM TEST_TABLE WHERE ID > 100");
        REQUIRE(result.GetCount( ) == 0);
    }

    SECTION("returns single element array") {
        wxArrayInt result = db.ReturnIntArrayFromSelectCommand(
                "SELECT ID FROM TEST_TABLE WHERE ID=3");
        REQUIRE(result.GetCount( ) == 1);
        REQUIRE(result[0] == 3);
    }

    db.Close( );
}

TEST_CASE("Database::ReturnLongArrayFromSelectCommand", "[database][crud][queries]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    CreateTestDatabase(db, test_db_path);
    CreateStandardTestTable(db);

    SECTION("returns array of long values") {
        for ( long i = 1; i <= 3; i++ ) {
            db.InsertOrReplace("TEST_TABLE", "ptri",
                               "ID", "NAME", "VALUE", "COUNT",
                               i * 1000000L, "big", 0.0, 0);
        }

        wxArrayLong ids = db.ReturnLongArrayFromSelectCommand(
                "SELECT ID FROM TEST_TABLE ORDER BY ID");
        REQUIRE(ids.GetCount( ) == 3);
        REQUIRE(ids[0] == 1000000L);
        REQUIRE(ids[1] == 2000000L);
        REQUIRE(ids[2] == 3000000L);
    }

    SECTION("returns empty array for no results") {
        wxArrayLong result = db.ReturnLongArrayFromSelectCommand(
                "SELECT ID FROM TEST_TABLE WHERE ID > 999999");
        REQUIRE(result.GetCount( ) == 0);
    }

    db.Close( );
}

TEST_CASE("Database::ReturnStringArrayFromSelectCommand", "[database][crud][queries]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    CreateTestDatabase(db, test_db_path);
    CreateStandardTestTable(db);
    PopulateTestTable(db, "TEST_TABLE", 4);

    SECTION("returns array of strings") {
        wxArrayString names = db.ReturnStringArrayFromSelectCommand(
                "SELECT NAME FROM TEST_TABLE ORDER BY ID");
        REQUIRE(names.GetCount( ) == 4);
        REQUIRE(names[0] == "item_1");
        REQUIRE(names[1] == "item_2");
        REQUIRE(names[3] == "item_4");
    }

    SECTION("returns empty array for no results") {
        wxArrayString result = db.ReturnStringArrayFromSelectCommand(
                "SELECT NAME FROM TEST_TABLE WHERE ID > 100");
        REQUIRE(result.GetCount( ) == 0);
    }

    SECTION("returns array with empty strings") {
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           10, "", 0.0, 0);

        wxArrayString names = db.ReturnStringArrayFromSelectCommand(
                "SELECT NAME FROM TEST_TABLE WHERE ID >= 10");
        REQUIRE(names.GetCount( ) == 1);
        REQUIRE(names[0] == "");
    }

    SECTION("returns array with special characters") {
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           20, "test'quote", 0.0, 0);
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           21, "test\"double", 0.0, 0);

        wxArrayString names = db.ReturnStringArrayFromSelectCommand(
                "SELECT NAME FROM TEST_TABLE WHERE ID >= 20 ORDER BY ID");
        REQUIRE(names.GetCount( ) == 2);
        REQUIRE(names[0] == "test'quote");
        REQUIRE(names[1] == "test\"double");
    }

    db.Close( );
}

// =============================================================================
// Query Integration Tests
// =============================================================================

TEST_CASE("Database query operations with complex data", "[database][crud][queries]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    CreateTestDatabase(db, test_db_path);
    CreateStandardTestTable(db);

    SECTION("queries with WHERE clauses") {
        PopulateTestTable(db, "TEST_TABLE", 10);

        wxArrayInt filtered_ids = db.ReturnIntArrayFromSelectCommand(
                "SELECT ID FROM TEST_TABLE WHERE COUNT > 50 ORDER BY ID");
        REQUIRE(filtered_ids.GetCount( ) == 5); // IDs 6,7,8,9,10 have COUNT > 50
    }

    SECTION("queries with ORDER BY") {
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           3, "charlie", 0.0, 0);
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           1, "alice", 0.0, 0);
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           2, "bob", 0.0, 0);

        wxArrayString names = db.ReturnStringArrayFromSelectCommand(
                "SELECT NAME FROM TEST_TABLE ORDER BY NAME");
        REQUIRE(names.GetCount( ) == 3);
        REQUIRE(names[0] == "alice");
        REQUIRE(names[1] == "bob");
        REQUIRE(names[2] == "charlie");
    }

    SECTION("aggregate functions") {
        PopulateTestTable(db, "TEST_TABLE", 5);

        int max_count = db.ReturnSingleIntFromSelectCommand(
                "SELECT MAX(COUNT) FROM TEST_TABLE");
        REQUIRE(max_count == 50);

        int min_id = db.ReturnSingleIntFromSelectCommand(
                "SELECT MIN(ID) FROM TEST_TABLE");
        REQUIRE(min_id == 1);

        double avg_value = db.ReturnSingleDoubleFromSelectCommand(
                "SELECT AVG(VALUE) FROM TEST_TABLE");
        REQUIRE(avg_value == Approx(3.5)); // (1.5+2.5+3.5+4.5+5.5)/5 = 3.5
    }

    db.Close( );
}
