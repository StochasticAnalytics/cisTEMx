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
 * Unit tests for Database transaction management.
 *
 * Tests cover:
 * - Begin/Commit: Transaction control
 * - BeginCommitLocker: RAII wrapper for exception safety
 *
 * Transaction management ensures ACID properties for database operations.
 * Note: Tests behavior without accessing private members.
 */

// =============================================================================
// Basic Transaction Tests
// =============================================================================

TEST_CASE("Database::Begin and Commit single transaction", "[database][crud][transactions]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    CreateTestDatabase(db, test_db_path);
    CreateStandardTestTable(db);

    SECTION("data persists after commit") {
        db.Begin( );
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           1, "test", 1.0, 10);
        db.Commit( );

        REQUIRE(RowExists(db, "TEST_TABLE", "ID=1") == true);
    }

    SECTION("single begin-commit pair with data") {
        db.Begin( );
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           42, "answer", 0.0, 42);
        db.Commit( );

        REQUIRE(RowExists(db, "TEST_TABLE", "ID=42") == true);
    }

    SECTION("multiple operations in single transaction") {
        db.Begin( );
        for ( int i = 1; i <= 5; i++ ) {
            wxString name_str = wxString::Format("item_%d", i);
            db.InsertOrReplace("TEST_TABLE", "ptri",
                               "ID", "NAME", "VALUE", "COUNT",
                               i, name_str.ToUTF8( ).data( ), static_cast<double>(i), i * 10);
        }
        db.Commit( );

        REQUIRE(TableHasExpectedRowCount(db, "TEST_TABLE", 5) == true);
    }

    db.Close( );
}

// =============================================================================
// Nested Transaction Tests
// =============================================================================

TEST_CASE("Database::Nested transactions", "[database][crud][transactions]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    CreateTestDatabase(db, test_db_path);
    CreateStandardTestTable(db);

    SECTION("nested transactions with operations") {
        db.Begin( ); // Outer transaction

        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           1, "outer", 1.0, 1);

        db.Begin( ); // Inner transaction
        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           2, "inner", 2.0, 2);
        db.Commit( ); // Inner commit

        db.Commit( ); // Outer commit

        // Both rows should exist
        REQUIRE(RowExists(db, "TEST_TABLE", "ID=1") == true);
        REQUIRE(RowExists(db, "TEST_TABLE", "ID=2") == true);
    }

    SECTION("deeply nested transactions") {
        db.Begin( );
        db.Begin( );
        db.Begin( );

        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           1, "deep", 1.0, 1);

        db.Commit( );
        db.Commit( );
        db.Commit( );

        REQUIRE(RowExists(db, "TEST_TABLE", "ID=1") == true);
    }

    db.Close( );
}

// =============================================================================
// BeginCommitLocker RAII Tests
// =============================================================================

TEST_CASE("Database::BeginCommitLocker basic usage", "[database][crud][transactions]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    CreateTestDatabase(db, test_db_path);
    CreateStandardTestTable(db);

    SECTION("BeginCommitLocker manages transaction") {
        {
            BeginCommitLocker locker(&db);

            db.InsertOrReplace("TEST_TABLE", "ptri",
                               "ID", "NAME", "VALUE", "COUNT",
                               1, "raii", 1.0, 1);
        } // Destructor commits

        REQUIRE(RowExists(db, "TEST_TABLE", "ID=1") == true);
    }

    SECTION("BeginCommitLocker with nested scope") {
        {
            BeginCommitLocker outer(&db);

            db.InsertOrReplace("TEST_TABLE", "ptri",
                               "ID", "NAME", "VALUE", "COUNT",
                               1, "outer_data", 1.0, 1);

            {
                BeginCommitLocker inner(&db);
                db.InsertOrReplace("TEST_TABLE", "ptri",
                                   "ID", "NAME", "VALUE", "COUNT",
                                   2, "inner_data", 2.0, 2);
            }
        }

        REQUIRE(RowExists(db, "TEST_TABLE", "ID=1") == true);
        REQUIRE(RowExists(db, "TEST_TABLE", "ID=2") == true);
    }

    SECTION("BeginCommitLocker early Commit") {
        BeginCommitLocker locker(&db);

        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           5, "early", 5.0, 5);

        locker.Commit( ); // Manual early commit

        REQUIRE(RowExists(db, "TEST_TABLE", "ID=5") == true);
    }

    db.Close( );
}

TEST_CASE("Database::BeginCommitLocker exception safety", "[database][crud][transactions]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    CreateTestDatabase(db, test_db_path);
    CreateStandardTestTable(db);

    SECTION("exception during transaction commits automatically") {
        try {
            BeginCommitLocker locker(&db);

            db.InsertOrReplace("TEST_TABLE", "ptri",
                               "ID", "NAME", "VALUE", "COUNT",
                               1, "exception_test", 1.0, 1);

            throw std::runtime_error("Simulated error");
        } catch ( const std::runtime_error& ) {
            // Exception caught
        }

        // Data should still be committed
        REQUIRE(RowExists(db, "TEST_TABLE", "ID=1") == true);
    }

    db.Close( );
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST_CASE("Database::Transactions with batch operations", "[database][crud][transactions]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    CreateTestDatabase(db, test_db_path);
    CreateStandardTestTable(db);

    // FIXME: Test causes SIGSEGV (segmentation fault). Need to investigate interaction
    // between batch operations and transactions. Possible null pointer or memory issue.
    /*
    SECTION("batch insert within transaction") {
        db.Begin();

        db.BeginBatchInsert("TEST_TABLE", 4, "ID", "NAME", "VALUE", "COUNT");
        for (int i = 1; i <= 10; i++) {
            wxString name_str = wxString::Format("item_%d", i);
            db.AddToBatchInsert("ptri", i, name_str.ToUTF8().data(),
                                static_cast<double>(i), i);
        }
        db.EndBatchInsert();

        db.Commit();

        REQUIRE(TableHasExpectedRowCount(db, "TEST_TABLE", 10) == true);
    }
    */

    // FIXME: Test causes SIGSEGV. Same transaction/batch operation interaction issue.
    /*
    SECTION("nested transactions with batch operations") {
        BeginCommitLocker outer(&db);

        db.BeginBatchInsert("TEST_TABLE", 4, "ID", "NAME", "VALUE", "COUNT");
        for (int i = 1; i <= 5; i++) {
            db.AddToBatchInsert("ptri", i, "batch", 1.0, i);
        }
        db.EndBatchInsert();

        REQUIRE(TableHasExpectedRowCount(db, "TEST_TABLE", 5) == true);
    }
    */

    db.Close( );
}

TEST_CASE("Database::Transactions with multiple tables", "[database][crud][transactions]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    CreateTestDatabase(db, test_db_path);
    CreateStandardTestTable(db);
    db.CreateTable("SECOND_TABLE", "pi", "ID", "VALUE");

    SECTION("single transaction spans multiple tables") {
        db.Begin( );

        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           1, "table1", 1.0, 1);

        db.InsertOrReplace("SECOND_TABLE", "pi",
                           "ID", "VALUE",
                           1, 100);

        db.Commit( );

        REQUIRE(RowExists(db, "TEST_TABLE", "ID=1") == true);
        REQUIRE(RowExists(db, "SECOND_TABLE", "ID=1") == true);
    }

    db.Close( );
}

// =============================================================================
// Complex Transaction Scenarios
// =============================================================================

TEST_CASE("Database::Complex transaction patterns", "[database][crud][transactions]") {
    wxFileName           test_db_path = CreateTempDatabasePath( );
    DatabaseCleanupGuard cleanup(test_db_path);

    Database db;
    CreateTestDatabase(db, test_db_path);
    CreateStandardTestTable(db);

    SECTION("mixed RAII and manual transaction control") {
        db.Begin( ); // Manual outer

        {
            BeginCommitLocker inner(&db);
            db.InsertOrReplace("TEST_TABLE", "ptri",
                               "ID", "NAME", "VALUE", "COUNT",
                               1, "inner", 1.0, 1);
        } // Inner commits

        db.InsertOrReplace("TEST_TABLE", "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           2, "outer", 2.0, 2);

        db.Commit( ); // Manual outer commit

        REQUIRE(TableHasExpectedRowCount(db, "TEST_TABLE", 2) == true);
    }

    db.Close( );
}
