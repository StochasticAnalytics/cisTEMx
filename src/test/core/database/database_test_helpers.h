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

#ifndef DATABASE_TEST_HELPERS_H
#define DATABASE_TEST_HELPERS_H

#include "../../../core/core_headers.h"
#include "../../../../include/catch2/catch.hpp"

/**
 * Shared test utilities for database unit tests.
 *
 * These helpers provide common functionality needed across multiple
 * database test files to reduce code duplication and ensure consistent
 * test patterns.
 *
 * Note: Includes core_headers.h which brings in wx types needed by Database API.
 */

namespace DatabaseTestHelpers {

/**
     * Creates a unique temporary database file path.
     *
     * Uses wxFileName::CreateTempFileName() to generate a guaranteed unique path.
     * The temporary file is created and then immediately deleted, leaving a unique
     * path for Database::CreateNewDatabase() which expects a non-existent file.
     *
     * @return wxFileName for a temporary database file
     */
inline wxFileName CreateTempDatabasePath( ) {
    wxString tempPath = wxFileName::CreateTempFileName("cistem_test_");
    wxRemoveFile(tempPath); // Remove the file, keep the unique path
    return wxFileName(tempPath);
}

/**
     * Creates a new test database with PROCESS_LOCK table.
     *
     * This is a convenience wrapper that creates a database and initializes
     * the PROCESS_LOCK table that Database::Close() expects to exist.
     *
     * @param db Database object to use
     * @param path Path where database should be created
     * @return true if successful
     */
inline bool CreateTestDatabase(Database& db, const wxFileName& path) {
    if ( ! db.CreateNewDatabase(path) ) {
        return false;
    }
    db.CreateProcessLockTable( );
    return true;
}

/**
     * Removes a temporary database file if it exists.
     *
     * Safe to call even if file doesn't exist.
     *
     * @param path Path to database file to remove
     */
inline void CleanupTempDatabase(const wxFileName& path) {
    if ( path.FileExists( ) )
        wxRemoveFile(path.GetFullPath( ));
}

/**
     * Creates a simple test table with standard columns.
     *
     * Table schema: TEST_TABLE (ID INTEGER PRIMARY KEY, NAME TEXT, VALUE REAL, COUNT INTEGER)
     *
     * @param db Database instance (must be open)
     * @param table_name Name of table to create
     * @return true if table created successfully
     */
inline bool CreateStandardTestTable(Database& db, const char* table_name = "TEST_TABLE") {
    return db.CreateTable(table_name, "ptri",
                          "ID", "NAME", "VALUE", "COUNT");
}

/**
     * Populates test table with simple sequential data.
     *
     * Inserts rows with pattern:
     * - ID: 1, 2, 3, ...
     * - NAME: "item_1", "item_2", ...
     * - VALUE: 1.5, 2.5, 3.5, ...
     * - COUNT: 10, 20, 30, ...
     *
     * @param db Database instance (must be open with table created)
     * @param table_name Table to populate
     * @param num_rows Number of rows to insert
     */
inline void PopulateTestTable(Database& db, const char* table_name, int num_rows) {
    for ( int i = 1; i <= num_rows; i++ ) {
        wxString name_str = wxString::Format("item_%d", i);
        db.InsertOrReplace(table_name, "ptri",
                           "ID", "NAME", "VALUE", "COUNT",
                           i,
                           name_str.ToUTF8( ).data( ),
                           static_cast<double>(i) + 0.5,
                           i * 10);
    }
}

/**
     * Checks if a table has the expected number of rows.
     *
     * @param db Database instance
     * @param table Table name
     * @param expected Expected row count
     * @return true if table has exactly expected rows
     */
inline bool TableHasExpectedRowCount(Database& db, const char* table, int expected) {
    int actual_count = db.ReturnSingleIntFromSelectCommand(
            wxString::Format("SELECT COUNT(*) FROM %s", table));
    return actual_count == expected;
}

/**
     * Checks if a specific row exists matching WHERE clause.
     *
     * Example: RowExists(db, "TEST_TABLE", "ID=5 AND NAME='test'")
     *
     * @param db Database instance
     * @param table Table name
     * @param where_clause SQL WHERE clause (without "WHERE" keyword)
     * @return true if at least one row matches
     */
inline bool RowExists(Database& db, const char* table, const char* where_clause) {
    int count = db.ReturnSingleIntFromSelectCommand(
            wxString::Format("SELECT COUNT(*) FROM %s WHERE %s", table, where_clause));
    return count > 0;
}

/**
     * Checks if a database file exists on filesystem.
     *
     * @param path Database file path
     * @return true if file exists and is accessible
     */
inline bool DatabaseFileExists(const wxFileName& path) {
    return path.FileExists( );
}

/**
     * Verifies table contains a specific integer value in a column.
     *
     * @param db Database instance
     * @param table Table name
     * @param column Column name
     * @param expected_value Value to check for
     * @return true if value found in column
     */
inline bool TableContainsValue(Database& db, const char* table,
                               const char* column, int expected_value) {
    wxArrayInt values = db.ReturnIntArrayFromSelectCommand(
            wxString::Format("SELECT %s FROM %s", column, table));
    for ( size_t i = 0; i < values.GetCount( ); i++ ) {
        if ( values[i] == expected_value )
            return true;
    }
    return false;
}

/**
     * RAII helper to automatically cleanup temp database on scope exit.
     *
     * Usage:
     *   wxFileName path = CreateTempDatabasePath();
     *   DatabaseCleanupGuard guard(path);
     *   // ... test code ...
     *   // Database automatically cleaned up when guard goes out of scope
     */
class DatabaseCleanupGuard {
  private:
    wxFileName path_;

  public:
    explicit DatabaseCleanupGuard(const wxFileName& path) : path_(path) {}

    ~DatabaseCleanupGuard( ) {
        CleanupTempDatabase(path_);
    }

    // Prevent copying
    DatabaseCleanupGuard(const DatabaseCleanupGuard&)            = delete;
    DatabaseCleanupGuard& operator=(const DatabaseCleanupGuard&) = delete;
};

} // namespace DatabaseTestHelpers

#endif // DATABASE_TEST_HELPERS_H
