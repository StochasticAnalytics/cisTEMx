# Database Development Guidelines for cisTEM

This file provides guidance for working with cisTEM's database layer and schema management.

## Database Schema Architecture

### Centralized Schema Definition

**Critical Success Factor:** The schema is centralized in `database_schema.h` using structured tuples, making schema changes significantly less error-prone than scattered SQL strings.

```cpp
// In database_schema.h
using TableData = std::tuple<wxString, char*, std::vector<wxString>>;

enum {
    TABLE_NAME,     // 0: Name of the table
    TABLE_TYPES,    // 1: Type encoding string (e.g., "plrliiliitrrrrrrrrrr")
    TABLE_COLUMNS   // 2: Vector of column names
};

std::vector<TableData> static_tables{
    {"TEMPLATE_MATCH_LIST", "plrliiliitrrrrrrrrrr",
     {"TEMPLATE_MATCH_ID", "DATETIME_OF_RUN", "ELAPSED_TIME_SECONDS", ...}},
    // More tables...
};
```

### Type Encoding Characters

The encoding string uses single characters to represent SQL types:

| Char | SQL Type | C/C++ Type | Use Case |
|------|----------|------------|----------|
| `p` | INTEGER PRIMARY KEY | `int` | Integer primary keys |
| `P` | INTEGER PRIMARY KEY | `long` | Long integer primary keys |
| `t` | TEXT | `wxString`, `const char*` | String data |
| `r` | REAL | `double` | Double-precision floats |
| `s` | REAL | `float` | Single-precision floats (stored as REAL) |
| `i` | INTEGER | `int` | Integer values |
| `l` | INTEGER | `long` | Long integer values |
| `f` | TEXT | `wxFileName` | Special text for filenames |

## Schema Update Best Practices

### Pattern That Made Recent Updates Successful

**The Key Innovation:** When adding/removing columns from a table, validate encoding strings by **counting characters programmatically** rather than manually.

#### Before (Error-Prone Method)
```cpp
// Manually counting: "p l r l i i l i i t r r r r r r r r r r"
//                     1 2 3 4 5 6 7 8 9...20
// Easy to miscount!
{"TEMPLATE_MATCH_LIST", "plrliiliitrrrrrrrrrr", {...}},
```

#### After (Validated Method)
```cpp
// Use your editor or a simple script to validate:
// 1. Count encoding string length
// 2. Count column names vector size
// 3. Verify they match
size_t type_count = strlen("plrliiliitrrrrrrrrrr");    // 20
size_t column_count = column_vector.size();              // 20
assert(type_count == column_count);  // MUST match!
```

### The Encoding String Validation Pattern

**Problem:** When updating table schemas, manually counting encoding string characters is extremely error-prone, especially for tables with 20+ columns.

**Solution Used in Recent Updates:**

```bash
# Quick shell validation - count characters in encoding string
echo "plrliiliitrrrrrrrrrr" | wc -c
# Output: 21 (includes newline, so subtract 1 = 20 characters)

# Count columns in the schema
# Just count the comma-separated column names
```

**Better Solution - Python validation script** (created during recent updates):

```python
#!/usr/bin/env python3
# Save to: .claude/cache/validate_db_schema.py

def validate_table(table_name, encoding, columns):
    """Validate that encoding string length matches column count"""
    enc_len = len(encoding)
    col_len = len(columns)

    if enc_len != col_len:
        print(f"❌ {table_name}: {enc_len} types != {col_len} columns")
        print(f"   Encoding: '{encoding}'")
        print(f"   Columns: {col_len}")
        return False
    else:
        print(f"✓ {table_name}: {enc_len} types = {col_len} columns")
        return True

# Example usage:
validate_table("TEMPLATE_MATCH_LIST",
               "plrliiliitrrrrrrrrrr",
               ["TEMPLATE_MATCH_ID", "DATETIME_OF_RUN", ...])  # 20 columns
```

### Schema Update Workflow

When modifying a table schema:

1. **Update the schema definition** in `database_schema.h`:
   ```cpp
   // OLD:
   {"TEMPLATE_MATCH_LIST", "plrliiliitrrrrrrrrrr",
    {"TEMPLATE_MATCH_ID", "DATETIME_OF_RUN", ..., "20_COLUMNS"}},

   // NEW (adding one REAL column):
   {"TEMPLATE_MATCH_LIST", "plrliiliitrrrrrrrrrrrr",  // +1 'r'
    {"TEMPLATE_MATCH_ID", "DATETIME_OF_RUN", ..., "21_COLUMNS"}},
   ```

2. **Validate encoding string length** matches column count:
   ```bash
   # Count encoding characters
   echo -n "plrliiliitrrrrrrrrrrrr" | wc -c  # Should be 21
   ```

3. **Update database access methods** that read/write this table:
   ```cpp
   // BeginBatchInsert - encoding must match table schema
   BeginBatchInsert("TEMPLATE_MATCH_LIST", 21,  // Column count
                    "TEMPLATE_MATCH_ID", "DATETIME_OF_RUN", ...);

   // AddToBatchInsert - encoding for VALUE types
   AddToBatchInsert("plrliiliitrrrrrrrrrrrr",  // Must match schema
                    id, datetime, elapsed, ...);

   // GetFromBatchSelect - encoding for reading
   GetFromBatchSelect("lrliiitrrrrrrrrrrr",  // Match column types
                      &id, &datetime, &elapsed, ...);
   ```

4. **Search and update all uses**:
   ```bash
   # Find all places this table is accessed
   grep -r "TEMPLATE_MATCH_LIST" src/

   # Check for encoding strings that need updating
   grep -r "plrliiliitrrrrrrrrrr" src/
   ```

## Type-Safe Database Operations (NEW - October 2025)

### Overview

cisTEM now supports **compile-time type-safe database operations** using template metaprogramming and the `DB_COL()` macro pattern. This eliminates manual encoding strings and provides refactor-safe, self-documenting code.

**Files:**
- `src/core/database/typesafe_database_schema.h` - Type-safe struct definitions and schemas
- `src/core/database/typesafe_database_helpers.h` - Template machinery and DB_COL macro

### Type-Safe Schema Definition

Instead of encoding strings like `"plrliiliitrrrrrrrrrr"`, define typed structs:

```cpp
// In typesafe_database_schema.h
struct template_match_list {
    static constexpr const char* table_name = "TEMPLATE_MATCH_LIST";

    int         template_match_id;        // p - PRIMARY KEY
    long        datetime_of_run;          // l
    long        elapsed_time_seconds;     // l
    // ... all 20 typed members
    std::string output_filename_base;     // t - TEXT (using std::string, not wxString)
    double      pixel_size;               // r - REAL
    // ... remaining members
};

// Schema connects struct to column names
using template_match_list_schema = TableSchema<template_match_list, 20>;
constexpr template_match_list_schema template_match_list_columns{{
    "template_match_id",
    "datetime_of_run",
    // ... column names matching struct members (lowercase, SQL case-insensitive)
}};
```

**Key Design Decisions:**
- **Struct members use `std::string`** instead of `wxString` for portability
- **Column names match struct members** (lowercase) - SQL is case-insensitive anyway
- **Table name embedded in struct** as `static constexpr`
- **std::string used for TEXT columns** - convert to/from wxString when interfacing with existing code

### DB_COL() Macro Pattern

The `DB_COL()` macro extracts member names from `struct.member` syntax:

```cpp
template_match_list record;

// DB_COL(record.pixel_size) expands to: "pixel_size", record.pixel_size
// Name extracted automatically - no manual typing!
```

### Type-Safe SELECT Example

```cpp
template_match_list record;

batch_select(template_match_list_columns,
             "template_match_id = 42",  // WHERE clause
             DB_COL(record.template_match_id),
             DB_COL(record.pixel_size),
             DB_COL(record.voltage));

// Generates: SELECT template_match_id, pixel_size, voltage
//            FROM TEMPLATE_MATCH_LIST
//            WHERE template_match_id = 42
```

**Benefits:**
- ✅ No encoding string required
- ✅ Column names validated at compile-time (or runtime if enabled)
- ✅ Refactor-safe: rename `record.pixel_size` → compiler finds all uses
- ✅ Self-documenting: struct IS the schema
- ✅ SQL generated via fold expressions

### Runtime Validation (Toggleable)

```cpp
// In typesafe_database_helpers.h
constexpr bool validate_db_schema_at_runtime = false;  // Zero overhead when disabled
```

When enabled:
```cpp
// Invalid column name detected at runtime
batch_select(schema, "...", "INVALID_COLUMN", value, ...);
// Throws: std::runtime_error("Column not in schema: INVALID_COLUMN")
```

**Validation guarantees:**
- When **disabled**: Complete optimization - zero runtime cost
- When **enabled**: Descriptive error messages with column names
- Uses `if constexpr` - validation code eliminated at compile time when disabled

### Integration with Existing Code

**Current approach** (still valid):
```cpp
InsertOrReplace("TEMPLATE_MATCH_LIST", "pllliiliitrrrrrrrrrr",
                "TEMPLATE_MATCH_ID", "DATETIME_OF_RUN", ...,
                id, datetime, ...);
```

**Future type-safe approach** (in development):
```cpp
template_match_list record = CreateTemplateMatchListRecord(id, job_details);
batch_insert(template_match_list_columns, DB_COL(record.template_match_id), ...);
```

### Migration Strategy

1. **Phase 1 (COMPLETE):** Type-safe schema definitions and SQL generation
2. **Phase 2 (IN PROGRESS):** Database integration - execute queries, bind results
3. **Phase 3:** Update `AddTemplateMatchingResult()` and `GetTemplateMatchingResultByID()`
4. **Phase 4:** Extend to other tables (TEMPLATE_MATCH_QUEUE, IMAGE_ASSETS, etc.)

### Testing Type-Safe Operations

See `.claude/cache/test_typesafe_db.cpp` for validation tests.

```bash
# Compile and run test
g++ -std=c++17 test_typesafe_db.cpp -o test_typesafe_db
./test_typesafe_db

# Expected output:
# SQL: SELECT template_match_id, pixel_size FROM TEMPLATE_MATCH_LIST WHERE ...
# ✓ Validation passed (with validation enabled)
# ✗ Caught validation error: Column not in schema: INVALID_NAME
```

## Common Database Operations

### Batch Insert Pattern

```cpp
// 1. Begin batch insert
BeginBatchInsert("TABLE_NAME", num_columns,
                 "COL1", "COL2", "COL3", ...);

// 2. Add rows with matching encoding
for (auto& item : items) {
    AddToBatchInsert("irt",  // Encoding: int, real, text
                     item.id, item.value, item.name.ToUTF8().data());
}

// 3. End batch (executes transaction)
EndBatchInsert();
```

### Batch Select Pattern

```cpp
// 1. Begin select
bool more_data = BeginBatchSelect("SELECT COL1, COL2, COL3 FROM TABLE");

// 2. Fetch rows
while (more_data) {
    int id;
    double value;
    wxString name;

    more_data = GetFromBatchSelect("irt",  // Match SELECT column types
                                   &id, &value, &name);
    // Process data...
}

// 3. End select
EndBatchSelect();
```

### Type Encoding Gotchas

**Float vs Double:**
```cpp
// WRONG - type mismatch
float my_float = 3.14f;
AddToBatchInsert("r", my_float);  // Expects double for 'r'

// CORRECT - explicit cast
AddToBatchInsert("r", double(my_float));  // Cast to double

// OR use 's' encoding and cast on read
AddToBatchInsert("s", my_float);  // Stored as REAL
// When reading:
float my_float;
GetFromBatchSelect("s", &my_float);  // Converts REAL to float
```

**Int vs Long:**
```cpp
// Be explicit about type sizes
int my_int = 42;
long my_long = 42L;

AddToBatchInsert("il", my_int, my_long);  // Explicit encoding

// When in doubt, check the schema:
// 'i' = int (4 bytes)
// 'l' = long (8 bytes on 64-bit systems)
```

## Schema Migration

### Adding New Columns

Use `AddColumnToTable` for backward compatibility:

```cpp
if (!DoesColumnExist("TABLE_NAME", "NEW_COLUMN")) {
    AddColumnToTable("TABLE_NAME", "NEW_COLUMN", "r", "0.0");
    //                table          column       type  default
}
```

### Dropping Columns

SQLite doesn't support DROP COLUMN easily. Pattern:

```cpp
// 1. Create new table with desired schema
CreateTable("TABLE_NAME_NEW", "irt", {"COL1", "COL3"});  // Skip COL2

// 2. Copy data
ExecuteSQL("INSERT INTO TABLE_NAME_NEW SELECT COL1, COL3 FROM TABLE_NAME");

// 3. Drop old table and rename
DeleteTable("TABLE_NAME");
ExecuteSQL("ALTER TABLE TABLE_NAME_NEW RENAME TO TABLE_NAME");
```

## Dynamic Tables

**Critical Indexing Rule:** Dynamic tables use **1-based indexing**, not 0-based.

```cpp
// WRONG:
wxString table = wxString::Format("REFINEMENT_RESULT_%li_0", ref_id);

// CORRECT:
wxString table = wxString::Format("REFINEMENT_RESULT_%li_1", ref_id);
```

**Dynamic Table Naming Pattern:**
- Pattern must end with underscore: `TEMPLATE_MATCH_PEAK_LIST_`
- Instances numbered from 1: `TEMPLATE_MATCH_PEAK_LIST_1`, `_2`, etc.
- **Never end pattern with digits**: Bad: `PARTICLES2_`, Good: `PARTICLES_`

See `database_schema.h` comments for detailed naming rules.

## Transaction Management

### Manual Transactions

```cpp
database.Begin();
try {
    database.ExecuteSQL(query1);
    database.ExecuteSQL(query2);
    database.Commit();
} catch (...) {
    database.Rollback();
    throw;
}
```

### Automatic RAII Transactions

```cpp
{
    BeginCommitLocker active_locker(&database);
    // All operations in this scope are in a transaction
    database.ExecuteSQL(query1);
    database.ExecuteSQL(query2);
}  // Commits automatically on scope exit, rolls back on exception
```

## Testing Database Changes

### Validate Schema After Changes

```cpp
// Verify table exists
MyDebugAssertTrue(DoesTableExist("TABLE_NAME", true),
                  "Table should exist");

// Verify column exists
MyDebugAssertTrue(DoesColumnExist("TABLE_NAME", "COLUMN_NAME"),
                  "Column should exist");

// Verify column count
int column_count = ReturnNumberOfColumnsInTable("TABLE_NAME");
MyDebugAssertTrue(column_count == EXPECTED_COUNT,
                  "Column count mismatch");
```

### Database Integrity Checks

```cpp
// Check for orphaned records
wxString check_query =
    "SELECT COUNT(*) FROM CHILD_TABLE "
    "WHERE PARENT_ID NOT IN (SELECT ID FROM PARENT_TABLE)";
int orphans = ReturnSingleIntFromSelectCommand(check_query);
MyDebugAssertTrue(orphans == 0, "Found orphaned records");
```

## Lessons Learned from Recent Schema Updates

### What Made the Template Matching Schema Refactor Successful

1. **Centralized schema in `database_schema.h`** - Single source of truth
2. **Programmatic encoding validation** - Count characters, don't guess
3. **Helper methods for filename generation** - Consolidate duplicated logic
4. **Comprehensive grep for all uses** - Find every affected call site
5. **Small, focused commits** - One logical change per commit
6. **Test compilation frequently** - Catch errors immediately

### What to Avoid

1. **Don't manually count encoding strings** - Use `wc -c` or a script
2. **Don't scatter SQL strings** - Centralize in schema or methods
3. **Don't assume encoding matches** - Validate with assertions
4. **Don't skip batch operations** - They're much faster than individual inserts
5. **Don't forget to update SELECT encodings** - Both INSERT and SELECT must match

## Performance Considerations

### Batch Operations

```cpp
// SLOW - Individual inserts
for (auto& item : items) {
    InsertOrReplace("TABLE", "irt",
                    "COL1", "COL2", "COL3",
                    item.id, item.value, item.name);
}

// FAST - Batch insert (100x+ faster for large datasets)
BeginBatchInsert("TABLE", 3, "COL1", "COL2", "COL3");
for (auto& item : items) {
    AddToBatchInsert("irt", item.id, item.value, item.name);
}
EndBatchInsert();
```

### Index Usage

```cpp
// Create index for frequently queried columns
ExecuteSQL("CREATE INDEX IF NOT EXISTS idx_template_match_search "
           "ON TEMPLATE_MATCH_LIST(SEARCH_ID)");

// Drop index before bulk inserts, recreate after
ExecuteSQL("DROP INDEX IF EXISTS idx_template_match_search");
// ... bulk inserts ...
ExecuteSQL("CREATE INDEX idx_template_match_search ...");
```

## Best Practices Summary

1. **Always use centralized schema** in `database_schema.h`
2. **Validate encoding strings** programmatically, don't manually count
3. **Use batch operations** for performance
4. **Test schema changes** with assertions
5. **Document schema changes** in commit messages
6. **Use transactions** for multi-step operations
7. **Maintain backward compatibility** with column additions
8. **Follow 1-based indexing** for dynamic tables
9. **Consolidate logic** in helper methods
10. **Grep exhaustively** when changing schemas
