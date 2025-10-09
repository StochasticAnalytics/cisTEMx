#ifndef _SRC_CORE_DATABASE_TYPESAFE_DATABASE_HELPERS_H_
#define _SRC_CORE_DATABASE_TYPESAFE_DATABASE_HELPERS_H_

#include <array>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

// Configuration flag for runtime validation
constexpr bool validate_db_schema_at_runtime = false;

// A tiny holder for a compile-time char array
template <std::size_t N>
struct ct_string {
    char v[N]{ };

    constexpr const char* c_str( ) const { return v; }

    static constexpr std::size_t size = N - 1;
};

// Compute length of a C string at compile time
constexpr std::size_t cstrlen(const char* s) {
    std::size_t n = 0;
    while ( s[n] != '\0' )
        ++n;
    return n;
}

// Extract substring after the first '.'; returns empty if not found
template <std::size_t MaxN>
constexpr ct_string<MaxN> rhs_after_dot(const char* s) {
    std::size_t n = cstrlen(s);
    std::size_t i = 0;
    while ( i < n && s[i] != '.' )
        ++i;
    ct_string<MaxN> out{ };
    if ( i == n )
        return out; // no dot, empty
    std::size_t j = i + 1;
    std::size_t k = 0;
    // copy until end or MaxN-1 chars
    while ( j < n && k + 1 < MaxN ) {
        out.v[k++] = s[j++];
    }
    out.v[k] = '\0';
    return out;
}

// Helper taking a string literal so we know a bound
template <std::size_t N>
constexpr auto extract_after_dot(const char (&lit)[N]) {
    return rhs_after_dot<N>(lit);
}

// Macro for compile-time stringification of member variable name
#define DB_COL(expr) extract_after_dot(#expr).c_str( ), expr

// Compile-time string comparison for membership check
constexpr bool str_equal(const char* a, const char* b) {
    while ( *a && *b ) {
        if ( *a++ != *b++ )
            return false;
    }
    return *a == *b;
}

// Compile-time schema presence check
template <std::size_t N>
constexpr bool schema_contains(const std::array<const char*, N>& schema, const char* field) {
    for ( std::size_t i = 0; i < N; ++i ) {
        if ( std::strcmp(schema[i], field) == 0 )
            return true;
    }
    return false;
}

// Helper to check if a value is a compile-time constant
template <const char* str, typename Schema>
struct ValidateColumn {
    static_assert(Schema{ }.contains(str), "Column not in schema");
};

// Helper to validate column names at runtime (only when enabled)
template <typename Schema, std::size_t... Is, typename... Ts>
void validate_columns_impl(const Schema& schema, std::index_sequence<Is...>, Ts&&... ts) {
    if constexpr ( validate_db_schema_at_runtime ) {
        auto tup = std::forward_as_tuple(std::forward<Ts>(ts)...);

        // Runtime validation - when disabled, this entire block is eliminated at compile time
        // Build error message with column name if validation fails
        ((void)(schema.contains(std::get<2 * Is>(tup)) ? 0 : throw std::runtime_error(std::string("Column not in schema: ") + std::get<2 * Is>(tup))), ...);
    }
    // When validate_db_schema_at_runtime is false, this function becomes empty
    // and will be completely optimized out by the compiler
}

// Process pairs using index sequence
template <typename Schema, typename... Ts, std::size_t... Is>
void func_impl(const Schema& schema, std::index_sequence<Is...>, Ts&&... ts) {
    auto tup = std::forward_as_tuple(std::forward<Ts>(ts)...);

    std::cout << "func called with: ";
    bool first = true;

    // For each pair i: name at 2*i, value at 2*i+1
    ((
             std::cout << (first ? (first = false, "") : ", "),
             std::cout << std::get<2 * Is>(tup) << "=" << std::get<2 * Is + 1>(tup)),
     ...);

    std::cout << std::endl;
}

// Main function that takes a schema type and alternating name,value pairs
template <typename Schema, typename... Ts>
void func(const Schema& schema, Ts&&... ts) {
    static_assert(sizeof...(Ts) % 2 == 0, "func expects name/value pairs");

    constexpr std::size_t N = sizeof...(Ts) / 2;

    // Validate all column names are in the schema
    validate_columns_impl(schema, std::make_index_sequence<N>{ }, std::forward<Ts>(ts)...);

    // Process the pairs
    func_impl(schema, std::make_index_sequence<N>{ }, std::forward<Ts>(ts)...);
}

// ==============================================================================
// Type-Safe Database SELECT Operations
// ==============================================================================

// Process pairs for batch_select using index sequence
// Builds SQL SELECT query and populates values from database
template <typename Schema, typename... Ts, std::size_t... Is>
void batch_select_impl(const Schema& schema, const char* where_clause,
                       std::index_sequence<Is...>, Ts&&... ts) {
    auto tup = std::forward_as_tuple(std::forward<Ts>(ts)...);

    // Get table name from schema's table_type
    constexpr const char* table_name = Schema::table_type::table_name;

    // Build SELECT column list using fold expression
    std::ostringstream query;
    query << "SELECT ";
    bool first = true;
    ((query << (first ? (first = false, "") : ", ") << std::get<2 * Is>(tup)), ...);
    query << " FROM " << table_name;
    if ( where_clause && where_clause[0] != '\0' ) {
        query << " WHERE " << where_clause;
    }

    std::cout << "SQL: " << query.str( ) << std::endl;

    // TODO: Execute query and populate values at indices 2*Is+1
    // This requires Database API integration
}

// Main batch_select function that takes schema, WHERE clause, and alternating name/value pairs
template <typename Schema, typename... Ts>
void batch_select(const Schema& schema, const char* where_clause, Ts&&... ts) {
    static_assert(sizeof...(Ts) % 2 == 0, "batch_select expects name/value pairs");

    constexpr std::size_t N = sizeof...(Ts) / 2;

    // Validate all column names are in the schema
    validate_columns_impl(schema, std::make_index_sequence<N>{ }, std::forward<Ts>(ts)...);

    // Build query and execute
    batch_select_impl(schema, where_clause, std::make_index_sequence<N>{ }, std::forward<Ts>(ts)...);
}

#endif // _SRC_CORE_DATABASE_TYPESAFE_DATABASE_HELPERS_H_