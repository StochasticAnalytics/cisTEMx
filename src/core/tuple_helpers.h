#ifndef _SRC_CORE_TUPLE_HELPERS_H_
#define _SRC_CORE_TUPLE_HELPERS_H_

#include "defines.h"
#include "../constants/constants.h"

// To work with Tuples.
#ifdef EXPERIMENTAL_CISTEMPARAMS
// simplified verions from https://blog.tartanllama.xyz/exploding-tuples-fold-expressions/
namespace cistem {

template <std::size_t... Idx>
auto make_index_dispatcher(std::index_sequence<Idx...>) {
    return [](auto&& f) { (f(std::integral_constant<std::size_t, Idx>{ }), ...); };
}

template <std::size_t N>
auto make_index_dispatcher( ) {
    return make_index_dispatcher(std::make_index_sequence<N>{ });
}

template <typename Tuple, typename Func>
void for_each(Tuple&& t, Func&& f) {
    constexpr auto n          = std::tuple_size<std::decay_t<Tuple>>::value;
    auto           dispatcher = make_index_dispatcher<n>( );
    dispatcher([&f, &t](auto idx) { f(std::get<idx>(std::forward<Tuple>(t))); });
}

} // namespace cistem

/// @brief /
/// @tparam T
/// @tparam Op
/// @param val
/// @param other_val /
template <typename T, cistem::tuple_ops::Enum Op>
void _BinaryTupleOp(T& val, const T& other_val) {
    static_assert(std::is_arithmetic_v<T>, "BinaryTupleOp only works with arithmetic types");
    static_assert(std::is_same_v<Op, cistem::tuple_ops::ADD> ||
                          std::is_same_v<Op, cistem::tuple_ops::SUBTRACT> ||
                          std::is_same_v<Op, cistem::tuple_ops::ADDSQUARE> ||
                          std::is_same_v<Op, cistem::tuple_ops::REPLACE_NAN_AND_INF> ||
                          std::is_same_v<Op, cistem::tuple_ops::MULTIPLY> ||
                          std::is_same_v<Op, cistem::tuple_ops::DIVIDE>,
                  "Unknown operation for arithmetic type");

    if constexpr ( std::is_same_v<Op, cistem::tuple_ops::ADD> ) {
        val += other_val;
    }
    else if constexpr ( std::is_same_v<Op, cistem::tuple_ops::SUBTRACT> ) {
        val -= other_val;
    }
    else if constexpr ( std::is_same_v<Op, cistem::tuple_ops::ADDSQUARE> ) {
        val += other_val * other_val;
    }
    else if constexpr ( std::is_same_v<Op, cistem::tuple_ops::MULTIPLY> ) {
        val *= other_val;
    }
    else if constexpr ( std::is_same_v<Op, cistem::tuple_ops::DIVIDE> ) {
        val /= other_val;
    }
    else if constexpr ( std::is_same_v<Op, cistem::tuple_ops::REPLACE_NAN_AND_INF> ) {
        if constexpr ( std::is_arithmetic_v<decltype(val)> ) {
            if ( ! std::isfinite(val) ) )
                val = other_val;
        }
        else {
            MyDebugAssertTrue(false, "cisTEMParameterLine::ReplaceNanAndInfWithOther() - Unknown type for parameter %s\n", cistem::parameter_names::names[counter]);
        }
    }
    else {
        MyDebugAssertTrue(false, "Unknown operation for arithmetic type")
    }
};

template <cistem::tuple_ops::Enum Op, typename TupleT, std::size_t... Is>
void _For_Tuple_BinaryOp_impl(TupleT& tp, const TupleT& other_tp, std::index_sequence<Is...>) {
    // Use a fold expression to call the tuple Op on each element of the tuple pair
    (_BinaryTupleOp<Op>(std::get<Is>(tp), std::get<Is>(other_tp)), ...);
}

// This is the driver function called by the user, it gets the tuple size which is needed to "loop"
// over all members at compile time. This method expects two tuples of the same time and applies a binary op
// which must have a type in cistem_contants.h:cistem::tuple_ops::Enum
template <cistem::tuple_ops::Enum Op, typename TupleT, std::size_t TupSize = std::tuple_size_v<TupleT>>
void For_Each_Tuple_BinaryOp(TupleT& tp, const TupleT& other_tp) {
    _For_Tuple_BinaryOp_impl<Op>(tp, other_tp, std::make_index_sequence<TupSize>{ });
}

template <typename T, cistem::tuple_ops::Enum Op>
void _UnaryTupleOp(T& val, const float constant_value) {
    static_assert(std::is_arithmetic_v<T>, "UnaryTupleOp only works with arithmetic types");
    static_assert(std::is_same_v<Op, cistem::tuple_ops::SET_TO_ZERO> ||
                          std::is_same_v<Op, cistem::tuple_ops::MULTIPLY_BY_CONSTANT> ||
                          std::is_same_v<Op, cistem::tuple_ops::DIVIDE_BY_CONSTANT>,
                  "Unknown operation for arithmetic type");

    if constexpr ( std::is_same_v<Op, cistem::tuple_ops::SET_TO_ZERO> ) {
        if constexpr ( std::is_integral_v<T> ) {
            val = 0;
        }
        else if constexpr ( std::is_floating_point_v<T> ) {
            val = 0.0f;
        }
        else if constexpr ( std::is_same_v<T, wxString> || std::is_same_v<T, std::string> ) {
            val = "";
        }
        else {
            MyDebugAssertTrue(false, "cisTEMParameterLine::SetAllToZero() - Unknown type for parameter %s\n", cistem::parameter_names::names[counter]);
        }
    }
    else if constexpr ( std::is_same_v<Op, cistem::tuple_ops::MULTIPLY_BY_CONSTANT> ) {

        val = static_cast<decltype(val)>(static_cast<float>(val) * constant_value);
    }
    else if constexpr ( std::is_same_v<Op, cistem::tuple_ops::DIVIDE_BY_CONSTANT> ) {
        // Leaving it up to the caller to decide what to do about zero division (probably check and skip before calling this method)
        val = static_cast<decltype(val)>(static_cast<float>(val) / constant_value);
    }
    else {
        MyDebugAssertTrue(false, "Unknown operation for arithmetic type")
    }
};

template <cistem::tuple_ops::Enum Op, typename TupleT, std::size_t... Is>
void _For_Each_Tuple_UnaryOp_impl(TupleT& tp, const float constant_value, std::index_sequence<Is...>) {
    // Use a fold expression to call the tuple Op on each element of the tuple pair
    (_UnaryTupleOp<Op>(std::get<Is>(tp), constant_value), ...);
}

// This is the driver function called by the user, it gets the tuple size which is needed to "loop"
// over all members at compile time. This method expects two tuples of the same time and applies a binary op
// which must have a type in cistem_contants.h:cistem::tuple_ops::Enum
template <cistem::tuple_ops::Enum Op, typename TupleT, std::size_t TupSize = std::tuple_size_v<TupleT>>
void For_Each_Tuple_UnaryOp(TupleT& tp, const float constant_value = 0.f) {
    _For_Each_Tuple_UnaryOp_impl<Op>(tp, constant_value, std::make_index_sequence<TupSize>{ });
}

///////////////////////////////////////

#endif // EXPERIMENTAL_CISTEMPARAMS
#endif // _SRC_CORE_TUPLE_HELPERS_H_