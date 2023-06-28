#!/bin/bash

# The rules to add a new parameter to cisTEM are enumerated at the top of cistem_parameters.cpp
# To help minimize mistakes in adding a paramter, this script will check the implementation of those rules.

# NOTE: a design of the class making the list of parameters iterable would be a better solution, but this is a quick and dirty solution.

# TODO: get this from the CLI
new_parameter_name="POSITION_IN_STACK"

# cistem_parameters.h
# -------------------
# 1. Add a bitwise definition for your data type to the top of cistem_parameters.h
msg="ERROR: cistem_parameters.h:1: $new_parameter_name not found"
[[ "cond_true" == $(awk -v new_parameter_name="$new_parameter_name" '($0 ~ /^#define/ && $2 ~ new_parameter_name){print "cond_true"}' src/core/cistem_parameters.h) ]] || echo $msg
# 2. Add it as a new member variable to  cisTEMParameterLine in cistem_parameters.h
# 3. Add it as a new member variable to  cisTEMParameterMask in cistem_parameters.h
# 4. Add a new method to return the parameter from a given line e.g. ReturnPositionInStack