#!/bin/bash

# This script is used to test the cistem package

echo $PATH
mkdir -p ~/.cistem

echo "Running cistem tests, output will be in ~/.cistem/cistem_tests.log"

echo -e "\n\n////////////////////////////////////////////////////////////////////////////////" >> ~/.cistem/cistem_tests.log
echo $(date) >> ~/.cistem/cistem_tests.log
echo "////////////////////////////////////////////////////////////////////////////////" >> ~/.cistem/cistem_tests.log

echo "Testing gpu device availability."
/cisTEMx/bin/gpu_devices >> ~/.cistem/cistem_tests.log

echo "Testing units."
/cisTEMx/bin/unit_test_runner  >> ~/.cistem/cistem_tests.log

echo "Testing methods."
/cisTEMx/bin/console_test  >> ~/.cistem/cistem_tests.log

echo "Testing algorithms."
/cisTEMx/bin/samples_functional_testing  >> ~/.cistem/cistem_tests.log

