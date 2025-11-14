---
title: "Testing"
status: legacy
description: "Testing frameworks and unit test documentation from cisTEM developmental-docs"
content_type: reference
audience: [developer, contributor]
level: intermediate
topics: [testing, contributing, quality]
components: [cli, algorithms, gpu]
date_extracted: 2025-11-05
source: developmental-docs
---

# Testing

Comprehensive guide to testing practices in cisTEMx, including unit tests, integration tests, and functional tests.

=== "Basic"

    ## Running Tests

    Quick guide to running the existing test suites in cisTEMx.

    ```mermaid
    graph LR
        subgraph Client["Client Layer"]
            RD[Remote Desktop]
            GUI[GUI Frontend]
        end

        subgraph Core["cisTEMx Core"]
            APP[Application Logic]
            JM[Job Manager]
            DB[(Project Database)]
        end

        subgraph Storage["Data Layer"]
            DS[Data Server]
            MRC[MRC Files]
            STAR[STAR Metadata]
        end

        subgraph Compute["Compute Layer"]
            WN1[Worker Node 1]
            WN2[Worker Node 2]
            WNn[Worker Node N]
        end

        USER((User)) --> RD
        RD --> GUI
        GUI <--> APP
        APP <--> DB
        APP <--> DS
        DS --- MRC
        DS --- STAR
        JM --> WN1
        JM --> WN2
        JM --> WNn
        WN1 -.Results.-> JM
        WN2 -.Results.-> JM
        WNn -.Results.-> JM
        APP --> JM

        style Core fill:#E3F2FD,stroke:#1976D2,stroke-width:2px
        style Storage fill:#FFF3E0,stroke:#F57C00,stroke-width:2px
        style Compute fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px
        style Client fill:#E8F5E9,stroke:#388E3C,stroke-width:2px
        style APP fill:#4A90E2,color:#fff
        style JM fill:#4A90E2,color:#fff
    ```

    ### Unit Tests

    Run the unit test suite:
    ```bash
    ./build/src/unit_test_runner
    ```

    ### Console Tests

    Run console-based functional tests:
    ```bash
    ./build/src/console_test
    ```

    ### Functional Tests

    Run the full functional test suite:
    ```bash
    ./build/src/samples_functional_testing
    ```

    ### Test Output

    - ✓ = Test passed
    - ✗ = Test failed
    - Tests report pass/fail status and timing information
    - Failed tests show assertion details

    ### Prerequisites

    Some tests require:
    - GPU hardware (for GPU-accelerated tests)
    - Test data files (usually in test artifacts)
    - Sufficient disk space for temporary files

    ### Quick Test Commands

    ```bash
    # Run all unit tests
    make test

    # Run specific test suite
    ./build/src/unit_test_runner "TestMRCFunctions"

    # Run tests with verbose output
    ./build/src/unit_test_runner --verbose
    ```

=== "Advanced"

    ## Writing Tests

    Advanced guide for writing unit tests, integration tests, and understanding the test framework.

    ### Test Framework: Catch2

    cisTEMx uses Catch2 for C++ unit testing. Basic test structure:

    ```cpp
    #include <catch2/catch.hpp>

    TEST_CASE("Description of what is tested", "[tag]") {
        // Arrange
        Image test_image;
        test_image.Allocate(128, 128, 1);

        // Act
        test_image.SetToConstant(42.0f);

        // Assert
        REQUIRE(test_image.real_values[0] == 42.0f);
    }
    ```

    ### Test Organization

    Tests are organized by functionality:
    - `src/test/` - Unit tests
    - Test files follow naming: `test_*.cpp`
    - Group related tests with TEST_CASE sections

    ### Creating Test Fixtures

    For tests that need setup/teardown:

    ```cpp
    struct ImageFixture {
        Image test_image;

        ImageFixture() {
            test_image.Allocate(128, 128, 1);
        }

        ~ImageFixture() {
            // Cleanup if needed
        }
    };

    TEST_CASE_METHOD(ImageFixture, "Test with fixture") {
        // test_image is available here
    }
    ```

    ### GPU Testing

    For GPU tests, use compute-sanitizer to detect memory errors:

    ```bash
    compute-sanitizer --tool memcheck ./build/src/unit_test_runner "GPUTest"
    ```

    ### Test Coverage

    Analyze test coverage:

    ```bash
    # Build with coverage flags
    ./configure --enable-coverage
    make

    # Run tests
    make test

    # Generate coverage report
    gcov src/core/*.cpp
    ```

    ### Best Practices

    - **One assertion per test** when possible
    - **Test edge cases**: empty inputs, boundary conditions
    - **Test error handling**: invalid parameters, null pointers
    - **Use descriptive names**: Test names should explain what is tested
    - **Keep tests fast**: Unit tests should complete in milliseconds
    - **Isolate tests**: Tests should not depend on each other

===+ "Developer"

    ## Legacy cisTEM Unit Test Suites

    This section documents the unit test suites from the original cisTEM codebase. These tests cover core image processing functionality.

    !!! note "Current cisTEMx Testing"
        For current cisTEMx testing practices, consult the **unit-testing skill** and modern test suites in `src/test/`.

    ### TestMRCFunctions

    **Functionality Tested:**

    - Open a file with .mrc extension
    - Check if the file is a valid MRC file
    - Read and store the header information
    - Read and store the image data

    !!! note
        - This only tests the original reading of data from MRC mode (TODO: list) and storing as 32 bit float.
        - Aside from the raw data stream, this also adds/removes padding for in place FFTs which is not currently tested

    **Depends on:**

    - MRCFile::OpenFile
    - MRCFile::ReadSlice
    - Artifact: hiv_images_80x80x10

    ---

    ### TestAssignmentOperatorsAndFunctions

    **Functionality Tested:**

    - Copy an Image object and it's data into this Image, when the memory for this Image is not allocated.
    - Copy an Image object and it's data into this Image, when the memory for this Image is allocated, and is a different size, such that it must first be deallocated.
    - Call the Image::CopyFrom method, which just invokes the operator
    ```c++
    Image & Image::operator = (const Image *other_image)
    ```
    - Assign by reference, which then also invokes the same operator as CopyFrom:
    ```c++
    Image & Image::operator = (const Image &other_image)
    {
        *this = &other_image;
        return *this;
    }
    ```
    - Unlike the previous tests, the Image::Consume method directly steals the memory pointed to by the other image.

    **Depends on:**

    - Image::Allocate

    !!! warning
        The Allocate method is not currently tested and does some fairly complicated things:
        - Mutex lock for FFT plan generation
        - Allocating memory including the proper padding
        - Asserting data memory is not currently allocated, and calling Image::Deallocate if needed
        - No asserts in place to confirm proper memory alignment either.

    ---

    ### TestFFTFunctions

    **Functionality Tested:**

    !!! warning
        - We really only every use the MKL routines so FFTW3 is not tested nearly so well.
        - Because FFTW3 is GPL3 the FFTW3 routines need to be replaced with the equivalent MKL calls to be compatible with the current cisTEM license.

    - Image::ForwardFFT
      - Inplace, single precision R2C using either FFTW3 or Intel MKL
    - Image::BackwardFFT
    - RemoveFFTWPadding

    **Depends on:**

    - MRCFile::OpenFile
    - MRCFile::ReadSlice
    - Image::SetToConstant
    - Artifact: sine_wave_128x128x1

    ---

    ### TestScalingAndSizingFunctions

    **Functionality Tested:**

    - Image::ClipInto
      - test real space clipping into a larger image.
      - test real space clipping into a smaller image.
      - test Fourier space clipping into a larger image.
      - test Fourier space clipping into a smaller image.
      - test real space clipping smaller to odd
      - test fourier space flipping smaller to odd

    - Image::Resize
      - Real space big
      - Real space small
      - Fourier space big
      - Fourier space small

    **Depends on:**

    - MRCFile::OpenFile
    - MRCFile::ReadSlice
    - Image::ForwardFFT
    - Image::Consume

    ---

    ### TestFilterFunctions

    **Functionality Tested:**

    - Image::ApplyBFactor

    **Depends on:**

    - MRCFile::OpenFile
    - MRCFile::ReadSlice
    - Image::ForwardFFT
    - Image::BackwardFFT

    ---

    ### TestAlignmentFunctions

    **Functionality Tested:**

    - Image::PhaseShift
    - Image::CalculateCrossCorrelationImageWith
    - Image::FindPeakWithIntegerCoordinates
    - Image::FindPeakWithParabolaFit

    !!! todo "Add test for SwapRealSpaceQuadrants"
        CalculateCrossCorrelationImageWith depends on SwapRealSpaceQuadrants, which depends on PhaseShift. TODO: SwapRealSpaceQuadrants is not currently tested.

    **Depends on:**

    - MRCFile::OpenFile
    - MRCFile::ReadSlice
    - Image::ForwardFFT
    - Image::BackwardFFT

    ---

    ### TestImageArithmeticFunctions

    **Functionality Tested:**

    - Image::AddImage
    - Image::SetToConstant

    !!! todo "Add tests for these other arithmetic functions"
        - Image::SubtractImage
        - Image::SubtractSquaredImage
        - Image::MakeAbsolute
        - Image::MultiplyPixelWiseReal
        - Image::MultiplyPixelWise
        - Image::ConjugateMultiplyPixelWise
        - Image::DividePixelWise

    ---

    ### Additional Test Suites

    - **TestSpectrumBoxConvolution** - Frequency-domain convolution
    - **TestImageLoopingAndAddressing** - Memory access patterns
    - **TestNumericTextFiles** - File I/O operations
    - **TestClipIntoFourier** - Fourier space clipping with odd/even dimensions
    - **TestMaskCentralCross** - Masking operations
    - **TestStarToBinaryFileConversion** - Parameter serialization
    - **TestElectronExposureFilter** - Dose calculations
    - **TestEmpiricalDistribution** - Statistical calculations
    - **TestSumOfSquaresFourierAndFFTNormalization** - DFT normalization
    - **TestRandomVariableFunctions** - Noise generation from multiple distributions
    - **TestIntegerShifts** - Rotation and integer displacement

    ### Testing Architecture

    - **Test harness**: Catch2 framework
    - **Artifacts**: Test data files in test artifacts directory
    - **Dependencies**: Many tests depend on MRC file I/O and FFT operations
    - **Coverage gaps**: Several methods lack tests (noted in warnings above)

    ### See Also

    - **unit-testing skill** - Modern cisTEMx unit testing practices
    - **gpu-test-debugger skill** - GPU test debugging workflow
    - **src/test/ directory** - Current test implementation

---

!!! warning "Legacy Documentation"
    This content was extracted from the cisTEM developmental-docs repository and may be outdated.
    It documents the original cisTEM testing framework. cisTEMx may have evolved significantly.
