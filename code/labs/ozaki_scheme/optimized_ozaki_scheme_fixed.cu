#include "ozaki_scheme_common.cuh"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    try {
        return ozaki_scheme::run_and_report(ozaki_scheme::Variant::kFixed, argc, argv);
    } catch (const std::exception& exc) {
        std::cerr << "optimized_ozaki_scheme_fixed failed: " << exc.what() << std::endl;
        return 1;
    }
}
