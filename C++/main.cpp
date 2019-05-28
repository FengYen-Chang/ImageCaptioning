/*******************************************************/
/*                                                     */
/*  Create and implement by Feng                       */
/*                                                     */
/*  Date:                                              */
/*      2019/05/27                                     */
/*          Implement the main function                */
/*                                                     */
/*******************************************************/


#include <iomanip>
#include <memory>
#include <string>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <ext_list.hpp>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>

#include "ie_plugin_config.hpp"
#include "hetero/hetero_plugin_config.hpp"

#include "src/argparse.h"
#include "src/vocab.hpp"

using namespace InferenceEngine;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_ni < 1) {
        throw std::logic_error("Parameter -ni should be greater than zero (default 1)");
    }

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m_d.empty()) {
        throw std::logic_error("Parameter -m_d is not set");
    }

    if (FLAGS_m_e.empty()) {
        throw std::logic_error("Parameter -m_e is not set");
    }

    if (FLAGS_l.empty()) {
        throw std::logic_error("Parameter -l is not set");
    }

    return true;
}

int main(int argc, char *argv[])
{
    // ------------------------------ Parsing and validation of input args ---------------------------------
    if (!ParseAndCheckCommandLine(argc, argv)) {
        return 0;
    }

    auto engine_Ptr = PluginDispatcher({"./lib", ""}).getPluginByDevice(FLAGS_d);

    InferencePlugin plugin_d(engine_Ptr), plugin_e(engine_Ptr);
    plugin_d.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());

    CNNNetReader network_reader_d, network_reader_e;
    
    // ------------------------------- DECODER

    network_reader_d.ReadNetwork(FLAGS_m_d);
    std::string binFileName_d = fileNameNoExt(FLAGS_m_d) + ".bin";
    network_reader_d.ReadWeights(binFileName_d);

    network_reader_d.getNetwork().setBatchSize(1);
    CNNNetwork network_d = network_reader_d.getNetwork();

    auto input_info_d = network_d.getInputsInfo().begin()->second;
    auto input_name_d = network_d.getInputsInfo().begin()->first;
    input_info_d->setInputPrecision(Precision::FP32);

    // ------------------------------- ENCODER

    network_reader_e.ReadNetwork(FLAGS_m_e);
    std::string binFileName_e = fileNameNoExt(FLAGS_m_e) + ".bin";
    network_reader_e.ReadWeights(binFileName_e);

    network_reader_e.getNetwork().setBatchSize(1);
    CNNNetwork network_e = network_reader_e.getNetwork();





    

}//MAIN



