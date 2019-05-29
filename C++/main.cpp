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

    // ------------------------------- ENCODER

    network_reader_e.ReadNetwork(FLAGS_m_e);
    std::string binFileName_e = fileNameNoExt(FLAGS_m_e) + ".bin";
    network_reader_e.ReadWeights(binFileName_e);

    network_reader_e.getNetwork().setBatchSize(1);
    CNNNetwork network_e = network_reader_e.getNetwork();

    std::vector<std::string> inputsName_e;
    InputsDataMap inputsInfo_e(network_e.getInputsInfo());
    std::size_t input_w = 0, input_h = 0; // avoid uninitial error

    for (auto & item : inputsInfo_e)
    {
        item.second->setPrecision(Precision::U8);
        inputsName_e.push_back(item.first);
        if (inputsName_e.size() == 1)
        {
            input_w = item.second->getDims()[0];
            input_h = item.second->getDims()[1];
        }
    }

    std::vector<std::string> outputsName_e;
    OutputsDataMap outputsInfo_e(network_e.getOutputsInfo());
    for (auto & item : outputsInfo_e)
    {
        item.second->setPrecision(Precision::FP32);
        outputsName_e.push_back(item.first);
    }
    
    // ------------------------------- DECODER

    network_reader_d.ReadNetwork(FLAGS_m_d);
    std::string binFileName_d = fileNameNoExt(FLAGS_m_d) + ".bin";
    network_reader_d.ReadWeights(binFileName_d);

    network_reader_d.getNetwork().setBatchSize(1);
    network_reader_d.getNetwork().addOutput("60");
    CNNNetwork network_d = network_reader_d.getNetwork();

    std::vector<std::string> inputsName_d;
    std::vector<std::size_t> inputsSize_d;
    // std::vector<InputInfo::Ptr> inputsData_d;
    // std::vector<DataPtr> inputs;
    
    InputsDataMap inputsInfo_d(network_d.getInputsInfo());
    for (auto & item : inputsInfo_d)
    // for (std::size_t i = 0; i < inputsInfo_d.size(); i++)
    {   
        item.second->setPrecision(Precision::FP32);
        inputsName_d.push_back(item.first);
        
        auto dims = item.second->getDims();
        int dims_t = 1;
        for(int i = 0; i < dims.size(); i++)
            dims_t *= dims[i];
        inputsSize_d.push_back(dims_t);
    }

    std::vector<std::string> outputsName_d;
    OutputsDataMap outputsInfo_d(network_d.getOutputsInfo());
    for (auto & item : outputsInfo_d)
    {
        item.second->setPrecision(Precision::FP32); 
        outputsName_d.push_back(item.first);
    }

    // --------------------------------- EXECUTION ENCODER

    auto executable_network_e = plugin_e.LoadNetwork(network_e, {});
    auto infer_request_e = executable_network_e.CreateInferRequest();
    
    auto input_e = infer_request_e.GetBlob(inputsName_e[0]);
    auto input_data_e = input_e->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();

    cv::Mat image = cv::imread(FLAGS_i);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(input_h, input_w), cv::INTER_LANCZOS4);
    
    for(std::size_t pid = 0; pid < input_h * input_w; pid++)
    {
        for (std::size_t ch = 0; ch < 3; ch++)
        {
            input_data_e[ch * input_h * input_w + pid] = image.at<cv::Vec3b>(pid)[ch];
        }        
    }
    
    infer_request_e.Infer();

    auto output_e = infer_request_e.GetBlob(outputsName_e[0]);
    auto output_data_point_e = output_e->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();

    // --------------------------------- EXECUTION DECODER

    auto executable_network_d = plugin_d.LoadNetwork(network_d, {});
    auto infer_request_d = executable_network_d.CreateInferRequest();
    
    std::vector<int> sentence_ids;
    
    for (int text = 0; text < FLAGS_tl; text++)
    {
        if (text == 0)
        {
            for (std::size_t pid = 0; pid < inputsSize_d[0]; pid++)
                infer_request_d.GetBlob(inputsName_d[0])->
                    buffer().as<PrecisionTrait<Precision::FP32>::value_type*>()[pid] = 
                infer_request_e.GetBlob(outputsName_e[0])->
                    buffer().as<PrecisionTrait<Precision::FP32>::value_type*>()[pid];
            
            for (std::size_t pid = 0; pid < inputsSize_d[1]; pid++)
            {
                infer_request_d.GetBlob(inputsName_d[1])->
                    buffer().as<PrecisionTrait<Precision::FP32>::value_type*>()[pid] = 0.f;
                infer_request_d.GetBlob(inputsName_d[2])->
                    buffer().as<PrecisionTrait<Precision::FP32>::value_type*>()[pid] = 0.f;
            }

        }
        else
        {
            for (std::size_t pid = 0; pid < inputsSize_d[0]; pid++)
                infer_request_d.GetBlob(inputsName_d[0])->
                    buffer().as<PrecisionTrait<Precision::FP32>::value_type*>()[pid] =
                infer_request_d.GetBlob(outputsName_d[2])->
                    buffer().as<PrecisionTrait<Precision::FP32>::value_type*>()[pid];
            
            for (std::size_t pid = 0; pid < inputsSize_d[1]; pid++)
            {
                infer_request_d.GetBlob(inputsName_d[1])->
                    buffer().as<PrecisionTrait<Precision::FP32>::value_type*>()[pid] =
                infer_request_d.GetBlob(outputsName_d[0])->
                    buffer().as<PrecisionTrait<Precision::FP32>::value_type*>()[pid];
        
                infer_request_d.GetBlob(inputsName_d[2])->
                    buffer().as<PrecisionTrait<Precision::FP32>::value_type*>()[pid] =
                infer_request_d.GetBlob(outputsName_d[1])->
                    buffer().as<PrecisionTrait<Precision::FP32>::value_type*>()[pid];
            }
        }

        infer_request_d.Infer();
        sentence_ids.push_back(
            infer_request_d.GetBlob(outputsName_d[3])
                ->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>()[0]
        );

        if (sentence_ids.back() == 2)
            break;
    }

    std::vector<std::string> sampled_caption;
    int real_lenght = sentence_ids.size();

    std::cout << "Caption : " << "\n";
    for (int i = 0; i < real_lenght; i++)
    {
        sampled_caption.push_back(vocab[sentence_ids[i]]);
        if (i > 0 && sentence_ids[i] != 2)     
            std::cout << sampled_caption[i] << " ";
    }
    std::cout << "\n";

    cv::imshow("Image", cv::imread(FLAGS_i));
    const int key = cv::waitKey(0);
    if (27 == key)  // Esc
        cv::destroyAllWindows();
    
    return 0;
}//MAIN



