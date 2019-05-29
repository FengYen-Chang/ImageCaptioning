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

    // ------------------------------- ENCODER

    network_reader_e.ReadNetwork(FLAGS_m_e);
    std::string binFileName_e = fileNameNoExt(FLAGS_m_e) + ".bin";
    network_reader_e.ReadWeights(binFileName_e);

    network_reader_e.getNetwork().setBatchSize(1);
    CNNNetwork network_e = network_reader_e.getNetwork();

    std::vector<std::string> inputsName_e;
    InputsDataMap inputsInfo_e(network_e.getInputsInfo());
    size_t input_w, input_h;
    for (auto & item : inputsInfo_e)
    {
        item.second->setPrecision(Precision::U8);
        inputsName_e.push_back(item.first);
        input_w = item.second->getDims()[0];
        input_h = item.second->getDims()[1];
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
    network_reader_d.getNetwork().addOutput("60")
    CNNNetwork network_d = network_reader_d.getNetwork();

    std::size_t hidden_state_size;
    
    std::vector<std::string> inputsName_d;
    std::vector<InputInfo::Ptr> inputsData_d;
    // std::vector<DataPtr> inputs;
    InputsDataMap inputsInfo_d(network_d.getInputsInfo());
    // for (auto & item : inputsInfo_d)
    for (std::size_t i = 0; i < inputsInfo_d.size(); i++)
    {   
        inputsInfo_d[i].second->setPrecision(Precision::FP32);
        inputsName_d.push_back(inputsInfo_d[i].first);
        
        if (i == 1)
            hidden_state_size = inputsInfo_d[i].second->getDims()[0] * 
                                inputsInfo_d[i].second->getDims()[1] * 
                                inputsInfo_d[i].second->getDims()[2];
    }

    std::vector<std::string> outputsName_d;
    OutputsDataMap outputsInfo_d(network_d.getOutputsInfo());
    for (auto & item : outputsInfo_d)
    {
        item.second->setPrecision(Precision::FP32); 
        outputsName_d.push_back(item.first);
    }

    // --------------------------------- EXECUTION ENCODER

    auto executable_network_e = plugin.LoadNetwork(network_e, {});
    auto infer_request_e = executable_network_e.CreateInferRequest();
    
    auto input_e = infer_request.GetBlob(inputsName_e[0]);
    auto input_data_e = input_e->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();

    cv::Mat image = cv::imread(FLAGS_i);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(input_h, input_w), INTER_LANCZOS4);

    size_t image_size = input_h * input_w;
    
    for(size_t pid = 0; pid < image_size; pid++)
    {
        for (size_t ch = 0; ch < 3; ch++)
        {
            input_data[ch * image_size + pid] = image.at<cv::Vec3b>(pid)[ch];
        }        
    }
    
    infer_request_e.Infer();

    auto output_e = infer_request_e.GetBlob(outputsName_e[0]);
    auto output_data_point_e = output_e->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();

    // --------------------------------- EXECUTION DECODER

    auto executable_network_d = plugin.LoadNetwork(network_d, {});
    auto infer_request_d = executable_network_d.CreateInferRequest();
    
    std::vector<int> sentence_ids;
    
    // cv::Mat init_state_h = cv::zeros(hidden_state_size[0], hidden_state_size[1], CV_32FC1);
    // cv::Mat init_state_c = cv::zeros(hidden_state_size[0], hidden_state_size[1], CV_32FC1);
    // (hidden_state_size[0], hidden_state_size[1], CV_32FC1, cv::S);

    for (int text = 0; text < FLAGS_tl; text++)
    {
        if (text == 0)
        {
            infer_request_d.GetBlob(inputsName_d[0])
                ->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>() = 
                infer_request_e.GetBlob(outputsName_e[0])
                    ->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
            
            for (size_t pid = 0; pid < hidden_state_size; pid++)
            {
                infer_request_d.GetBlob(inputsName_d[1])
                    ->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>()[pid] = 0.f;
                infer_request_d.GetBlob(inputsName_d[2])
                    ->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>()[pid] = 0.f;
            }

        }
        else
        {
            infer_request_d.GetBlob(inputsName_d[0])
                ->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>() =
                infer_request_d.GetBlob(outputsName_d[2])
                    ->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();

            infer_request_d.GetBlob(inputsName_d[1])
                ->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>() =
                infer_request_d.GetBlob(outputsName_d[0])
                    ->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
        
            infer_request_d.GetBlob(inputsName_d[2])
                ->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>() =
                infer_request_d.GetBlob(outputsName_d[1])
                    ->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
        }

        infer_request_d.Infer();
        sentence_ids.push_back(
            infer_request_d.GetBlob(outputsName_d[3])
                ->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>()[0];
        );

        if (sentence_ids.back() == 2)
            break;
    }

    for (int i = 0; i < sentence_ids.size(); i++)
        cout << sentence_ids[i] << "\t";
    cout << "\n";

    std::vector<std::string> sampled_caption;
    for (int i = 0; i < sentence_ids.size(); i++)
        sampled_caption.push_back(vocab[sentence_ids[i]]);

    for (int i = 0; i < sampled_caption.size(); i++)
        cout << sampled_caption[i] << " ";
    cout << "\n";

    
    return 0;
}//MAIN



