
#include <iostream>
#include "rc.hpp"


RegionClassifierModel::RegionClassifierModel(const std::string& model_path) 
: device(torch::kCPU) 
{
    load_model(model_path);
    set_device();
}

void RegionClassifierModel::load_model(const std::string& model_path) 
{
    try {
        model = torch::jit::load(model_path);
        std::cout << "Model loaded successfully from " << model_path << "\n";
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what_without_backtrace() << "\n";
        throw;
    }
}

void RegionClassifierModel::set_device() {
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout << "Using CUDA\n";
    } else {
        device = torch::Device(torch::kCPU);
        std::cout << "Using CPU\n";
    }
    model.to(device);
}

torch::Tensor RegionClassifierModel::run_inference(torch::Tensor input) {
    input = input.to(device);  
    return model.forward({input}).toTensor();
}

int main(int argc, const char* argv[]) 
{
    if (argc != 2) {
        std::cerr << "usage: ts-infer <path-to-exported-model>\n";
        return -1;
    }


    try 
    {
        RegionClassifierModel model(argv[1]);
        torch::Tensor input = torch::randn({1, 3, 224, 224});
        torch::Tensor output = model.run_inference(input);

        std::cout << "Output tensor: " << output << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << "\n";
        return -1;
    }

    return 0;
}
