#include <cxxplot/cxxplot>
#include <getopt.h>

#include "../include/dataloader.h"
#include "../include/neuralnetwork.h"

// to delete all previously written files inside "data" except the dataset

void delete_data_directory_contents(const std::filesystem::path& dir, const std::filesystem::path& file_to_exclude)
{
    for (const auto& entry : std::filesystem::directory_iterator(dir)) 
    {
        if (entry.is_regular_file() && entry.path() != file_to_exclude)
        {
            std::filesystem::remove(entry.path());
        }
    }
}

int main(int argc, char **argv)
{
    return cxxplot::exec(argc, argv, [&]()
    {
        char *file_path = nullptr;
        bool add_noise = true;
        std::vector<int> topology;
        double learn_rate = 1e-2;
        int epochs = 100;

        struct option long_options[] = 
        {
            {"file", required_argument, nullptr, 'f'},    
            {"noise", required_argument, nullptr, 'n'},     
            {"topology", required_argument, nullptr, 't'},  
            {"learnrate", required_argument, nullptr, 'l'}, 
            {"epochs", required_argument, nullptr, 'e'},   
            {nullptr, 0, nullptr, 0}  
        };

        int opt;

        while((opt = getopt_long(argc, argv, "f:n:t:l:e:", long_options, NULL)) != -1) 
        {
            switch(opt) 
            {
                case 'f':   
                {
                    file_path = optarg;
                }
                break;

                case 'n':   
                {
                    add_noise = (std::string(optarg) == "true")? true: false;
                }
                break;    

                case 't':   
                {
                    std::stringstream ss(optarg);
                    int layer_size;
                    while (ss >> layer_size) 
                    {
                        topology.push_back(layer_size);
                        if (ss.peek() == ' ') 
                        {
                            ss.ignore();
                        }
                    }
                }
                break; 

                case 'l':
                {
                    learn_rate = std::stod(optarg);
                }
                break;

                case 'e':
                {
                    epochs = atoi(optarg);  
                }
                break;

                default:
                {
                    std::cerr << "Usage: " << argv[0] << " --file=<path/to/dataset> --noise=<true/false> --topology=<\"space-separated topology\"> --learnrate=<learning_rate> --epochs=<epochs>\n";
                    return 1;
                }
                
            }
        }

        // std::cout << "\n"
        //         << atoi(argv[3]) << " layers";

        NeuralNetwork net(topology, learn_rate);
        std::vector<double> input, output, target;
        int epoch{};

        std::vector<cxxplot::point2d> loss_vs_epoch{};

        while (epoch < epochs)
        {
            delete_data_directory_contents("data", file_path);
            DataLoader data(file_path, add_noise); 
            int iter{}, correct{};
            double loss{};

            while (!data.is_eof())
            {
                iter++;

                if (data.get_input_target(input, target) != topology[0])
                {
                    std::cout << "\nInput size and input layer size does not match!" << std::endl;
                    exit(0);
                }

                net.feed_forward(input);
                net.calc_output(output);

                if ((target[0] == 1 && output[0] >= 0.5) || (target[0] == 0 && output[0] < 0.5))
                {
                    correct++;
                }

                net.backprop(target, loss);
            }

            double epoch_loss = loss / iter, epoch_accuracy = (double)correct / iter;
            std::cout << "\n\nEpoch " << epoch + 1 << " \tTraining Loss = " << epoch_loss << "\t \tTraining Accuracy = " << epoch_accuracy * 100 << " %";
            loss_vs_epoch.push_back({(double)epoch + 1, epoch_loss});
            epoch++;
        }

        std::cout << "\n\nUpdated weights after epoch " << epoch;
        net.print_weights();
        std::cout << std::endl;

        if (!std::filesystem::is_directory("out") || !std::filesystem::exists("out")) 
        { 
            std::filesystem::create_directory("out"); 
        }
        net.save_weights("out/weights.bin");

        cxxplot::plot
        ( 
            loss_vs_epoch,
            cxxplot::named_parameters::window_title_ = "Training Loss vs Epoch", cxxplot::named_parameters::xlabel_ = "Epoch", 
            cxxplot::named_parameters::ylabel_ = "Training Loss"
        );

        return 0; 
    });
}