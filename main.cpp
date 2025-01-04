#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <cxxplot/cxxplot>

class DataLoader
{
public:
    DataLoader(std::string filename, bool add_noise)
    {
        std::vector<std::vector<double>> data;
        std::fstream file(filename);
        std::string line;

        while (getline(file, line))
        {
            std::stringstream ss(line);
            std::string value;
            std::vector<double> row;

            while (getline(ss, value, ','))
            {
                row.push_back(std::stod(value));
            }

            data.push_back(row);
        }

        // introducing gaussian noise to the dataset

        if (add_noise)
        {
            std::random_device rd;
            std::default_random_engine generator(rd());
            float mean = 0, std = 0.1;
            std::normal_distribution<double> distribution(mean, std);

            for (int i{}; i < data.size(); ++i)
            {
                for (int j{}; j < data[0].size() - 1; ++j)
                {
                    data[i][j] += distribution(generator);
                }
            }
            
            write_to_csv("./data/noisy_dataset.csv", data);
        }

        // robust scaler

        for (int i{}; i < data[0].size(); ++i)
        {
            std::vector<double> col_vals;

            for (int j{}; j < data.size() - 1; ++j)
            {
                col_vals.push_back(data[j][i]);
            }

            std::sort(col_vals.begin(), col_vals.end());
            int n = col_vals.size();
            double median = ((n % 2 != 0) ? (col_vals[n / 2]) : ((col_vals[(n / 2) - 1] + col_vals[n / 2]) / 2));
            double iqr = col_vals[3 * n / 4] - col_vals[n / 4];

            for (int j{}; j < data.size(); ++j)
            {
                data[j][i] = (data[j][i] - median) / iqr;
            }
        }

        write_to_csv("./data/normalised_dataset.csv", data);

        dataset.open("./data/normalised_dataset.csv");
    }

    bool is_eof()
    {
        return dataset.eof();
    }

    int get_input_target(std::vector<double> &input, std::vector<double> &target)
    {
        input.clear();
        target.clear();
        std::string line;

        if (std::getline(dataset, line))
        {
            std::stringstream ss(line);
            std::string row;

            while (std::getline(ss, row, ','))
            {
                input.push_back(std::stod(row));
            }

            target.push_back(input.back());
            input.pop_back();
            return input.size();
        }

        return 0;
    }

private:
    std::fstream dataset;

    void write_to_csv(std::string filename, std::vector<std::vector<double>> data)
    {
        std::ofstream out(filename);

        for (int i{}; i < data.size(); ++i)
        {
            for (int j{}; j < data[i].size(); ++j)
            {
                out << data[i][j];

                if (j != data[i].size() - 1)
                {
                    out << ',';
                }
            }

            if (i != data.size() - 1)
            {
                out << '\n';
            }
        }
    }
};

struct Edge
{
    double weight, delta_weight;
};

class Node;

typedef std::vector<Node> Layer;

class Node
{
public:
    Node(int n_outputs, int idx)
    {
        for (int i{}; i < n_outputs; ++i)
        {
            node_output_weight.push_back(Edge());
            node_output_weight.back().weight = rand() / double(RAND_MAX);
        }

        id = idx;
    }

    void set_output(double val)
    {
        node_output = val;
    }

    double get_output()
    {
        return node_output;
    }

    std::vector<Edge> get_output_weight()
    {
        return node_output_weight;
    }

    void feed_forward(Layer &prev_layer, int flag)
    {
        double sum{};

        for (int i{}; i < prev_layer.size(); ++i)
        {
            sum += prev_layer[i].node_output * prev_layer[i].node_output_weight[id].weight;
        }

        node_output = ((flag == 1) ? (activation(output_activation, sum)) : (activation(activation_fn, sum)));
    }

    void calc_out_grad(double target)
    {
        double error = target - node_output;
        grad = error * activation_derivative(activation_fn, node_output);
    }

    void calc_hidden_grad(Layer &next_layer)
    {
        double sum{};

        for (int i{}; i < next_layer.size() - 1; ++i)
        {
            sum += node_output_weight[i].weight * next_layer[i].grad;
        }

        grad = sum * activation_derivative(activation_fn, node_output);
    }

    void update_weight(Layer &prev_layer)
    {
        for (int i{}; i < prev_layer.size() - 1; ++i)
        {
            Node &node = prev_layer[i];
            double old_delta_weight = node.node_output_weight[id].delta_weight;
            double new_delta_weight = (learn_rate * node.node_output * grad) + (momentum * old_delta_weight);
            node.node_output_weight[id].delta_weight = new_delta_weight;
            node.node_output_weight[id].weight += new_delta_weight;
        }
    }

    std::vector<Edge> print_output_weights()
    {
        return node_output_weight;
    }

private:
    int id;
    double node_output, grad;
    std::vector<Edge> node_output_weight;

    // hyper parameters

    double learn_rate = 0.01, momentum = 0.3;
    std::string activation_fn = "relu", output_activation = "sigmoid";

    double activation(std::string function, double x)
    {
        if (function == "relu")
        {
            if (x > 0)
            {
                return x;
            }
            else
            {
                return 0;
            }
        }

        else if (function == "sigmoid")
        {
            return (1.0 / (1.0 + std::exp(-x)));
        }

        else
        {
            return 0;
        }
    }

    double activation_derivative(std::string function, double x)
    {
        if (function == "relu")
        {
            return ((x > 0) ? 1 : 0);
        }

        else if (function == "sigmoid")
        {
            return x * (1 - x);
        }

        else
        {
            return 0;
        }
    }
};

class NeuralNetwork
{
public:
    NeuralNetwork(std::vector<int> &topology)
    {
        for (int i{}; i < topology.size(); ++i)
        {
            layers.push_back(Layer());
            int n_outputs = ((i == topology.size() - 1) ? 1 : (topology[i + 1]));

            if (i == 0)
            {
                std::cout << "\n\nInput Layer";
            }
            else if (i == topology.size() - 1)
            {
                std::cout << "\n\nOutput Layer";
            }
            else
            {
                std::cout << "\n\nLayer " << i + 1;
            }

            for (int j{}; j <= topology[i]; ++j)
            {
                layers.back().push_back(Node(n_outputs, j));

                if (j < topology[i])
                {
                    std::cout << "\nNode " << j + 1;
                    std::cout << " - \tAssociated Weights : \t ";

                    for (const auto &edge : layers[i][j].print_output_weights())
                    {
                        std::cout << edge.weight << " \t";
                    }
                }
                else
                {
                    std::cout << "\nBias Node";
                }
            }

            layers.back().back().set_output(1.0);
        }

        std::cout << std::endl;

        // the constructor automatically intializes weights for the output layer's nodes too (will not get involved during backprop), and also inserts a bias node in the output layer
    }

    void feed_forward(std::vector<double> input_vals)
    {
        assert(input_vals.size() == layers[0].size() - 1);

        for (int i{}; i < input_vals.size(); ++i)
        {
            layers[0][i].set_output(input_vals[i]);
        }

        int flag{};

        for (int i = 1; i < layers.size(); ++i)
        {
            Layer &prev_layer = layers[i - 1];

            if (i == layers.size() - 1)
            {
                flag = 1;
            }

            for (int j{}; j < layers[i].size() - 1; ++j)
            {
                layers[i][j].feed_forward(prev_layer, flag);
            }
        }
    }

    void calc_output(std::vector<double> &output)
    {
        output.clear();

        for (int i{}; i < layers.back().size() - 1; ++i)
        {
            output.push_back(layers.back()[i].get_output());
        }
    }

    void backprop(std::vector<double> &target, double &loss)
    {
        Layer &output_layer = layers.back();
        error = 0;

        // BCE loss

        for (int i{}; i < output_layer.size() - 1; ++i)
        {
            error += -((target[i] * std::log(output_layer[i].get_output())) + ((1 - target[i]) * std::log(1 - output_layer[i].get_output())));
        }

        loss += error;

        for (int i{}; i < output_layer.size() - 1; ++i)
        {
            output_layer[i].calc_out_grad(target[i]);
        }

        for (int i = layers.size() - 2; i > 0; --i)
        {
            Layer &hidden_layer = layers[i], &next_layer = layers[i + 1];

            for (int j{}; j < hidden_layer.size(); ++j)
            {
                hidden_layer[j].calc_hidden_grad(next_layer);
            }
        }

        for (int i = layers.size() - 1; i > 0; --i)
        {
            Layer &curr_layer = layers[i], &prev_layer = layers[i - 1];

            for (int j{}; j < curr_layer.size() - 1; ++j)
            {
                curr_layer[j].update_weight(prev_layer);
            }
        }
    }

    std::vector<Layer> get_layer()
    {
        return layers;
    }

    void print_weights()
    {
        for (int i = 0; i < layers.size(); ++i)
        {
            if (i == 0)
            {
                std::cout << "\n\nInput Layer";
            }

            else if (i == layers.size() - 1)
            {
                std::cout << "\n\nOutput Layer";
            }

            else
            {
                std::cout << "\n\nLayer " << i + 1;
            }

            for (int j = 0; j < layers[i].size() - 1; ++j)
            {
                std::cout << "\nNode " << j + 1 << " - \tAssociated weights : \t ";

                for (const auto &edge : layers[i][j].print_output_weights())
                {
                    std::cout << edge.weight << " \t";
                }
            }
        }
    }

private:
    std::vector<Layer> layers;
    double error, delta_weight;
};

int main(int argc, char **argv)
{
    return cxxplot::exec(
        argc, argv, [&]()
        {
            bool add_noise = (std::string(argv[2]) == "gaussian_noise=on") ? true: false;

            std::vector<int> topology;
            std::cout << "\n"
                    << atoi(argv[3]) << " layers";

            int n{};
            for (n = 4; n < 4 + atoi(argv[3]); ++n)
            {
                topology.push_back(atoi(argv[n]));
            }

            NeuralNetwork net(topology);
            std::vector<double> input, output, target;
            int epochs = atoi(argv[n++]), epoch{};

            std::vector<cxxplot::point2d> loss_vs_epoch{};

            while (epoch < epochs)
            {
                DataLoader data(argv[1], add_noise); 
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

            cxxplot::plot
            ( 
                loss_vs_epoch,
                cxxplot::named_parameters::window_title_ = "Training Loss vs Epoch", cxxplot::named_parameters::xlabel_ = "Epoch", 
                cxxplot::named_parameters::ylabel_ = "Training Loss"
            );

            return 0; });
}