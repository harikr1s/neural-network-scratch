#include <iostream>
#include <vector>
#include <cassert>

#include "node.h"

class NeuralNetwork
{
public:
    NeuralNetwork(std::vector<int> &topology, double learn_rate)
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
                layers.back().push_back(Node(n_outputs, j, learn_rate));

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