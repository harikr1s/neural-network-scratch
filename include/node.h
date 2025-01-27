#include <iostream>
#include <vector>
#include <cmath>

struct Edge
{
    double weight, delta_weight;
};

class Node;

typedef std::vector<Node> Layer;

class Node
{
public:
    Node(int n_outputs, int idx, double learn_rate)
    {
        for (int i{}; i < n_outputs; ++i)
        {
            node_output_weight.push_back(Edge());
            node_output_weight.back().weight = rand() / double(RAND_MAX);
        }

        id = idx;
        this->learn_rate = learn_rate;
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

    double learn_rate, momentum = 0.3;
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