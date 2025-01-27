#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>


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