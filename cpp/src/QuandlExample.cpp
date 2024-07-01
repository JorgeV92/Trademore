#include <iostream>
#include "Quandl.h"
#include <CLI/CLI.hpp>

int main(int argc, char** argv) {

    std::cout << "Compiled Quandl\n"; 

    std::string api_key;
    std::string dataset;
    std::string start;
    std::string end;

    CLI::App app{"Quandl Market Data Fetcher"};

    app.add_option("--api-key", api_key, "Quandl API Key")->required();
    app.add_option("--dataset", dataset, "Dataset (e.g., WIKI/AAPL)")->required();
    app.add_option("--start", start, "Start Date (YYYY-MM-DD)");
    app.add_option("--end", end, "End Date (YYYY-MM-DD)");

    CLI11_PARSE(app, argc, argv);

    Quandl quandl(api_key);
    json data = quandl.getHistoricalData(dataset, start, end);

    if (!data.is_null()) {
        std::cout << data.dump(4) << std::endl;
    }
    
    std::cout << "End of program\n";

    return 0;
}
