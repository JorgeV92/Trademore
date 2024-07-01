#include <cpr/cpr.h>
#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>
#include <iostream>


using json = nlohmann::json;

class Alpaca {
public:
    Alpaca(const std::string& api_key, const std::string& api_secret)
        : api_key_(api_key), api_secret_(api_secret) {}

    json getHistoricalData(const std::string& symbol, const std::string& start, const std::string& end) {
        std::string url = "https://data.alpaca.markets/v2/stocks/" + symbol + "/bars?start=" + start + "&end=" + end + "&timeframe=1Day";
        
        auto response = cpr::Get(cpr::Url{url},
                                 cpr::Header{{"APCA-API-KEY-ID", api_key_},
                                             {"APCA-API-SECRET-KEY", api_secret_}});
        
        if (response.status_code != 200) {
            std::cerr << "Error: " << response.status_code << " - " << response.error.message << std::endl;
            return nullptr;
        }

        return json::parse(response.text);
    }

private:
    std::string api_key_;
    std::string api_secret_;
};

int main(int argc, char** argv) {

    std::cout << "compiled Alpaca\n"; 

    std::string api_key;
    std::string api_secret;
    std::string symbol;
    std::string start;
    std::string end;

    CLI::App app{"Alpaca Market Data Fetcher"};

    app.add_option("--api-key", api_key, "Alpaca API Key")->required();
    app.add_option("--api-secret", api_secret, "Alpaca API Secret")->required();
    app.add_option("--symbol", symbol, "Stock Symbol")->required();
    app.add_option("--start", start, "Start Date (YYYY-MM-DD)")->required();
    app.add_option("--end", end, "End Date (YYYY-MM-DD)")->required();

    CLI11_PARSE(app, argc, argv);

    Alpaca alpaca(api_key, api_secret);
    json data = alpaca.getHistoricalData(symbol, start, end);

    if (!data.is_null()) {
        std::cout << data.dump(4) << std::endl;
    }
    
    std::cout << "end of program\n";

    return 0;
}