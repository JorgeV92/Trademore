#include "Alpaca.h"
#include <cpr/cpr.h>
#include <iostream>

json Alpaca::getHistoricalData(const std::string& symbol, const std::string& start, const std::string& end) {
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