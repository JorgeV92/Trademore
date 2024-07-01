#ifndef ALPACA_H
#define ALPACA_H

#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class Alpaca {
public:
    Alpaca(const std::string& api_key, const std::string& api_secret)
        : api_key_(api_key), api_secret_(api_secret) {}

    json getHistoricalData(const std::string& symbol, const std::string& start, const std::string& end);
    
private:
    std::string api_key_;
    std::string api_secret_;
};


#endif // ALPACA_H