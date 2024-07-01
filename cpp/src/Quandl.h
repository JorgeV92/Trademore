#ifndef QUANDL_H
#define QUANDL_H

#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class Quandl {
public:
    Quandl(const std::string& api_key)
        : api_key_(api_key) {}

    json getHistoricalData(const std::string& dataset, const std::string& start_date = "", const std::string& end_date = "");

private:
    std::string api_key_;
};

#endif // QUANDL_H
