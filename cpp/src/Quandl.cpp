#include "Quandl.h"
#include <cpr/cpr.h>
#include <iostream>

json Quandl::getHistoricalData(const std::string& dataset, const std::string& start_date, const std::string& end_date) {
    std::string url = "https://www.quandl.com/api/v3/datasets/" + dataset + ".json?api_key=" + api_key_;
    if (!start_date.empty()) {
        url += "&start_date=" + start_date;
    }
    if (!end_date.empty()) {
        url += "&end_date=" + end_date;
    }

    auto response = cpr::Get(cpr::Url{url});

    if (response.status_code != 200) {
        std::cerr << "Error: " << response.status_code << " - " << response.error.message << std::endl;
        return nullptr;
    }

    return json::parse(response.text);
}
