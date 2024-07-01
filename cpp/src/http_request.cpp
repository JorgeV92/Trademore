#include <cpr/cpr.h>

#include <iostream>

int main() {
    cpr::Response r = cpr::Get(cpr::Url{"http://example.com"});

    std::cout << "Status code: " << r.status_code << std::endl;
    std::cout << "Reponse body:\n" << r.text << std::endl;

    return 0;
}
