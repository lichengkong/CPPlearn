#include <boost/config.hpp> // for BOOST_SYMBOL_EXPORT
#include "my_plugin_api.hpp"
#include <iostream>

namespace my_namespace
{

    class my_plugin_sum : public my_plugin_api
    {
    public:
        my_plugin_sum();

        std::string name();

        float calculate(float x, float y);

        ~my_plugin_sum();
    };

} // namespace my_namespace