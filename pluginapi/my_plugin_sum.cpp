#include <boost/config.hpp> // for BOOST_SYMBOL_EXPORT
#include "my_plugin_sum.h"
#include <iostream>

namespace my_namespace
{
    my_plugin_sum::my_plugin_sum()
    {
        std::cout << "Constructing my_plugin_sum" << std::endl;
    }

    std::string my_plugin_sum::name()
    {
        return "sum";
    }

    float my_plugin_sum::calculate(float x, float y)
    {
        return x + y;
    }

    my_plugin_sum::~my_plugin_sum()
    {
        std::cout << "Destructing my_plugin_sum ;o)" << std::endl;
    }

    // Exporting `my_namespace::plugin` variable with alias name `plugin`
    // (Has the same effect as `BOOST_DLL_ALIAS(my_namespace::plugin, plugin)`)
    extern "C" BOOST_SYMBOL_EXPORT my_plugin_sum plugin;

    
    my_plugin_sum plugin;

} // namespace my_namespace