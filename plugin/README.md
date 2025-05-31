# plugincpp
a modern c++ plugin library. It is in experiment.

Dependency:

1. Boost.dll, boost.filesystem
2. msgpack
<!-- 3. cinatra(http server) -->
4. rest_rpc(rpc server)
5. asio

# 项目介绍
祁宇 在2020 purecpp 大会上的分享  [plugincpp](https://github.com/qicosmos/plugincpp/)



# 需求
我司的一个算法部署框架，依赖不解耦，部署特别麻烦，一直想要实现一个真正的插件系统，只用把新算法的动态库传入服务端，便可以顺利出版本，并进行rpc调用。

plugincpp 这个项目解决了c++插件开发的关键问题，具体可以看有关ppt。原始项目无法跑起来，也没有维护，花了三天时间研究了一下想要基于yalantinglibs实现一个新版本的插件系统，未果，最后还是把本项目先跑通。



# 原理
1. rpc不像http，可以使用网址路由到相应的服务，rpc客户端需要知道服务的接口，然后才能调用。
2. 当新增一个服务时，需要修改接口和所有插件的实现，当我们开发了应用市场可以给用户自定义时，即使暴露接口也难以覆盖需求，因此需要一个仅仅根据符号就能查找到具体服务的方法。
3. 这里在注册服务时进行了类型擦除，在路由查找服务时进行了类型萃取和还原，主要是通过返回值和参数来查找的，跟接口的名字无关。这里比较神奇，还没有搞清楚。#TODO
4. 接口的参数统一为指针和大小两个参数，这样通过序列化和反序列化就可以实现任意类型的参数和返回值。路由表中的实现。
5. 项目中使用http post方法将插件注册进一个哈希表，这里为了简便直接写死了，主要原因是cinatra的版本不对，找不到合适的版本，试了下老版本也不行，这里的原因e可能是该项目本身就是个演示demo...
6. 项目主要依赖rest_rpc,他提供了编译时反射，不用手写序列化，特别是proto的序列化反序列化，写的难受容易出问题。
7. 其他依赖原本都拷到了thirdparty中，但是git容易爆，加之本身就是通用的依赖，就直接装到系统目录了（这是我家里的开发机）

# TODO
1. 熟悉clang，强制使用C++20 yalantinglibs
2. 熟悉rest_rpc coro_rpc的原理，尤其这里使用的msgpac看，而后使用struct_pack序列化
3. 将依赖改为yalantinglibs，协程版本
4. 看能不能改造算法平台