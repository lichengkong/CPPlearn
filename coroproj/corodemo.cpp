#include <coroutine>
#include <iostream>
#include <stdexcept>
#include <thread>


template <typename T>
struct coro_ret
{
    struct promise_type;
    using handle_type = std::coroutine_handle<promise_type>;

    handle_type coro_handle_;

    coro_ret(handle_type h)
        : coro_handle_(h)
    {
    }
    coro_ret(const coro_ret&) = delete;
    coro_ret(coro_ret&& s)
        : coro_handle_(s.coro_)
    {
        s.coro_handle_ = nullptr;
    }
    ~coro_ret()
    {

        if (coro_handle_)
            coro_handle_.destroy();
    }
    coro_ret& operator=(const coro_ret&) = delete;
    coro_ret& operator=(coro_ret&& s)
    {
        coro_handle_ = s.coro_handle_;
        s.coro_handle_ = nullptr;
        return *this;
    }


    bool move_next()
    {
        coro_handle_.resume();
        return coro_handle_.done();
    }

    T get()
    {
        return coro_handle_.promise().return_data_;
    }

    struct promise_type
    {
        promise_type() = default;
        ~promise_type() = default;


        auto get_return_object()
        {
            return coro_ret<T>{handle_type::from_promise(*this)};
        }

        auto initial_suspend()
        {
            //return std::suspend_never{};
            return std::suspend_always{};
        }

        void return_value(T v)
        {
            return_data_ = v;
            return;
        }
        
        auto yield_value(T v)
        {
            std::cout << "yield_value invoked." << std::endl;
            return_data_ = v;
            return std::suspend_always{};
        }

        auto final_suspend() noexcept
        {
            std::cout << "final_suspend invoked." << std::endl;
            return std::suspend_always{};
        }
        
        void unhandled_exception()
        {
            std::exit(1);
        }

        T return_data_;
    };
};

coro_ret<int> coroutine_7in7out()
{

    std::cout << "Coroutine co_await std::suspend_never" << std::endl;
    //co_await std::suspend_never{} ???????
    co_await std::suspend_never{};
    std::cout << "Coroutine co_await std::suspend_always" << std::endl;
    co_await std::suspend_always{};

    std::cout << "Coroutine stage 1 ,co_yield" << std::endl;
    co_yield 101;
    std::cout << "Coroutine stage 2 ,co_yield" << std::endl;
    co_yield 202;
    std::cout << "Coroutine stage 3 ,co_yield" << std::endl;
    co_yield 303;
    std::cout << "Coroutine stage end, co_return" << std::endl;
    co_return 808;
}

int main(int argc, char* argv[])
{
    bool done = false;
    std::cout << "Start coroutine_7in7out ()\n";
    auto c_r = coroutine_7in7out();
    std::cout << "Coroutine " << (done ? "is done " : "isn't done ")
        << "ret =" << c_r.get() << std::endl;
    done = c_r.move_next();
    std::cout << "Coroutine " << (done ? "is done " : "isn't done ")
        << "ret =" << c_r.get() << std::endl;
    done = c_r.move_next();

    std::cout << "Coroutine " << (done ? "is done " : "isn't done ")
        << "ret =" << c_r.get() << std::endl;
    done = c_r.move_next();
    std::cout << "Coroutine " << (done ? "is done " : "isn't done ")
        << "ret =" << c_r.get() << std::endl;
    done = c_r.move_next();
    std::cout << "Coroutine " << (done ? "is done " : "isn't done ")
        << "ret =" << c_r.get() << std::endl;
    done = c_r.move_next();
    std::cout << "Coroutine " << (done ? "is done " : "isn't done ")
        << "ret =" << c_r.get() << std::endl;
    return 0;
}