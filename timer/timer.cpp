#include <iostream>
#include <map>
#include <chrono>
#include <thread>
#include <functional>

struct TimerTask {
    std::function<void()> task; 
    std::chrono::time_point<std::chrono::steady_clock> expirationTime; 

    TimerTask(std::function<void()> t, std::chrono::time_point<std::chrono::steady_clock> exp)
        : task(t), expirationTime(exp) {
    }
};

class TimerManager {
private:
    std::multimap<std::chrono::time_point<std::chrono::steady_clock>, TimerTask> taskQueue;

public:

    void addTask(std::function<void()> task, std::chrono::milliseconds delay) {
        auto expiration = std::chrono::steady_clock::now() + delay;
        taskQueue.insert({ expiration, TimerTask(task, expiration) });
    }


    void run() {
        while (!taskQueue.empty()) {
            auto now = std::chrono::steady_clock::now();
            auto it = taskQueue.begin();


            if (it->first <= now) {
                it->second.task();
                taskQueue.erase(it);
            }
            else {

                std::this_thread::sleep_until(it->first);
            }
        }
    }
};


void exampleTask() {
    std::cout << "Task executed at " << std::chrono::steady_clock::now().time_since_epoch().count() << std::endl;
}

int main() {
   TimerManager timerManager;


   timerManager.addTask(exampleTask, std::chrono::milliseconds(2000));
   timerManager.addTask(exampleTask, std::chrono::milliseconds(1000));
   timerManager.addTask(exampleTask, std::chrono::milliseconds(3000));


   timerManager.run();

   return 0;
}