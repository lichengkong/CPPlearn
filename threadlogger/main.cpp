#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

class Logger
{
public:
	Logger() : stop(false)
	{
		worker = std::thread(&Logger::process, this);
	}

	~Logger()
	{
		{
			std::unique_lock<std::mutex> lock(mtx);
			stop = true;
		}
		cv.notify_all();
		worker.join();
	}

	void log(const std::string &msg)
	{
		{
			std::unique_lock<std::mutex> lock(mtx);
			queue.push(msg);
		}
		cv.notify_one();
	}

private:
	void process()
	{
		while (true)
		{
			std::unique_lock<std::mutex> lock(mtx);
			cv.wait(lock, [this]
					{ return !queue.empty() || stop; });

			if (stop && queue.empty())
				return;

			auto msg = queue.front();
			queue.pop();
			lock.unlock();

			std::cout << "msg:  " << msg << std::endl;
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
	}

	std::queue<std::string> queue;
	std::mutex mtx;
	std::condition_variable cv;
	std::thread worker;
	std::atomic<bool> stop;
};

int main()
{
	Logger logger;

	std::thread t1(
		[&logger]
		{
			for (size_t i = 0; i < 5; i++)
			{
				logger.log("Thread 1 : Log " + std::to_string(i));
			}
		});

	std::thread t2(
		[&logger]
		{
			for (size_t i = 0; i < 5; i++)
			{
				logger.log("Thread 2 : Log " + std::to_string(i));
			}
		});

	t1.join();
	t2.join();
	return 0;
}