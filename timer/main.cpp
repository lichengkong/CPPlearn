#include <sys/epoll.h>
#include <functional>
#include <chrono>
#include <set>
#include <memory>
#include <iostream>

using namespace std;

struct TimerNodeBase
{
	time_t expire;
	int64_t id;
};

// TimerNode 继承 TimerNodeBase
struct TimerNode:public TimerNodeBase
{
	// C++ 11特性，使用函数对象。降低拷贝消耗，提高效率
	using Callback = std::function<void(const TimerNode &node)>;
	Callback func;

	// 构造函数，只构造一次
	TimerNode(int64_t id,time_t expire,Callback func):func(func){
		this->id = id;
		this->expire = expire;

	}
};

// 基类引用，多态特性
bool operator<(const TimerNodeBase &lhd, const TimerNodeBase &rhd)
{
	if (lhd.expire < rhd.expire)
		return true;
	else if (lhd.expire > rhd.expire)
		return false;
	return lhd.id < rhd.id;
}

class Timer
{
public:
	static time_t GetTick()
	{
		/* C++ 11时间库chrono */
		//表示一个具体时间
		auto sc = chrono::time_point_cast<chrono::milliseconds>(chrono::steady_clock::now());
		auto tmp = chrono::duration_cast<chrono::milliseconds>(sc.time_since_epoch());
		return tmp.count();
	}

	TimerNodeBase AddTimer(time_t msec, TimerNode::Callback func)
	{
		time_t expire = GetTick() + msec;
		//避免拷贝、移动构造
		auto ele = timermap.emplace(GenID(), expire, func);

		return static_cast<TimerNodeBase>(*ele.first);
	}

	bool DelTimer(TimerNodeBase &node)
	{
		// C++ 14新特性，不在需要传一个key对象，传递一个key的等价值
		auto iter = timermap.find(node);
		if (iter != timermap.end())
		{
			timermap.erase(iter);
			return true;
		}
		return false;
	}

	bool CheckTimer()
	{
		auto iter = timermap.begin();

		if (iter != timermap.end() && iter->expire <= GetTick())
		{
			iter->func(*iter);
			timermap.erase(iter);
			return true;
		}
		return false;
	}

	time_t TimeToSleep()
	{
		auto iter = timermap.begin();
		if (iter == timermap.end())
			return -1;//没有定时任务，设置epoll一直阻塞。
		time_t diss = iter->expire - GetTick();

		return diss > 0 ? diss : 0;
	}
private:
	static int64_t GenID()
	{
		return gid++;
	}

	static int64_t gid;
	set<TimerNode, std::less<>> timermap;
};

int64_t Timer::gid = 0;

#define EPOLL_EV_LENFTH	1024
int main()
{
	int epfd = epoll_create(1);

	unique_ptr<Timer> timer = make_unique<Timer>();

	int i = 0;
	timer->AddTimer(1000, [&](const TimerNode &node) {
		cout << Timer::GetTick() << "node id:" <<  node.id << " revoked times:" << ++i << endl;
	});

	timer->AddTimer(1000, [&](const TimerNode &node) {
		cout << Timer::GetTick() << "node id:" <<  node.id << " revoked times:" << ++i << endl;
	});

	timer->AddTimer(3000, [&](const TimerNode &node) {
		cout << Timer::GetTick() << "node id:" <<  node.id << " revoked times:" << ++i << endl;
	});

	auto node = timer->AddTimer(2100, [&](const TimerNode &node) {
		cout << Timer::GetTick() << "node id:" <<  node.id << " revoked times:" << ++i << endl;
	});

	timer->DelTimer(node);

	cout << "now time:" << Timer::GetTick() << endl;

	epoll_event evs[EPOLL_EV_LENFTH] = { 0 };

	while (1)
	{
		int nready = epoll_wait(epfd, evs, EPOLL_EV_LENFTH, timer->TimeToSleep());
		for (int i = 0; i < nready; i++)
		{
			/*处理IO事件*/
		}

		// timer检测和处理
		while (timer->CheckTimer());
	}

	return 0;
}