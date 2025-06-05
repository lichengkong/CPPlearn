#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/epoll.h>
#include <unistd.h>
#include <fcntl.h>
#include <vector>

// 设置文件描述符为非阻塞模式
void setNonBlocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

// 事件处理器基类
class EventHandler {
public:
    virtual int getFd() const = 0;
    virtual void handleRead() = 0;
    virtual void handleWrite() = 0;
    virtual ~EventHandler() {}
};

// 处理新连接的事件处理器
class Acceptor : public EventHandler {
public:
    Acceptor(int port) {
        listenFd = socket(AF_INET, SOCK_STREAM, 0);
        if (listenFd == -1) {
            perror("socket");
            exit(EXIT_FAILURE);
        }

        sockaddr_in serverAddr{};
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_addr.s_addr = INADDR_ANY;
        serverAddr.sin_port = htons(port);

        if (bind(listenFd, reinterpret_cast<sockaddr*>(&serverAddr), sizeof(serverAddr)) == -1) {
            perror("bind");
            exit(EXIT_FAILURE);
        }

        if (listen(listenFd, SOMAXCONN) == -1) {
            perror("listen");
            exit(EXIT_FAILURE);
        }

        setNonBlocking(listenFd);
    }

    int getFd() const override {
        return listenFd;
    }

    void handleRead() override {
        sockaddr_in clientAddr{};
        socklen_t clientAddrLen = sizeof(clientAddr);
        int connFd = accept(listenFd, reinterpret_cast<sockaddr*>(&clientAddr), &clientAddrLen);
        if (connFd == -1) {
            perror("accept");
            return;
        }
        setNonBlocking(connFd);
        std::cout << "New connection from " << inet_ntoa(clientAddr.sin_addr) << ":" << ntohs(clientAddr.sin_port) << std::endl;
        // 这里可以将新的连接 fd 注册到 epoll 中处理读写事件
    }

    void handleWrite() override {}

    ~Acceptor() {
        close(listenFd);
    }

private:
    int listenFd;
};

// Reactor 类
class Reactor {
public:
    Reactor() {
        epollFd = epoll_create1(0);
        if (epollFd == -1) {
            perror("epoll_create1");
            exit(EXIT_FAILURE);
        }
    }

    void registerHandler(EventHandler* handler) {
        epoll_event ev{};
        ev.events = EPOLLIN;
        ev.data.ptr = handler;
        if (epoll_ctl(epollFd, EPOLL_CTL_ADD, handler->getFd(), &ev) == -1) {
            perror("epoll_ctl");
            exit(EXIT_FAILURE);
        }
    }

    void run() {
        std::vector<epoll_event> events(10);
        while (true) {
            int nfds = epoll_wait(epollFd, events.data(), events.size(), -1);
            if (nfds == -1) {
                perror("epoll_wait");
                continue;
            }
            for (int i = 0; i < nfds; ++i) {
                EventHandler* handler = static_cast<EventHandler*>(events[i].data.ptr);
                if (events[i].events & EPOLLIN) {
                    handler->handleRead();
                }
                if (events[i].events & EPOLLOUT) {
                    handler->handleWrite();
                }
            }
        }
    }

    ~Reactor() {
        close(epollFd);
    }

private:
    int epollFd;
};

int testReactor() {
    Acceptor acceptor(8888);
    Reactor reactor;
    reactor.registerHandler(&acceptor);
    reactor.run();
    return 0;
}