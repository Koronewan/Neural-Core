#include "EventManager.h"

#include <stdexcept>

void EventManager::notify(const std::string &event)
{
    std::vector<InterfaceEventListener*> &listeners = this->listeners_[event];
    for (InterfaceEventListener* existingListener : listeners)
    {
        existingListener->update(event);
    }
}

void EventManager::subscribe(InterfaceEventListener *listener, const std::string &event)
{
    if (listener == nullptr)
        throw std::invalid_argument("listener is null");

    std::vector<InterfaceEventListener*> &listeners = this->listeners_[event];
    for (InterfaceEventListener* existingListener : listeners)
    {
        if (existingListener == listener)
            throw std::invalid_argument("listener is already subscribed");
    }

    listeners.push_back(listener);
}



void EventManager::unsubscribe(const InterfaceEventListener *listener, const std::string &event)
{
    if (listener == nullptr)
        throw std::invalid_argument("listener is null");

    std::vector<InterfaceEventListener*> &listeners = this->listeners_[event];
    for (int i = 0; i < listeners.size(); i++)
    {
        if (listeners[i] == listener)
        {
            listeners.erase(listeners.begin() + i);
            return;
        }
    }

    throw std::invalid_argument("listener is not subscribed");
}
