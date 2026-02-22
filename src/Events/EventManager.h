#ifndef UWU_LEARNER_EVENT_MANAGER_H
#define UWU_LEARNER_EVENT_MANAGER_H

#include <map>
#include <string>
#include <vector>
#include <iostream>
#include "InterfaceEventListener.h"

class EventManager
{
    std::map<std::string, std::vector<InterfaceEventListener*>> listeners_;

public:
    void subscribe(InterfaceEventListener* listener, const std::string &event);
    void unsubscribe(const InterfaceEventListener* listener, const std::string &event);
    void notify(const std::string &event);
};

#endif 
