#ifndef UWU_LEARNER_INTERFACE_EVENT_LISTENER_H
#define UWU_LEARNER_INTERFACE_EVENT_LISTENER_H

#include <string>

class InterfaceEventListener
{
    public:
        virtual ~InterfaceEventListener() = default;
        virtual void update(const std::string &event) = 0;
};

#endif 
