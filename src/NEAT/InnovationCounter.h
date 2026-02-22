//
// Created by korone on 1/13/25.
//

#ifndef UWU_LEARNER_INNOVATIONCOUNTER_H
#define UWU_LEARNER_INNOVATIONCOUNTER_H

class InnovationCounter
{
    int currentInnovation = 0;

public:
    int getNextInnovation()
    {
        return currentInnovation++;
    }

    ~InnovationCounter() = default;
};


#endif //UWU_LEARNER_INNOVATIONCOUNTER_H
