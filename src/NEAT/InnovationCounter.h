//
// Created by korone on 1/13/25.
//

#ifndef NEURAL_CORE_INNOVATIONCOUNTER_H
#define NEURAL_CORE_INNOVATIONCOUNTER_H

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


#endif //NEURAL_CORE_INNOVATIONCOUNTER_H
