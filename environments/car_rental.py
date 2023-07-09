import math

from utils.environment import Environment


def get_poisson_prob(n, lam):
    return (math.pow(lam, n) / math.factorial(n)) * math.exp(-lam)


class CarRental(Environment):
    def __init__(self, car_states=21):
        super(CarRental, self).__init__(car_states * car_states, 11)
        self.req1_probs = [get_poisson_prob(n, 3) for n in range(car_states - 1)]
        self.req2_probs = [get_poisson_prob(n, 4) for n in range(car_states - 1)]
        self.ret1_probs = [get_poisson_prob(n, 3) for n in range(car_states - 1)]
        self.ret2_probs = [get_poisson_prob(n, 2) for n in range(car_states - 1)]

        self.req1_probs.append(1 - sum(self.req1_probs))
        self.req2_probs.append(1 - sum(self.req1_probs))
        self.ret1_probs.append(1 - sum(self.req1_probs))
        self.ret2_probs.append(1 - sum(self.req1_probs))
        self.car_states = car_states

    def get_transitions(self, state, action):
        cars_at_loc_1 = int(state / self.car_states)
        cars_at_loc_2 = int(state % self.car_states)

        number_of_cars_to_move = int(action) - 5
        cars_at_loc_1 -= number_of_cars_to_move
        cars_at_loc_2 += number_of_cars_to_move

        if cars_at_loc_1 < 0 or cars_at_loc_2 < 0:
            return []

        move_cost = 2 * abs(number_of_cars_to_move)

        req1_probs = self.req1_probs[:cars_at_loc_1]
        req1_probs.append(sum(self.req1_probs[cars_at_loc_1:]))
        ret1_probs = self.ret1_probs[:(self.car_states - 1) - cars_at_loc_1]
        ret1_probs.append(sum(self.ret1_probs[(self.car_states - 1) - cars_at_loc_1:]))

        possible_transitions_at_loc_1 = self.get_possible_transitions(cars_at_loc_1, req1_probs, ret1_probs)

        req2_probs = self.req2_probs[:cars_at_loc_2]
        req2_probs.append(sum(self.req2_probs[cars_at_loc_2:]))
        ret2_probs = self.ret2_probs[:(self.car_states - 1) - cars_at_loc_2]
        ret2_probs.append(sum(self.ret2_probs[(self.car_states - 1) - cars_at_loc_2:]))
        possible_transitions_at_loc_2 = self.get_possible_transitions(cars_at_loc_2, req2_probs, ret2_probs)

        transitions = []
        for possible_transition_at_loc_1 in possible_transitions_at_loc_1:
            for possible_transition_at_loc_2 in possible_transitions_at_loc_2:
                state = possible_transition_at_loc_1[0] * 21 + possible_transition_at_loc_2[0]
                reward = possible_transition_at_loc_1[1] + possible_transition_at_loc_2[1] - move_cost
                probability = possible_transition_at_loc_1[2] * possible_transition_at_loc_2[2]

                transitions.append((state, reward, probability))

        return transitions

    def get_possible_transitions(self, cars, req_probs, ret_probs):
        possible_transitions = []
        for req_cars in range(cars + 1):
            for ret_cars in range(self.car_states - cars):
                new_cars = cars + ret_cars - req_cars
                reward = req_cars * 10
                prob = req_probs[req_cars] * ret_probs[ret_cars]

                possible_transitions.append((new_cars, reward, prob))
        return possible_transitions

    def get_actions(self, state):
        cars_at_loc_1 = int(state / self.car_states)
        cars_at_loc_2 = int(state % self.car_states)

        return [i for i in range(11) if cars_at_loc_1 - (i - 5) >= 0 and cars_at_loc_2 + (i - 5) >= 0]
