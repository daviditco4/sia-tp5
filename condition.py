# Creates a class with functions to check the condition to stop
# the algorithm and to replace the current value of the algorithm

class NonAccuracyCondition:
    def __init__(self, epsilon):
        self._epsilon = epsilon

    def check_stop(self, curr_error):
        return curr_error <= self._epsilon

    def check_replace(self, curr_error, new_error):
        return new_error < curr_error


# To be used in accuracy error, where we want a higher value
class AccuracyCondition:
    def __init__(self, epsilon):
        self._epsilon = epsilon

    def check_stop(self, curr_error):
        # El caso de 1 requiere el = (nunca va a ser mayor)
        return curr_error >= self._epsilon

    def check_replace(self, curr_error, new_error):
        return new_error > curr_error


# To be used in autoencoder, where we want a minimum error in pixels
class PixelError:
    def __init__(self, min_pixel):
        self.min_pixel = min_pixel

    def check_stop(self, curr_min_pixel):
        return curr_min_pixel <= self.min_pixel

    def check_replace(self, curr_min_pixel, new_min_pixel):
        return new_min_pixel < curr_min_pixel


class Structure:
    def __init__(self, config):
        self.threshold = config['structure_threshold']
        self.condition = from_str(config['structure_condition'], config)
        self.current = 0

    def check_stop(self, value):
        if self.condition.check_stop(value):
            self.current += 1
        else:
            self.current = 0
        return self.current >= self.threshold

    def check_replace(self, curr_value, new_value):
        return self.condition.check_replace(curr_value, new_value)


def from_str(string, config):
    match string.upper():
        case "NON_ACCURACY": return NonAccuracyCondition(epsilon=config['epsilon'])
        case "ACCURACY": return AccuracyCondition(epsilon=config['epsilon'])
        case "PIXEL_ERROR": return PixelError(min_pixel=config['min_pixel_error'])
        case "STRUCTURE": return Structure(config=config)
        case _: return NonAccuracyCondition(epsilon=config['epsilon'])