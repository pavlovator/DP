import json
from utils.xml import XML


class OutputConfig:
    def __init__(self, config_file):
        self.height = 1080
        self.width = 1920
        with open(config_file) as fp:
            self.outputs = json.load(fp)

    def is_consistent(self, xml, verbose_inconsistency=False):
        '''
        Test if xml file is consistently marked with markers described in config_file.
        :param xml: XML object
        :param verbose_inconsistency: if true, show detailed information of pointer causing inconsistency.
        :return: Boolean
        '''
        for dir in range(0, 360, 45):
            at = xml.get_pointer_attributes(dir)
            xml_pointers = []
            for idx, v in enumerate(at['visibility']):
                pointer = [at['x'][idx], at['y'][idx]]
                xml_pointers.append(pointer)
                if pointer in self.outputs[str(dir)] and v == "VISIBILITY_NOT_SET":
                    if verbose_inconsistency:
                        print("direction", dir, 'date', xml.date, pointer, at['label'][idx], at['distance'][idx], v)
                    return False
            for conf_pointer in self.outputs[str(dir)]:
                if conf_pointer not in xml_pointers:
                    return False
        return True

    def max_output_length(self):
        '''
        :return: number representing maximal number of pointers among all directions
        '''
        return max(list(self.output_lengths().values()))

    def output_lengths(self):
        '''
        :return: dictionary of pointer counts per direction
        '''
        lengths = dict()
        for key in self.outputs:
            lengths[key] = len(self.outputs[key])
        return lengths

    def get_output_positions(self, direction):
        '''
        :param direction:
        :return: sorted array of x,y positions
        '''
        xy_arr = []
        for x, y in self.outputs[str(direction)]:
            xy_arr.append((x, y))
        xy_arr.sort()
        return xy_arr

    def get_relative_positions(self, direction):
        '''
        :param direction:
        :return: array of x,y relative positions
        '''
        xy_arr = []
        for x, y in self.outputs[str(direction)]:
            xy_arr.append((float(x)/self.width, float(y)/self.height))
        return xy_arr
