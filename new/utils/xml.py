import xml.etree.ElementTree as ET
import numpy as np


class XML:
    def __init__(self, file_name):
        '''
        :param date:  name of xml file eg. ../data/xmls/202002181420.xml
        '''
        self.dir_indices = {0: 0, 45: 1, 90: 2, 135: 3, 180: 4, 225: 5, 270: 6, 315: 7}
        self.file_name = file_name
        self.date = file_name.split(".")[-2][-12:]
        with open(file_name) as file:
            xml_string = file.read()
        self.root = ET.fromstring(xml_string)

    def _get_pointer_elements(self, direction):
        return list(list(self.root[self.dir_indices[direction]])[0])

    def _get_element_distance(self, element):
        return list(element)[0].items()[0][1]

    def _get_element_type(self, e):
        '''
        :param e: element
        :return: "DN" = day and night, "N" = night, "D" = day
        '''
        e_type = list(list(e)[1])
        if len(e_type) == 2:
            return "DN"
        if e_type[0].text == "LABEL_TYPE_NIGHT":
            return "N"
        return "D"


    def length(self, direction):
        '''
        :param direction:
        :return: number of pointers for given direction
        '''
        return len(self._get_pointer_elements(direction))

    def get_all_labels(self, direction):
        '''
        :param direction:
        :return: list of labels for all pointers, given the direction
        '''
        labels = []
        for e in self._get_pointer_elements(direction):
            labels.append(e.items()[0][1])
        return labels

    def list_pointer_visibilities(self, direction):
        '''
        :param direction:
        :return: list of visibilities for all pointers, given the direction
        '''
        visibilities = []
        for e in self._get_pointer_elements(direction):
            visibilities.append(e.items()[1][1])
        return visibilities

    def get_pointer_attributes(self, direction):
        '''
        Key type represents
        :param direction:
        :return: dictionary of attributes (label, visibility, x, y, etc ...) for all pointers, given the direction
        '''
        dictionary = {'label':[], 'visibility':[], 'arrowPosition':[],'panoramaX':[], 'panoramaY':[],
                      'x':[], 'y':[], 'latitude':[], 'longitude':[], 'distance':[], 'type':[]}
        for e in self._get_pointer_elements(direction):
            for k, v in e.items():
                dictionary[k].append(v)
            dictionary['type'].append(self._get_element_type(e))
            dictionary['distance'].append(self._get_element_distance(e))
        return dictionary

    def get_direction_output(self, direction, out_conf):
        '''
        Creates desired output of network. pointers are sorted by its (x,y) coordinates on image
        :param direction:
        :return: desired output of network
        '''
        arr = []
        attributes = self.get_pointer_attributes(direction)
        conf_xy = out_conf.get_output_positions(direction)
        for idx, vis in enumerate(attributes['visibility']):
            x, y = attributes['x'][idx], attributes['y'][idx]
            if (x, y) in conf_xy:
                if vis == "VISIBILITY_VISIBLE":
                    arr.append((x, y, 1))
                elif vis == "VISIBILITY_OBSCURED":
                    arr.append((x, y, 0))
        arr.sort()
        output = []
        for e in arr:
            output.append(e[2])
        return output


    def get_output_prevailing_visibility(self, config, direction, output):
        '''
        Return prevailing visibility of output (net) direction based on config file
        :param config:
        :param direction:
        :param output:
        :return:
        '''
        output = np.round(output)
        array_dict = dict()
        xy_arr = []
        config_pointers = config.get_output_positions(direction)
        atributes = self.get_pointer_attributes(direction)
        for i in range(len(atributes['x'])):
            x, y = atributes['x'][i], atributes['y'][i]
            if (x, y) in config_pointers:
                viz, dist = atributes['visibility'][i], atributes['distance'][i]
                array_dict[(x, y)] = (float(dist), viz)
                xy_arr.append((x, y))
        xy_arr.sort()
        visible_points = []
        for i, point in enumerate(xy_arr):
            dist, viz = array_dict[point]
            visible_points.append((dist, output[i]))
        visible_points.sort()
        for i, (dist, viz) in enumerate(visible_points):
            if viz == 0:
                if i == 0:
                    return 0
                return visible_points[i - 1][0]
        return visible_points[-1][0]


    def get_date(self):
        '''
        :return: date of xml
        '''
        return self.date

