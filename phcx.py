import re
import xml.etree.cElementTree as ET
import numpy

################################################################################

def readDataBlock(xmlnode):
    """ Turn any 'DataBlock' XML node into a numpy array of floats
    """
    vmin = float(xmlnode.get('min'))
    vmax = float(xmlnode.get('max'))
    string = xmlnode.text
    string = re.sub("[\t\s\n]", "", string)
    data = numpy.asarray(
        bytearray.fromhex(string),
        dtype = float
        )
    return data * (vmax - vmin) / 255. + vmin


class Candidate(object):
    def __init__(self, fname):
        """ Build a new Candidate object from a PHCX file path.
        """
        xmlroot = ET.parse(fname).getroot()
        # Read CentreFreq and BandWidth
        coordNode = xmlroot.find('head')
        self.fc = float(coordNode.find('CentreFreq').text)
        self.bw = float(coordNode.find('BandWidth').text)
        # Get PDMP section
        for section in xmlroot.findall('Section'):
            if 'pdmp' in section.get('name').lower():
                opt_section = section
        # Best values as returned by PDMP
        opt_values = {
            node.tag : float(node.text)
            for node in opt_section.find('BestValues').getchildren()
            }
        self.bary_period = opt_values['BaryPeriod']
        self.dm = opt_values['Dm']
        self.snr = opt_values['Snr']
        self.width = opt_values['Width']

        ### Sub-Integrations
        subintsNode = opt_section.find('SubIntegrations')
        self.nsubs = int(subintsNode.get('nSub'))
        nsubs_subints = int(subintsNode.get('nSub'))
        self.nbins = int(subintsNode.get('nBins'))
        nbins_subints = int(subintsNode.get('nBins'))
        self.subints = readDataBlock(subintsNode).reshape(nsubs_subints, nbins_subints)
        ### Profile
        profileNode = opt_section.find('Profile')
        self.profile = readDataBlock(profileNode)
        self.nbins_profile = int(profileNode.get('nBins'))
################################################################################

if __name__ == '__main__':
    import os
    # Load example.phcx file (must be in the same directory as this python script)
    directory, fname = os.path.split(
        os.path.abspath(__file__)
        )
    cand = Candidate(
        os.path.join(directory, 'example.phcx')
        )
