
"""
:class:`.BrownArcGIS` geocoder.
"""

import json
from time import time
from math import ceil
from geopy.compat import urlencode, urlopen, Request
from geopy.geocoders import ArcGIS
from geopy.geocoders.base import Geocoder
from geopy.exc import GeocoderServiceError, GeocoderAuthenticationFailure
from geopy.exc import ConfigurationError
from geopy.location import Location
from geopy.util import logger
import urllib

__all__ = ("BrownArcGIS", )

def read_file(path):
    with open(path, 'r') as path_stream:
        rstr = path_stream.read()
        path_stream.close()
    return rstr

class BrownArcGIS(ArcGIS):
    """
    Extend ArcGIS class from GeoPy 1.11.0
    """

    auth_api = 'http://quidditch.gis.brown.edu:6080/arcgis/tokens/generateToken'

    def __init__(self, **kwargs):

        super(BrownArcGIS, self).__init__(scheme='https', **kwargs)

        self.scheme = 'http' # https not supported

        self.api = (
            '%s://quidditch.gis.brown.edu:6080/arcgis/rest/services/brown_geocoding'
            '/Street_Addresses_US/GeocodeServer/findAddressCandidates' % self.scheme
        )
        self.batch_api = (
            '%s://quidditch.gis.brown.edu:6080/arcgis/rest/services/brown_geocoding'
            '/Street_Addresses_US/GeocodeServer/geocodeAddresses' % self.scheme
        )
        self.reverse_api = (
            '%s://quidditch.gis.brown.edu:6080/arcgis/rest/services/brown_geocoding'
            '/Street_Addresses_US/GeocodeServer/reverseGeocode' % self.scheme
        )

    def geocode(self, query='', street='', city='', state='', zip_cd='',
                n_matches=1, timeout=60):
        """
        Return a ranked list of locations for an address.
        :param string query: The single-line address you wish to geocode.
        :param string street: Optional, Street if not using single-line
        :param string city: Optional, City
        :param string state: Optional, State
        :param string zip_cd: Optional, Zip Code
        :param int n_matches: Return top n matches.
        :param int timeout: Time, in seconds, to wait for the geocoding service
            to respond before raising a :class:`geopy.exc.GeocoderTimedOut`
            exception. Set this only if you wish to override, on this call
            only, the value set during the geocoder's initialization.
        """

        params = {'Single Line Input': query,
                  'Street': street,
                  'City': city,
                  'State': state,
                  'ZIP': zip_cd,
                  'f': 'json',
                  'maxLocations': n_matches}

        if not (len(query) or len(street)):
            raise ConfigurationError(
                "Street or Full Address must be entered."
            )

        url = "?".join((self.api, urlencode(params)))

        #print(url)

        #url = url.encode('utf-8')

        logger.debug("%s.geocode: %s", self.__class__.__name__, url)
        response = self._call_geocoder(url, timeout=timeout)

        # Handle any errors; recursing in the case of an expired token
        if 'error' in response:
            if response['error']['code'] == self._TOKEN_EXPIRED:
                self.retry += 1
                self._refresh_authentication_token()
                return self.geocode(query, street, city, state, zip_cd, n_matches, timeout)
            raise GeocoderServiceError(str(response['error']))

        if not len(response['candidates']):
            return None

        #TODO 
        geocoded = []
        candidate_cnt = 1
        for candidate in response['candidates']:
            geocoded.append({
                'candidate':candidate_cnt,
                'attributes':{
                    'score':candidate['score'],
                    'match_addr':candidate['address'],
                    'location':{'x':candidate['location']['x'],
                                'y':candidate['location']['y']}}})
            candidate_cnt += 1

        return {'candidates':geocoded}


    def _refresh_authentication_token(self):
        self.retry = 0
        self.token_expiry = int(time()) + self.token_lifetime
        self.token = read_file('token.txt')




    def _refresh_authentication_token_broken(self):
        """
        POST to ArcGIS requesting a new token.
        """
        if self.retry == self._MAX_RETRIES:
            raise GeocoderAuthenticationFailure(
                'Too many retries for auth: %s' % self.retry
            )
        token_request_arguments = {
            'username': self.username,
            'password': self.password,
            'client': 'requestip',
            #'referer': 'requestip',
            'expiration': self.token_lifetime,
            'f': 'json'
        }
        self.token_expiry = int(time()) + self.token_lifetime
        data = urlencode(token_request_arguments)
        print(data)
        data = data.encode("utf-8")
        print(data)
        req = Request(url=self.auth_api, headers=self.headers)
        print(req)
        page = urlopen(req, data=data, timeout=self.timeout)
        print(page)
        page = page.read()
        print(page)
        response = json.loads(page.decode('unicode_escape'))
        if not 'token' in response:
            raise GeocoderAuthenticationFailure(
                'Missing token in auth request. '
                'Request URL: %s?%s. '
                'Response JSON: %s. ' %
                (self.auth_api, data, json.dumps(response))
            )
        self.retry = 0
        self.token = response['token']
