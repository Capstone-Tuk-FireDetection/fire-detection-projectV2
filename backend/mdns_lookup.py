# mdns_lookup.py
from zeroconf import Zeroconf, ServiceBrowser
import socket
import time

class MDNSListener:
    def __init__(self, target_name):
        self.target_name = target_name.lower()
        self.found_ip = None

    def remove_service(self, zeroconf, type, name):
        pass

    def add_service(self, zeroconf, type, name):
        info = zeroconf.get_service_info(type, name)
        if info and self.target_name in name.lower():
            self.found_ip = socket.inet_ntoa(info.addresses[0])

# ex: resolve_mdns("espcam1") → '192.168.x.x'
def resolve_mdns(hostname="espcam1"):
    zeroconf = Zeroconf()
    listener = MDNSListener(hostname)
    ServiceBrowser(zeroconf, "_http._tcp.local.", listener)

    timeout = time.time() + 3
    while not listener.found_ip and time.time() < timeout:
        time.sleep(0.1)

    zeroconf.close()
    return listener.found_ip
