import xapp_sdk as ric   # SWIG-generated Python binding to the underlying C++ FlexRIC xApp SDK
import time
import os
import pdb


#####################################################################################################################
#       Layer           Purpose                 Callback Class              Subscription API
#        MAC             UE scheduling stats     MACCallback                 report_mac_sm
#        RLC             RLC buffer/info         RLCCallback                 report_rlc_sm
#        PDCP            PDCP layer stats        PDCPCallback                report_pdcp_sm
#        GTP             Transport layer         GTPCallback                 report_gtp_sm
#####################################################################################################################


####################
#### MAC INDICATION CALLBACK
####################

#  MACCallback class is defined and derived from C++ class mac_cb
class MACCallback(ric.mac_cb):                                  # inherit from C++ callback interface and override handle() methods
    # Define Python class 'constructor'
    def __init__(self):
        # Call C++ base class constructor
        ric.mac_cb.__init__(self)
    # Override C++ method: virtual void handle(swig_mac_ind_msg_t a) = 0;
    def handle(self, ind):                                      # get an ind object, a SWIG-wrapped structure
        # Print swig_mac_ind_msg_t
        if len(ind.ue_stats) > 0:                               # In MAC layer, the stats are reported per UE: ind.ue_stats
            t_now = time.time_ns() / 1000.0
            t_mac = ind.tstamp / 1.0
            t_diff = t_now - t_mac
            print('MAC Indication tstamp = ' + str(t_mac) + ' latency = ' + str(t_diff) + ' μs')
            # print('MAC rnti = ' + str(ind.ue_stats[0].rnti))
# MAC callback
# called when a MAC layer indication is received
# calculate latency between the timestamp in the indication and the current time
# print latency in microseconds (us)



####################
#### RLC INDICATION CALLBACK
####################

class RLCCallback(ric.rlc_cb):
    # Define Python class 'constructor'
    def __init__(self):
        # Call C++ base class constructor
        ric.rlc_cb.__init__(self)
    # Override C++ method: virtual void handle(swig_rlc_ind_msg_t a) = 0;
    def handle(self, ind):
        # Print swig_rlc_ind_msg_t
        if len(ind.rb_stats) > 0:                               # statistics per RB - for RLC and PDCP layers
            t_now = time.time_ns() / 1000.0
            t_rlc = ind.tstamp / 1.0
            t_diff = t_now - t_rlc
            print('RLC Indication tstamp = ' + str(ind.tstamp) + ' latency = ' + str(t_diff) + ' μs')
            # print('RLC rnti = '+ str(ind.rb_stats[0].rnti))

####################
#### PDCP INDICATION CALLBACK
####################

class PDCPCallback(ric.pdcp_cb):
    # Define Python class 'constructor'
    def __init__(self):
        # Call C++ base class constructor
        ric.pdcp_cb.__init__(self)
   # Override C++ method: virtual void handle(swig_pdcp_ind_msg_t a) = 0;
    def handle(self, ind):
        # Print swig_pdcp_ind_msg_t
        if len(ind.rb_stats) > 0:
            t_now = time.time_ns() / 1000.0
            t_pdcp = ind.tstamp / 1.0
            t_diff = t_now - t_pdcp
            print('PDCP Indication tstamp = ' + str(ind.tstamp) + ' latency = ' + str(t_diff) + ' μs')

            # print('PDCP rnti = '+ str(ind.rb_stats[0].rnti))

####################
#### GTP INDICATION CALLBACK
####################

# Create a callback for GTP which derived it from C++ class gtp_cb
# handle GTP (GPRS Tunneling protocol) statistics from the transport layer
class GTPCallback(ric.gtp_cb):
    def __init__(self):
        # Inherit C++ gtp_cb class
        ric.gtp_cb.__init__(self)
    # Create an override C++ method 
    def handle(self, ind):
        if len(ind.gtp_stats) > 0:
            t_now = time.time_ns() / 1000.0
            t_gtp = ind.tstamp / 1.0
            t_diff = t_now - t_gtp
            print('GTP Indication tstamp = ' + str(ind.tstamp) + ' diff = ' + str(t_diff) + ' μs')


####################
####  GENERAL 
####################

ric.init()                                                  # initialize the xApp and connect it to the RIC runtime

conn = ric.conn_e2_nodes()                                  # a list of connected E2 nodes (gNodeBs or CU/DU) 
assert(len(conn) > 0)
for i in range(0, len(conn)):
    print("Global E2 Node [" + str(i) + "]: PLMN MCC = " + str(conn[i].id.plmn.mcc))        # prints PLMN MCC and MNC for each node
    print("Global E2 Node [" + str(i) + "]: PLMN MNC = " + str(conn[i].id.plmn.mnc))        # ric.Interval_ms_1： subscribes to each type of measurement at 1ms interval

####################
#### MAC INDICATION
####################

mac_hndlr = []                                                     # subscribes to each type of measurement at 1ms interval: ric.Interval_ms_1
for i in range(0, len(conn)):
    mac_cb = MACCallback()
    hndlr = ric.report_mac_sm(conn[i].id, ric.Interval_ms_1, mac_cb)        
    mac_hndlr.append(hndlr)     
    time.sleep(1)

####################
#### RLC INDICATION
####################

rlc_hndlr = []
for i in range(0, len(conn)):
    rlc_cb = RLCCallback()
    hndlr = ric.report_rlc_sm(conn[i].id, ric.Interval_ms_1, rlc_cb)
    rlc_hndlr.append(hndlr) 
    time.sleep(1)

####################
#### PDCP INDICATION
####################

pdcp_hndlr = []
for i in range(0, len(conn)):
    pdcp_cb = PDCPCallback()
    hndlr = ric.report_pdcp_sm(conn[i].id, ric.Interval_ms_1, pdcp_cb)
    pdcp_hndlr.append(hndlr) 
    time.sleep(1)

####################
#### GTP INDICATION
####################

gtp_hndlr = []
for i in range(0, len(conn)):
    gtp_cb = GTPCallback()
    hndlr = ric.report_gtp_sm(conn[i].id, ric.Interval_ms_1, gtp_cb)
    gtp_hndlr.append(hndlr)
    time.sleep(1)

time.sleep(10)                                                      # the xApp will print indications for 10 seconds

### End

for i in range(0, len(mac_hndlr)):
    ric.rm_report_mac_sm(mac_hndlr[i])                              # after 10s, unsubscribe all measurement reports

for i in range(0, len(rlc_hndlr)):
    ric.rm_report_rlc_sm(rlc_hndlr[i])

for i in range(0, len(pdcp_hndlr)):
    ric.rm_report_pdcp_sm(pdcp_hndlr[i])

for i in range(0, len(gtp_hndlr)):
    ric.rm_report_gtp_sm(gtp_hndlr[i])




# Avoid deadlock. ToDo revise architecture 
while ric.try_stop == 0:
    time.sleep(1)                                                   # loop to keep the app alive. prevents app from exiting if sth. is still running internally

print("Test finished")
