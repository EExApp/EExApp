import xapp_sdk as ric
import time
import pdb
import json

# 3 slicing algorithms:
# static - [pos_low, pos_high]  provide hard isolation, best for guaranteed QoS
# NVS (non-volatile slicing) - 
#   NVS-rate-based: assigns rate-based bandwidth to slices, useful for bandwidth-sensitive applications like video streaming
#   NVS-capacity-based: assign percentage of total capacity, ideal for isolation and fairness, not absolute bandwidth
#   EDF (earliest deadline first) - for ultra-low-latency apps like URLLC or VoIP

# monitoring (indication) and control (configuration/command) for DL RAN slicing
# initialize and connect to E2 nodes
# subscribe to slice indications (slice conf. + UE associations)
# send control messages to:
# add/mod slices
# associate UEs to slices
# delete slices
# reset to no-slice state
# saves indication data to rt_slice_stats.json

# RNTI (radio network temporary identifier) - unique identifier assigned to a UE by the base station during a session

####################
####  SLICE INDICATION MSG TO JSON
####################



# indication to JSON - converts swig_slice_ind_msg_t into a JSON structure
# extract DL RAN slice stats: number, algorithm (STATIC, NVS, EDF), scheduling params
#         UE-to-slice associations: which RNTI is mapped to which slice
# write JSON to file rt_slice_stats.json

# slice_stats: dict, holds full JSON tree
# RAN -> dl, holds DL slice count, algorithm, and per-slice info
# UE -> holds UE RNTI and slice associations
# global assoc_rnti -> used later for UE slice control
# output -> written to rt_slice_stats.json

def slice_ind_to_dict_json(ind):        # ind: a swig_slice_ind_msg_t object received from the FlexRIC SDK, containing current slice conf. and UE associations            
    slice_stats = {                     # initialize empty dictionary
        "RAN" : {
            "dl" : {}
            # TODO: handle the ul slice stats, currently there is no ul slice stats in database(SLICE table)
            # "ul" : {}
        },
        "UE" : {}
    }

    # RAN - dl
    dl_dict = slice_stats["RAN"]["dl"]                                      # parse downlink slices
    if ind.slice_stats.dl.len_slices <= 0:                                  # if no DL slices exist. slice_stats.dl.len_slices -> number of DL slices currently active in RAN
        dl_dict["num_of_slices"] = ind.slice_stats.dl.len_slices            # store slice count = 0
        dl_dict["slice_sched_algo"] = "null"                                # set scheduler info to 'null'
        dl_dict["ue_sched_algo"] = ind.slice_stats.dl.sched_name[0]
    else:
        dl_dict["num_of_slices"] = ind.slice_stats.dl.len_slices
        dl_dict["slice_sched_algo"] = "null"
        dl_dict["slices"] = []
        slice_algo = ""
        for s in ind.slice_stats.dl.slices:
            if s.params.type == 1:                                          # TODO: convert from int to string, ex: type = 1 -> STATIC
                slice_algo = "STATIC"                                       # slicing algorithms
            elif s.params.type == 2:
                slice_algo = "NVS"
            elif s.params.type == 4:
                slice_algo = "EDF"
            else:
                slice_algo = "unknown"
            dl_dict.update({"slice_sched_algo" : slice_algo})               # update slice scheduling algorithm

            slices_dict = {                                                 # build an entry for this slice
                "index" : s.id,
                "label" : s.label[0],
                "ue_sched_algo" : s.sched[0],
            }
            if dl_dict["slice_sched_algo"] == "STATIC":                     # static params - slice is assigned a fixed range of PRB positions (pos_low, pos_high)
                slices_dict["slice_algo_params"] = {
                    "pos_low" : s.params.u.sta.pos_low,
                    "pos_high" : s.params.u.sta.pos_high
                }
            elif dl_dict["slice_sched_algo"] == "NVS":                      # NVS params (rate or capacity)
                if s.params.u.nvs.conf == 0:                                # TODO: convert from int to string, ex: conf = 0 -> RATE
                    slices_dict["slice_algo_params"] = {
                        "type" : "RATE",
                        "mbps_rsvd" : s.params.u.nvs.u.rate.u1.mbps_required,
                        "mbps_ref" : s.params.u.nvs.u.rate.u2.mbps_reference
                    }
                elif s.params.u.nvs.conf == 1:                              # TODO: convert from int to string, ex: conf = 1 -> CAPACITY
                    slices_dict["slice_algo_params"] = {
                        "type" : "CAPACITY",
                        "pct_rsvd" : s.params.u.nvs.u.capacity.u.pct_reserved
                    }
                else:
                    slices_dict["slice_algo_params"] = {"type" : "unknown"}
            elif dl_dict["slice_sched_algo"] == "EDF":                      # EDF params
                slices_dict["slice_algo_params"] = {
                    "deadline" : s.params.u.edf.deadline,
                    "guaranteed_prbs" : s.params.u.edf.guaranteed_prbs,
                    "max_replenish" : s.params.u.edf.max_replenish
                }
            else:
                print("unknown slice algorithm, cannot handle params")
            dl_dict["slices"].append(slices_dict)

    # RAN - ul
    # TODO: handle the ul slice stats, currently there is no ul slice stats in database(SLICE table)
    # ul_dict = slice_stats["RAN"]["ul"]
    # if ind.slice_stats.ul.len_slices <= 0:
    #     dl_dict["num_of_slices"] = ind.slice_stats.ul.len_slices
    #     dl_dict["slice_sched_algo"] = "null"
    #     dl_dict["ue_sched_algo"] = ind.slice_stats.ul.sched_name

    # UE-to-slice association
    global assoc_rnti
    ue_dict = slice_stats["UE"]
    if ind.ue_slice_stats.len_ue_slice <= 0:
        ue_dict["num_of_ues"] = ind.ue_slice_stats.len_ue_slice
    else:
        ue_dict["num_of_ues"] = ind.ue_slice_stats.len_ue_slice
        ue_dict["ues"] = []
        for u in ind.ue_slice_stats.ues:
            ues_dict = {}
            dl_id = "null"
            if u.dl_id >= 0 and dl_dict["num_of_slices"] > 0:           # get UE's RNTI and its associated DL slice ID
                dl_id = u.dl_id
            ues_dict = {                                                # store this in a dictionary
                "rnti" : hex(u.rnti),
                "assoc_dl_slice_id" : dl_id
                # TODO: handle the associated ul slice id, currently there is no ul slice id in database(UE_SLICE table)
                # "assoc_ul_slice_id" : ul_id
            }
            ue_dict["ues"].append(ues_dict)                             # append this UE entry
            assoc_rnti = u.rnti                                         # global - stores the RNTI globally for reuse elsewhere

    ind_dict = slice_stats
    ind_json = json.dumps(ind_dict)                                     # dump JSON to file

    with open("rt_slice_stats.json", "w") as outfile:
        outfile.write(ind_json)
    # print(ind_dict)

    return json                                                         # return value

####################
#### SLICE INDICATION CALLBACK
####################

class SLICECallback(ric.slice_cb):                                  # callback class for processing slice indication messages
                                                                    # On each indication, calls slice_ind_to_dict_json() to serialize the current slice state.
    # Define Python class 'constructor'                             
    def __init__(self):
        # Call C++ base class constructor
        ric.slice_cb.__init__(self)
    # Override C++ method: virtual void handle(swig_slice_ind_msg_t a) = 0;
    def handle(self, ind):
        # Print swig_slice_ind_msg_t
        #if (ind.slice_stats.dl.len_slices > 0):
        #     print('SLICE Indication tstamp = ' + str(ind.tstamp))
        #     print('SLICE STATE: len_slices = ' + str(ind.slice_stats.dl.len_slices))
        #     print('SLICE STATE: sched_name = ' + str(ind.slice_stats.dl.sched_name[0]))
        #if (ind.ue_slice_stats.len_ue_slice > 0):
        #    print('UE ASSOC SLICE STATE: len_ue_slice = ' + str(ind.ue_slice_stats.len_ue_slice))
        slice_ind_to_dict_json(ind)

####################
####  SLICE CONTROL FUNCS
####################
def create_slice(slice_params, slice_sched_algo):                   # build a single dl slice object fr_slice_t
    s = ric.fr_slice_t()                                            # create an instance of fr_slice_t, the C++ struct (exposed via SWIG), representing a single FlexRIC slice
    s.id = slice_params["id"]
    s.label = slice_params["label"]
    s.len_label = len(slice_params["label"])
    s.sched = slice_params["ue_sched_algo"]                         # UE scheduling algorithm within the slice ('RR', 'RF', 'MT')
    s.len_sched = len(slice_params["ue_sched_algo"])
    if slice_sched_algo == "STATIC":                                                    # slice is assigned a fixed range of PRB positions (pos_low, pos_high)
        s.params.type = ric.SLICE_ALG_SM_V0_STATIC
        s.params.u.sta.pos_low = slice_params["slice_algo_params"]["pos_low"]
        s.params.u.sta.pos_high = slice_params["slice_algo_params"]["pos_high"]
    elif slice_sched_algo == "NVS":                                                     
        s.params.type = ric.SLICE_ALG_SM_V0_NVS
        if slice_params["type"] == "SLICE_SM_NVS_V0_RATE":                                              # NVS - rate mode, used when slices are bandwidth-based (e.g., video streaming)
            s.params.u.nvs.conf = ric.SLICE_SM_NVS_V0_RATE                                              
            s.params.u.nvs.u.rate.u1.mbps_required = slice_params["slice_algo_params"]["mbps_rsvd"]     # reserve X Mbps, but aim for Y Mbps reference
            s.params.u.nvs.u.rate.u2.mbps_reference = slice_params["slice_algo_params"]["mbps_ref"]
            # print("ADD NVS DL SLCIE: id", s.id,
            # ", conf", s.params.u.nvs.conf,
            # ", mbps_rsrv", s.params.u.nvs.u.rate.u1.mbps_required,
            # ", mbps_ref", s.params.u.nvs.u.rate.u2.mbps_reference)
        elif slice_params["type"] == "SLICE_SM_NVS_V0_CAPACITY":                                        # percentage-based reservations instead of Mbps
            s.params.u.nvs.conf = ric.SLICE_SM_NVS_V0_CAPACITY
            s.params.u.nvs.u.capacity.u.pct_reserved = slice_params["slice_algo_params"]["pct_rsvd"]
            # print("ADD NVS DL SLCIE: id", s.id,
            # ", conf", s.params.u.nvs.conf,
            # ", pct_rsvd", s.params.u.nvs.u.capacity.u.pct_reserved)
        else:
            print("Unkown NVS conf")
    elif slice_sched_algo == "EDF":                                                                     # EDF (earliest deadline first), latency-sensitive traffic (e.g., VoIP, control signaling)
        s.params.type = ric.SLICE_ALG_SM_V0_EDF
        s.params.u.edf.deadline = slice_params["slice_algo_params"]["deadline"]                         # deadline: max allowed delay in ms
        s.params.u.edf.guaranteed_prbs = slice_params["slice_algo_params"]["guaranteed_prbs"]           # minimum PRBs to allocate
        s.params.u.edf.max_replenish = slice_params["slice_algo_params"]["max_replenish"]               # how fast the credit (resources) refills
    else:
        print("Unkown slice algo type")


    return s                                                                    # the fully populated 'fr_slice_t' object is returned and ready to be inserted into a slice control message

####################
####  SLICE CONTROL PARAMETER EXAMPLE - ADD SLICE
####################

# They are used to create control messages via:
# fill_slice_ctrl_msg(ctrl_type, ctrl_msg)
# which supports:
# "ADDMOD": Add or modify slice
# "DEL": Delete specific slice(s)
# "ASSOC_UE_SLICE": Map a UE to a slice


add_static_slices = {
    "num_slices" : 3,                                                           # total number of slices in this request
    "slice_sched_algo" : "STATIC",                                              
    "slices" : [                                                                # list of individual slice definitions
        {
            "id" : 0,                                                           
            "label" : "s1",                                                     
            "ue_sched_algo" : "PF",                                             # UEs in the slices are scheduling using proportional fair (PF)
            "slice_algo_params" : {"pos_low" : 0, "pos_high" : 2},              # params depend on scheduler type
        },
        {
            "id" : 2,
            "label" : "s2",
            "ue_sched_algo" : "PF",
            "slice_algo_params" : {"pos_low" : 3, "pos_high" : 10},
        },
        {
            "id" : 5,
            "label" : "s3",
            "ue_sched_algo" : "PF",
            "slice_algo_params" : {"pos_low" : 11, "pos_high" : 13},
        }
    ]
}

add_nvs_slices_rate = {
    "num_slices" : 2,
    "slice_sched_algo" : "NVS",
    "slices" : [
        {
            "id" : 0,
            "label" : "s1",
            "ue_sched_algo" : "PF",
            "type" : "SLICE_SM_NVS_V0_RATE",
            "slice_algo_params" : {"mbps_rsvd" : 60, "mbps_ref" : 120},
        },
        {
            "id" : 2,
            "label" : "s2",
            "ue_sched_algo" : "PF",
            "type" : "SLICE_SM_NVS_V0_RATE",
            "slice_algo_params" : {"mbps_rsvd" : 60, "mbps_ref" : 120},         # reserve 60 Mbps for this slice, try to keep it around 120 Mbps (reference bandwidth)
        }
    ]
}

add_nvs_slices_cap = {
    "num_slices" : 3,
    "slice_sched_algo" : "NVS",
    "slices" : [
        {
            "id" : 0,
            "label" : "s1",
            "ue_sched_algo" : "PF",
            "type" : "SLICE_SM_NVS_V0_CAPACITY",
            "slice_algo_params" : {"pct_rsvd" : 0.5},
        },
        {
            "id" : 2,
            "label" : "s2",
            "ue_sched_algo" : "PF",
            "type" : "SLICE_SM_NVS_V0_CAPACITY",
            "slice_algo_params" : {"pct_rsvd" : 0.3},
        },
        {
            "id" : 5,
            "label" : "s3",
            "ue_sched_algo" : "PF",
            "type" : "SLICE_SM_NVS_V0_CAPACITY",
            "slice_algo_params" : {"pct_rsvd" : 0.2},
        }
    ]
}

add_nvs_slices = {                                                          # mixed NVS mode of rate and capacity types
    "num_slices" : 3,                                                           
    "slice_sched_algo" : "NVS",
    "slices" : [
        {
            "id" : 0,
            "label" : "s1",
            "ue_sched_algo" : "PF",
            "type" : "SLICE_SM_NVS_V0_CAPACITY",
            "slice_algo_params" : {"pct_rsvd" : 0.5},
        },
        {
            "id" : 2,
            "label" : "s2",
            "ue_sched_algo" : "PF",
            "type" : "SLICE_SM_NVS_V0_RATE",
            "slice_algo_params" : {"mbps_rsvd" : 50, "mbps_ref" : 120},
        },
        {
            "id" : 5,
            "label" : "s3",
            "ue_sched_algo" : "PF",
            "type" : "SLICE_SM_NVS_V0_RATE",
            "slice_algo_params" : {"mbps_rsvd" : 5, "mbps_ref" : 120},
        }
    ]
}

add_edf_slices = {
    "num_slices" : 3,
    "slice_sched_algo" : "EDF",
    "slices" : [
        {
            "id" : 0,
            "label" : "s1",
            "ue_sched_algo" : "PF",                                                                     # Slices use different UE schedulers: PF, RR (round robin), MT (max throughput)
            "slice_algo_params" : {"deadline" : 10, "guaranteed_prbs" : 20, "max_replenish" : 0},       # ddl of 10 ms (for tight latency), guaranteed 20 PRBs for the slice, refill mechanism for credit is disabled
        },                                                                                              # "max_replenish" : 0 - Once the slice uses up its guaranteed_prbs, it won’t automatically get more — at least not until the system resets or reschedules.
        {
            "id" : 2,
            "label" : "s2",
            "ue_sched_algo" : "RR",
            "slice_algo_params" : {"deadline" : 20, "guaranteed_prbs" : 20, "max_replenish" : 0},
        },
        {
            "id" : 5,
            "label" : "s3",
            "ue_sched_algo" : "MT",
            "slice_algo_params" : {"deadline" : 40, "guaranteed_prbs" : 10, "max_replenish" : 0},
        }
    ]
}

reset_slices = {
    "num_slices" : 0
}

####################
####  SLICE CONTROL PARAMETER EXAMPLE - DELETE SLICE
####################
delete_slices = {
    "num_dl_slices" : 1,                                                                                # delete one dl slice
    "delete_dl_slice_id" : [5]                                                                          # ID 5
}

####################
####  SLICE CONTROL PARAMETER EXAMPLE - ASSOC UE SLICE
####################
assoc_ue_slice = {                                                               # assign a UE (identified by RNTI) to an existing DL slice (ID = 2)
    "num_ues" : 1,
    "ues" : [
        {
            "rnti" : 0,                                                    # TODO: replace this with a real RNTI from slice_ind_to_dict_json()
            "assoc_dl_slice_id" : 2
        }
    ]
}


# core of slice control message builder
# create and return a slice_ctrl_msg_t object, the message you send to FlexRIC to control slicing
# depending on the ctrl_type, it builds one of three message types: ADDMOD/DEL/ASSOC_UE_SLICE

def fill_slice_ctrl_msg(ctrl_type, ctrl_msg):       # ctrl_type: str: ADDMOD/DEL/ASSOC_UE_SLICE, ctrl_msg: a dictionary that holds the necessary conf.: add_static_slices/delete_slices
    msg = ric.slice_ctrl_msg_t()                    # main container for your control message
    if (ctrl_type == "ADDMOD"):                     # CASE 1: Add or Modify Slices ("ADDMOD")
        msg.type = ric.SLICE_CTRL_SM_V0_ADD                 # set control message type
        dl = ric.ul_dl_slice_conf_t()                       # set slice scheduling algorithm
        # TODO: UL SLICE CTRL ADD
        # ul = ric.ul_dl_slice_conf_t()

        # ue_sched_algo can be "RR"(round-robin), "PF"(proportional fair) or "MT"(maximum throughput) and it has to be set in any len_slices
        ue_sched_algo = "PF"                                # hardcoded, but this can vary per slice
        dl.sched_name = ue_sched_algo
        dl.len_sched_name = len(ue_sched_algo)

        dl.len_slices = ctrl_msg["num_slices"]              # create slices - an array
        slices = ric.slice_array(ctrl_msg["num_slices"])
        for i in range(0, ctrl_msg["num_slices"]):
            slices[i] = create_slice(ctrl_msg["slices"][i], ctrl_msg["slice_sched_algo"])

        dl.slices = slices                                  # attach the slice array to the message
        msg.u.add_mod_slice.dl = dl
        # TODO: UL SLICE CTRL ADD
        # msg.u.add_mod_slice.ul = ul;
    elif (ctrl_type == "DEL"):
        msg.type = ric.SLICE_CTRL_SM_V0_DEL

        msg.u.del_slice.len_dl = ctrl_msg["num_dl_slices"]          # prepare delete list
        del_dl_id = ric.del_dl_array(ctrl_msg["num_dl_slices"])
        for i in range(ctrl_msg["num_dl_slices"]):
            del_dl_id[i] = ctrl_msg["delete_dl_slice_id"][i]
        # print("DEL DL SLICE: id", del_dl_id)

        # TODO: UL SLCIE CTRL DEL
        msg.u.del_slice.dl = del_dl_id
    elif (ctrl_type == "ASSOC_UE_SLICE"):
        msg.type = ric.SLICE_CTRL_SM_V0_UE_SLICE_ASSOC

        msg.u.ue_slice.len_ue_slice = ctrl_msg["num_ues"]           # build UE-to-slice association array
        assoc = ric.ue_slice_assoc_array(ctrl_msg["num_ues"])
        for i in range(ctrl_msg["num_ues"]):
            a = ric.ue_slice_assoc_t()
            a.rnti = assoc_rnti                                    # TODO: assign the rnti after get the indication msg from slice_ind_to_dict_json()
            a.dl_id = ctrl_msg["ues"][i]["assoc_dl_slice_id"]       # assoc_ue_slice["ues"][0]["rnti"] should be dynamically filled from slice_ind_to_dict_json() output.
            # TODO: UL SLICE CTRL ASSOC
            # a.ul_id = 0
            assoc[i] = a
            # print("ASSOC DL SLICE: <rnti:", a.rnti, "(NEED TO FIX)>, id", a.dl_id)
        msg.u.ue_slice.ues = assoc                                  # set the association

    return msg                                                      # return the message


####################
####  GENERAL
####################

ric.init()                                                                                              # RIC initialization

conn = ric.conn_e2_nodes()                                                                              # connect to all available E2 nodes (gNBs/CUs/DUs), conn is a list of connected nodes, each with an id, plmn, etc
assert(len(conn) > 0)                                                                                   # ensure there's at least one node - otherwise the xApp exits

node_idx = 0
#for i in range(0, len(conn)):
#    if conn[i].id.plmn.mcc == 1:
#        node_idx = i

#print("Global E2 Node [" + str(node_idx) + "]: PLMN MCC = " + str(conn[node_idx].id.plmn.mcc))
#print("Global E2 Node [" + str(node_idx) + "]: PLMN MNC = " + str(conn[node_idx].id.plmn.mnc))

####################
#### SLICE INDICATION (slice monitoring)
####################

slice_cb = SLICECallback()                                                                              # subscribe to slice indications (slice monitoring)
hndlr = ric.report_slice_sm(conn[node_idx].id, ric.Interval_ms_5, slice_cb)                             # register a subscription to periodic slice stats, every 5 ms
time.sleep(5)                                                                                           # every few milliseconds, FlexRIC will call SLICECallback.handle(ind)
                                                                                                        # that ind message gets serialized into rt_slice_stats.json by slice_ind_to_dict_json()                   


####################
####  SLICE CTRL ADD
####################

msg = fill_slice_ctrl_msg("ADDMOD", add_static_slices)                                                  # add/modify Static slices, calls fill_slice_ctrl_msg() to construct a slice control message, use add_static_slices, which defines 3 STATIC slices (IDs: 0, 2, 5)
ric.control_slice_sm(conn[node_idx].id, msg)                                                            # send the control messag via control_slice_sm
time.sleep(20)                                                                                          # 20s yo run in the system

####################
####  SLICE CTRL ASSOC
####################

msg = fill_slice_ctrl_msg("ASSOC_UE_SLICE", assoc_ue_slice)                                          # associate UE to slice, construct and sends a message to associate UE (by RNTI) to slice ID 2
ric.control_slice_sm(conn[node_idx].id, msg)                                                        # use assoc_ue_slice dictionary (which must have the real RNTI)
time.sleep(20)

####################
####  SLICE CTRL DEL
####################

msg = fill_slice_ctrl_msg("DEL", delete_slices)                                                     # delete slice
ric.control_slice_sm(conn[node_idx].id, msg)
time.sleep(10)

####################
####  SLICE CTRL RESET
####################

msg = fill_slice_ctrl_msg("ADDMOD", reset_slices)                                                   # reset all slices
ric.control_slice_sm(conn[node_idx].id, msg)
time.sleep(5)

with open("rt_slice_stats.json", "w") as outfile:                                                   # clean up monitoring
    outfile.write(json.dumps({}))                                                                   # empty out JSON file used for indication logging

### End
ric.rm_report_slice_sm(hndlr)                                                                       # unsubscribe from slice indications using the earlier handle hndlr

# Avoid deadlock. ToDo revise architecture 
while ric.try_stop == 0:                                                                            # wait for RIC shutdown
    time.sleep(1)                                                                                   # keep the xApp alive until the RIC signals shutdown

print('Test finished' )

