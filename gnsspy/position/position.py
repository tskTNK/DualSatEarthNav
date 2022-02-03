"""
Position computation and related functions
"""
# ===========================================================
# ========================= imports =========================
import time
from datetime import timedelta as _timedelta
import numpy as _np
import pandas as _pd
from operator import itemgetter as _itemgetter
from gnsspy.geodesy.coordinate import _distance_euclidean
from gnsspy.position.atmosphere import tropospheric_delay
from gnsspy.position.satellite import _reception_coord, _sagnac, _azel, _relativistic_clock
from gnsspy.funcs.constants import (_SYSTEM_RNX2, _SYSTEM_RNX3,
                                    _SYSTEM_NAME, _CLIGHT)

from gnsspy.geodesy.coordinate import cart2ell, ell2topo
import csv
import matplotlib.pyplot as plt
import pymap3d as pm

# ===========================================================
__all__ = ["spp","multipath"]

def mdpo(station, stationR, orbit, system="G", cut_off=7.0):

    start = time.time() # Time of start
    debug_out = []
    debug_output = []
    debug_output2 = []

    if len(system)>1:
        raise Warning("SPP does not support multiple satellite system | This feature will be implemented in the next version")
    observation_list = _observation_picker(station, system)
    observation_listR = _observation_picker(stationR, system)

    print(observation_list)
    print(observation_listR)

    gnss = gnssDataframe(station, orbit, system, cut_off)
    gnssR = gnssDataframe(stationR, orbit, system, cut_off)

    #-----------------------------------------------------------------------------
    if len(observation_list) >=2:
        carrierPhase1 = getattr(gnss,observation_list[0][2])
        carrierPhase2 = getattr(gnss,observation_list[1][2])
        pseudorange1  = getattr(gnss,observation_list[0][3])
        pseudorange2  = getattr(gnss,observation_list[1][3])
        frequency1 = observation_list[0][4]
        frequency2 = observation_list[1][4]
        carrierPhase1R = getattr(gnssR,observation_listR[0][2])
        carrierPhase2R = getattr(gnssR,observation_listR[1][2])
        pseudorange1R  = getattr(gnssR,observation_listR[0][3])
        pseudorange2R  = getattr(gnssR,observation_listR[1][3])
        frequency1R = observation_listR[0][4]
        frequency2R = observation_listR[1][4]
    else:
        raise Warning("Ionosphere-free combination is not available")
    # ----------------------------------------------------------------------------

    gnss["Ionosphere_Free"] = (frequency1**2*pseudorange1-frequency2**2*pseudorange2)/(frequency1**2-frequency2**2)
    gnss = gnss.dropna(subset = ['Ionosphere_Free'])
    gnss["Travel_time"] = gnss["Ionosphere_Free"] / _CLIGHT
    gnss["X_Reception"],gnss["Y_Reception"],gnss["Z_Reception"] = _reception_coord(gnss.X, gnss.Y, gnss.Z, gnss.Vx, gnss.Vy, gnss.Vz, gnss.Travel_time)

    gnssR["Ionosphere_Free"] = (frequency1R**2*pseudorange1R-frequency2R**2*pseudorange2R)/(frequency1R**2-frequency2R**2)
    gnssR = gnssR.dropna(subset = ['Ionosphere_Free'])
    gnssR["Travel_time"] = gnssR["Ionosphere_Free"] / _CLIGHT
    gnssR["X_Reception"],gnssR["Y_Reception"],gnssR["Z_Reception"] = _reception_coord(gnssR.X, gnssR.Y, gnssR.Z, gnssR.Vx, gnssR.Vy, gnssR.Vz, gnssR.Travel_time)

    # satList = []
    # satList.append(gnss.index.get_level_values("SV").unique().sort_values())
    # satList.append(gnssR.index.get_level_values("SV").unique().sort_values())
    # print (satList)
    satList = ['G01','G02','G03','G04','G05','G06','G07','G08','G09','G10','G11','G12','G13','G14','G15','G16','G17','G18','G19','G20','G21','G22','G23','G24','G25','G26','G27','G28','G29','G30','G31']
    # satList = ['G01','G02','G03','G04','G05','G06','G07','G08','G09','G10']
    # satList = ['G10','G23']

    for i in range(len(satList)):
        for j in range(i):
            satA = satList[i]
            satB = satList[j]

            print (i,j)
            debug_output3 = []
            debug_output4 = []

            epochList =gnss.index.get_level_values("Epoch").unique().sort_values()
            epoch_start = epochList[0]
            epochListR =gnssR.index.get_level_values("Epoch").unique().sort_values()
            epoch_startR = epochListR[0]

            #-----------------------------------------------------------------------------
            if not epoch_start == epoch_startR:
                raise Warning("Start epochs are different")
            # ----------------------------------------------------------------------------

            epoch_offset= _timedelta(seconds=300)
            epoch_interval = _timedelta(seconds=station.interval-0.000001)
            epoch_stop  = epochList[-1] + _timedelta(seconds=0.000001)

            approx_position = [station.approx_position[0], station.approx_position[1], station.approx_position[2]]
            receiver_clock = station.receiver_clock
            position_list = []

            approx_positionR = [stationR.approx_position[0], stationR.approx_position[1], stationR.approx_position[2]]
            receiver_clockR = stationR.receiver_clock
            position_listR = []

            approx_positionD = [station.approx_position[0]-stationR.approx_position[0], station.approx_position[1]-stationR.approx_position[1], station.approx_position[2]-stationR.approx_position[2]]
            receiver_clockD = station.receiver_clock - stationR.receiver_clock
            position_listD = []

            numOfEpochs = 0
            maxObservationEpochs = 3 # the number of observation epochs used for the MDPO estimation (use -1)
            newtonRaphsonUpdate = 0
            maxNewtonRaphsonUpdate = 1 # max number of NR iterations

            coeffMatrixD = _np.zeros([maxObservationEpochs,2])
            coeffMatrixD2 = _np.zeros([maxObservationEpochs,2])
            lMatrixD = _np.zeros([maxObservationEpochs,1])

            previous_matchSatelliteList_select = []
            numberOfSatellite = 0
            numOfEpochs = 0
            invalid_Sat_Combination = 0

            [lat0, lon0, h0] = pm.ecef2geodetic(approx_positionR[0], approx_positionR[1], approx_positionR[2])
            userENU = pm.ecef2enu(approx_position[0],approx_position[1],approx_position[2], lat0, lon0, h0)
            userENU = _np.array(userENU)
            # print('userENU ini')
            # print(userENU)

            while True:

                epoch_step = epoch_start + epoch_interval

                gnss_temp = gnss.xs((slice(epoch_start,epoch_step))).copy()
                gnss_tempR = gnssR.xs((slice(epoch_start,epoch_step))).copy()

                # pick up satellite that are seen from both stations
                satelliteList = gnss_temp.index.get_level_values("SV").unique().sort_values()
                satelliteListR = gnss_tempR.index.get_level_values("SV").unique().sort_values()
                matchSatelliteList = []
                for count in range(len(satelliteList)):
                    for count2 in range(len(satelliteListR)):
                        if  satelliteList[count] == satelliteListR[count2]:
                            matchSatelliteList.append(satelliteList[count])

                numberOfSatellite = len(matchSatelliteList)
                matchSatelliteList_select = []

                # print('matching test')
                matching = 0
                for count in range(len(matchSatelliteList)):
                    if matchSatelliteList[count] == satA:
                        matching += 1
                    if matchSatelliteList[count] == satB:
                        matching += 1
                if matching == 2:
                    # print('match')
                    pickupSatelliteList = [satA,satB]
                    gnss_temp2 = gnss_temp.loc[pickupSatelliteList]
                    gnss_temp2R = gnss_tempR.loc[pickupSatelliteList]
                    matchSatelliteList = pickupSatelliteList
                    matchSatelliteList_select = matchSatelliteList
                    invalid_Sat_Combination = 0
                else:
                    # print('no matching')
                    invalid_Sat_Combination = 1

                gnss_temp2 = gnss_temp.loc[matchSatelliteList]
                gnss_temp2R = gnss_tempR.loc[matchSatelliteList]
                gnss_temp = gnss_temp2
                gnss_tempR = gnss_temp2R

                # adding satellite clock bias for debug
                for count in range(len(gnss_temp)):
                    gnss_temp.Relativistic_clock[count] = 0.0000002*count
                for count in range(len(gnss_tempR)):
                    gnss_tempR.Relativistic_clock[count] = 0.0000002*count

                if previous_matchSatelliteList_select != matchSatelliteList_select or invalid_Sat_Combination == 1:
                    # print ('skip estimation')
                    numOfEpochs = 0
                else:
                    gnss_temp2 = gnss_temp.loc[matchSatelliteList_select]
                    gnss_temp2R = gnss_tempR.loc[matchSatelliteList_select]
                    gnss_temp = gnss_temp2
                    gnss_tempR = gnss_temp2R

                    distance = _distance_euclidean(approx_position[0],approx_position[1],approx_position[2], gnss_temp.X_Reception, gnss_temp.Y_Reception, gnss_temp.Z_Reception)
                    gnss_temp["Distance"] = distance + _sagnac(approx_position[0],approx_position[1],approx_position[2], gnss_temp.X_Reception, gnss_temp.Y_Reception, gnss_temp.Z_Reception)
                    gnss_temp["Azimuth"], gnss_temp["Elevation"], gnss_temp["Zenith"] = _azel(station.approx_position[0], station.approx_position[1], station.approx_position[2], gnss_temp.X, gnss_temp.Y, gnss_temp.Z, gnss_temp.Distance)
                    gnss_temp["Tropo"] = tropospheric_delay(station.approx_position[0],station.approx_position[1],station.approx_position[2], gnss_temp.Elevation, station.epoch)

                    distanceR = _distance_euclidean(approx_positionR[0],approx_positionR[1],approx_positionR[2], gnss_tempR.X_Reception, gnss_tempR.Y_Reception, gnss_tempR.Z_Reception)
                    gnss_tempR["Distance"] = distanceR + _sagnac(approx_positionR[0],approx_positionR[1],approx_positionR[2], gnss_tempR.X_Reception, gnss_tempR.Y_Reception, gnss_tempR.Z_Reception)
                    gnss_tempR["Azimuth"], gnss_tempR["Elevation"], gnss_tempR["Zenith"] = _azel(stationR.approx_position[0], stationR.approx_position[1], stationR.approx_position[2], gnss_tempR.X, gnss_tempR.Y, gnss_tempR.Z, gnss_tempR.Distance)
                    gnss_tempR["Tropo"] = tropospheric_delay(stationR.approx_position[0],stationR.approx_position[1],stationR.approx_position[2], gnss_tempR.Elevation, stationR.epoch)

                    sat1 = 0
                    sat2 = 1

                    enuSatX, enuSatY, enuSatZ  = pm.ecef2enu(gnss_temp.X_Reception, gnss_temp.Y_Reception, gnss_temp.Z_Reception, lat0, lon0, h0)
                    enuSatX = _np.array(enuSatX)
                    enuSatY = _np.array(enuSatY)
                    enuSatZ = _np.array(enuSatZ)

                    vector_sat1 = [enuSatX[sat1], enuSatY[sat1], enuSatZ[sat1]]
                    vector_sat2 = [enuSatX[sat2], enuSatY[sat2], enuSatZ[sat2]]
                    unitvector_sat1 = vector_sat1 / _np.linalg.norm(vector_sat1)
                    unitvector_sat2 = vector_sat2 / _np.linalg.norm(vector_sat2)
                    dot_product = _np.dot(unitvector_sat1,unitvector_sat2)
                    angle = _np.degrees(_np.arccos(dot_product))
                    el = (gnss_temp.Elevation[sat1]+gnss_temp.Elevation[sat2])/2

                    coeffMatrixD[numOfEpochs,0] = (userENU[0] - enuSatX[sat1]) / gnss_temp.Distance[sat1] - ((userENU[0] - enuSatX[sat2]) / gnss_temp.Distance[sat2])
                    coeffMatrixD[numOfEpochs,1] = (userENU[1] - enuSatY[sat1]) / gnss_temp.Distance[sat1] - ((userENU[1] - enuSatY[sat2]) / gnss_temp.Distance[sat2])

                    coeffMatrixD2[numOfEpochs,0] = (approx_position[0] - gnss_temp.X_Reception[sat1]) / gnss_temp.Distance[sat1] - ((approx_position[0] - gnss_temp.X_Reception[sat2]) / gnss_temp.Distance[sat2])
                    coeffMatrixD2[numOfEpochs,1] = (approx_position[1] - gnss_temp.Y_Reception[sat1]) / gnss_temp.Distance[sat1] - ((approx_position[1] - gnss_temp.Y_Reception[sat2]) / gnss_temp.Distance[sat2])

                    lMatrix = gnss_temp.Ionosphere_Free[sat1] - gnss_temp.Distance[sat1] + _CLIGHT * (gnss_temp.DeltaTSV[sat1] + gnss_temp.Relativistic_clock[sat1] - receiver_clock) - gnss_temp.Tropo[sat1] - (gnss_temp.Ionosphere_Free[sat2] - gnss_temp.Distance[sat2] + _CLIGHT * (gnss_temp.DeltaTSV[sat2] + gnss_temp.Relativistic_clock[sat2] - receiver_clock) - gnss_temp.Tropo[sat2])
                    lMatrixR = gnss_tempR.Ionosphere_Free[sat1] - gnss_tempR.Distance[sat1] + _CLIGHT * (gnss_tempR.DeltaTSV[sat1] + gnss_tempR.Relativistic_clock[sat1] - receiver_clockR) - gnss_tempR.Tropo[sat1] - (gnss_tempR.Ionosphere_Free[sat2] - gnss_tempR.Distance[sat2] + _CLIGHT * (gnss_tempR.DeltaTSV[sat2] + gnss_tempR.Relativistic_clock[sat2] - receiver_clockR) - gnss_tempR.Tropo[sat2])
                    lMatrixD[numOfEpochs,0] = lMatrix - lMatrixR

                    numOfEpochs = numOfEpochs + 1

                    # if newtonRaphsonUpdate == 0:
                    #     print ('epoch_start')
                    #     print (epoch_start)
                    #     print (matchSatelliteList)

                    if newtonRaphsonUpdate == maxNewtonRaphsonUpdate:
                        debug_output3.append([epoch_start, matchSatelliteList_select[sat1], matchSatelliteList_select[sat2], gnss_temp.Azimuth[sat1], gnss_temp.Elevation[sat1], gnss_tempR.Azimuth[sat1], gnss_tempR.Elevation[sat1], gnss_temp.Azimuth[sat2], gnss_temp.Elevation[sat2], gnss_tempR.Azimuth[sat2], gnss_tempR.Elevation[sat2]])

                    if numOfEpochs > maxObservationEpochs-1:

                        # DOP Calculation
                        GTG = _np.dot(coeffMatrixD.T, coeffMatrixD)
                        DOP = _np.linalg.inv(GTG)
                        GDOP = (DOP[0][0]+DOP[1][1])**0.5
                        XDOP = (DOP[0][0])**0.5
                        YDOP = (DOP[1][1])**0.5
                        HDOP = (DOP[0][0]+DOP[1][1])**0.5

                        GTG2 = _np.dot(coeffMatrixD2.T, coeffMatrixD2)
                        DOP2 = _np.linalg.inv(GTG2)
                        GDOP2 = (DOP2[0][0]+DOP2[1][1])**0.5
                        XDOP2 = (DOP2[0][0])**0.5
                        YDOP2 = (DOP2[1][1])**0.5
                        HDOP2 = (DOP2[0][0]+DOP2[1][1])**0.5

                        # if newtonRaphsonUpdate == 1:
                        #     print('newtonRaphsonUpdate == 1')
                        #     print(XDOP, YDOP, HDOP)
                        #     print(XDOP2, YDOP2, HDOP2)

                        if GDOP > 300 and newtonRaphsonUpdate == maxNewtonRaphsonUpdate:
                            debug_output4.append(debug_output3[-3])
                            debug_output4.append(debug_output3[-2])
                            debug_output4.append(debug_output3[-1])

                        if GDOP > 300 and newtonRaphsonUpdate == maxNewtonRaphsonUpdate:
                            # print([epoch_start, matchSatelliteList_select[0], matchSatelliteList_select[1], GDOP, XDOP, YDOP, float(posD[0]), float(posD[1]), float(posD[2]), gnss_temp.Azimuth[0], gnss_temp.Elevation[0], gnss_tempR.Azimuth[0], gnss_tempR.Elevation[0], gnss_temp.Azimuth[1], gnss_temp.Elevation[1], gnss_tempR.Azimuth[1], gnss_tempR.Elevation[1], coeffMatrixD[0,0], coeffMatrixD[0,1], coeffMatrixD[1,0], coeffMatrixD[1,1], coeffMatrixD[2,0], coeffMatrixD[2,1]])
                            debug_output2.append([epoch_start, matchSatelliteList_select[0], matchSatelliteList_select[1], GDOP, XDOP, YDOP, float(posD[0]), float(posD[1]), float(posD[2]), angle, el, gnss_temp.Azimuth[0], gnss_temp.Elevation[0], gnss_tempR.Azimuth[0], gnss_tempR.Elevation[0], gnss_temp.Azimuth[1], gnss_temp.Elevation[1], gnss_tempR.Azimuth[1], gnss_tempR.Elevation[1], coeffMatrixD[0,0], coeffMatrixD[0,1], coeffMatrixD[1,0], coeffMatrixD[1,1], coeffMatrixD[2,0], coeffMatrixD[2,1]])
                            # debug_output4.append([epoch_start, matchSatelliteList_select[0], matchSatelliteList_select[1], GDOP, XDOP, YDOP, float(posD[0]), float(posD[1]), float(posD[2]), gnss_temp.Azimuth[0], gnss_temp.Elevation[0], gnss_tempR.Azimuth[0], gnss_tempR.Elevation[0], gnss_temp.Azimuth[1], gnss_temp.Elevation[1], gnss_tempR.Azimuth[1], gnss_tempR.Elevation[1], coeffMatrixD[0,0], coeffMatrixD[0,1], coeffMatrixD[1,0], coeffMatrixD[1,1], coeffMatrixD[2,0], coeffMatrixD[2,1]])

                        if GDOP < 1000000000:   # dop cut

                            newtonRaphsonUpdate = newtonRaphsonUpdate + 1

                            if newtonRaphsonUpdate > maxNewtonRaphsonUpdate:
                                newtonRaphsonUpdate = 0
                                # print([epoch_start, matchSatelliteList_select[0], matchSatelliteList_select[1], GDOP, XDOP, YDOP, float(posD[0]), float(posD[1]), float(posD[2])])
                                debug_output.append([epoch_start, matchSatelliteList_select[0], matchSatelliteList_select[1], GDOP, XDOP, YDOP, float(posD[0]), float(posD[1]), float(posD[2]), angle, el])
                                position_listD.append(posD)
                                # print(float(posD[0]), float(posD[1]), float(posD[2]), XDOP, YDOP, HDOP, angle, el)
                                approx_position = [station.approx_position[0], station.approx_position[1], station.approx_position[2]]
                                userENU = pm.ecef2enu(approx_position[0],approx_position[1],approx_position[2], lat0, lon0, h0)
                                userENU = _np.array(userENU)

                            else:
                                try:
                                    linearEquationSolutionD = _np.linalg.lstsq(coeffMatrixD,lMatrixD,rcond=None)
                                    xMatrixD = linearEquationSolutionD[0]
                                    # approx_position[0], approx_position[1], approx_position[2] = approx_position[0] + xMatrixD[0], approx_position[1] + xMatrixD[1], approx_position[2]
                                    # posD = [approx_position[0]-approx_positionR[0], approx_position[1]-approx_positionR[1], approx_position[2]-approx_positionR[2]]
                                    posD = [userENU[0] + xMatrixD[0], userENU[1] + xMatrixD[1], userENU[2]]
                                    userENU[0], userENU[1], userENU[2] = posD[0], posD[1], posD[2]
                                    approx_position[0],approx_position[1],approx_position[2] = pm.enu2ecef(userENU[0], userENU[1], userENU[2], lat0, lon0, h0)
                                    approx_position = _np.array(approx_position)
                                    epoch_start -= 3*epoch_offset
                                    epoch_step  -= 3*epoch_offset
                                except:
                                    # print("Cannot solve normal equations for epoch", epoch_start,"| Skipping...")
                                    epoch_start -= 3*epoch_offset
                                    epoch_step  -= 3*epoch_offset

                        numOfEpochs = 0
                        coeffMatrixD = _np.zeros([maxObservationEpochs,2])
                        lMatrixD = _np.zeros([maxObservationEpochs,1])

                previous_matchSatelliteList_select = matchSatelliteList_select

                epoch_start += epoch_offset
                epoch_step  += epoch_offset
                if (epoch_step - epoch_stop) > _timedelta(seconds=station.interval):
                    break

            # skyplot
            az11 = []
            el11 = []
            az12 = []
            el12 = []
            az21 = []
            el21 = []
            az22 = []
            el22 = []
            azh11 = []
            elh11 = []
            azh12 = []
            elh12 = []

            if len(debug_output3) > 0:

                for id in range(len(debug_output3)):
                    az = debug_output3[id][3]
                    el = debug_output3[id][4]
                    azr = _np.deg2rad(az)
                    elr = _np.deg2rad(el)
                    if az < 0:
                        az = az + 360
                    if el < 0:
                        el = 0
                    az11.append(azr)
                    el11.append(el)

                for id in range(len(debug_output3)):
                    # print(id)
                    az = debug_output3[id][7]
                    el = debug_output3[id][8]
                    azr = _np.deg2rad(az)
                    elr = _np.deg2rad(el)
                    if az < 0:
                        az = az + 360
                    if el < 0:
                        el = 0
                    az12.append(azr)
                    el12.append(el)

                if len(debug_output4) > 0:

                    for id in range(len(debug_output4)):
                        az = debug_output4[id][3]
                        el = debug_output4[id][4]
                        azr = _np.deg2rad(az)
                        elr = _np.deg2rad(el)
                        if az < 0:
                            az = az + 360
                        if el < 0:
                            el = 0
                        azh11.append(azr)
                        elh11.append(el)

                    for id in range(len(debug_output4)):
                        az = debug_output4[id][7]
                        el = debug_output4[id][8]
                        azr = _np.deg2rad(az)
                        elr = _np.deg2rad(el)
                        if az < 0:
                            az = az + 360
                        if el < 0:
                            el = 0
                        azh12.append(azr)
                        elh12.append(el)

                if len(debug_output) > 14:

                    for id in range(len(debug_output)):
                        debug_out.append(debug_output[id])
                    debug_output.clear()

                    fig = plt.figure()
                    ax = fig.add_subplot(projection='polar')
                    ax.set_theta_zero_location('N')
                    ax.set_yticks(_np.arange(0, 91, 15))
                    ax.set_rlim(bottom=90, top=0)
                    ax.scatter(az11, el11, color = 'b')
                    ax.scatter(az12, el12, color = 'r')
                    ax.scatter(azh11, elh11, color = 'g')
                    ax.scatter(azh12, elh12, color = 'g')
                    # plt.show()
                    filename = "{}{}sky.png".format(satA, satB)
                    plt.savefig(filename)
                    plt.clf()

                    fig = plt.figure()
                    ax = fig.add_subplot(projection='polar')
                    ax.set_theta_zero_location('N')
                    ax.set_yticks(_np.arange(0, 91, 15))
                    ax.set_rlim(bottom=90, top=0)
                    ax.scatter(az11, el11, color = 'b')
                    ax.scatter(az12, el12, color = 'r')
                    # plt.show()
                    filename = "{}{}sky_raw.png".format(satA, satB)
                    plt.savefig(filename)
                    plt.clf()
                else:
                    debug_output.clear()

            # debug out to csv (for every combination of satellites)
            # fields = ['epoch_start', 'SV1', 'SV2', 'sta1 sat1 AZ', 'sta1 sat1 EL', 'sta2 sat1 AZ', 'sta2 sat1 EL','sta1 sat2 AZ', 'sta1 sat2 EL', 'sta2 sat2 AZ', 'sta2 sat2 EL']
            # filename = "data_debug2.csv"
            # with open(filename, 'w', newline='') as csvfile:
            #     csvwriter = csv.writer(csvfile)
            #     csvwriter.writerow(fields)
            #     csvwriter.writerows(debug_output3)

    # debug out to csv
    fields = ['epoch_start', 'SV1', 'SV2', 'GDOP', 'XDOP', 'YDOP', 'pos est X', 'pos est Y', 'pos est Z', 'angle', 'el']
    filename = "data.csv"
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(debug_out)

    fields = ['epoch_start', 'SV1', 'SV2', 'GDOP', 'XDOP', 'YDOP', 'pos est X', 'pos est Y', 'pos est Z', 'angle', 'el', 'sta1 sat1 AZ', 'sta1 sat1 EL', 'sta2 sat1 AZ', 'sta2 sat1 EL','sta1 sat2 AZ', 'sta1 sat2 EL', 'sta2 sat2 AZ', 'sta2 sat2 EL', 'G[0][0]', 'G[0][1]', 'G[1][0]', 'G[1][1]', 'G[2][0]', 'G[2][1]']
    filename = "data_debug.csv"
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(debug_output2)

    x_coordinateD = _np.mean([posD[0] for posD in position_listD])
    y_coordinateD = _np.mean([posD[1] for posD in position_listD])
    z_coordinateD = _np.mean([posD[2] for posD in position_listD])
    x_coordinateD = _np.mean([posD[0] for posD in position_listD])
    y_coordinateD = _np.mean([posD[1] for posD in position_listD])
    z_coordinateD = _np.mean([posD[2] for posD in position_listD])

    finish = time.time()     # Time of finish
    print("Pseudorange calculation is done in", "{0:.2f}".format(finish-start), "seconds.")
    print ("\n")

    return (x_coordinateD, y_coordinateD, z_coordinateD)

def dpp(station, stationR, orbit, system="G", cut_off=7.0):
    start = time.time() # Time of start
    if len(system)>1:
        raise Warning("SPP does not support multiple satellite system | This feature will be implemented in the next version")
    observation_list = _observation_picker(station, system)
    observation_listR = _observation_picker(stationR, system)

    gnss = gnssDataframe(station, orbit, system, cut_off)
    gnssR = gnssDataframe(stationR, orbit, system, cut_off)

    #-----------------------------------------------------------------------------
    if len(observation_list) >=2:
        carrierPhase1 = getattr(gnss,observation_list[0][2])
        carrierPhase2 = getattr(gnss,observation_list[1][2])
        pseudorange1  = getattr(gnss,observation_list[0][3])
        pseudorange2  = getattr(gnss,observation_list[1][3])
        frequency1 = observation_list[0][4]
        frequency2 = observation_list[1][4]
        carrierPhase1R = getattr(gnssR,observation_listR[0][2])
        carrierPhase2R = getattr(gnssR,observation_listR[1][2])
        pseudorange1R  = getattr(gnssR,observation_listR[0][3])
        pseudorange2R  = getattr(gnssR,observation_listR[1][3])
        frequency1R = observation_listR[0][4]
        frequency2R = observation_listR[1][4]
    else:
        raise Warning("Ionosphere-free combination is not available")
    # ----------------------------------------------------------------------------

    gnss["Ionosphere_Free"] = (frequency1**2*pseudorange1-frequency2**2*pseudorange2)/(frequency1**2-frequency2**2)
    gnss = gnss.dropna(subset = ['Ionosphere_Free'])
    gnss["Travel_time"] = gnss["Ionosphere_Free"] / _CLIGHT
    gnss["X_Reception"],gnss["Y_Reception"],gnss["Z_Reception"] = _reception_coord(gnss.X, gnss.Y, gnss.Z, gnss.Vx, gnss.Vy, gnss.Vz, gnss.Travel_time)

    gnssR["Ionosphere_Free"] = (frequency1R**2*pseudorange1R-frequency2R**2*pseudorange2R)/(frequency1R**2-frequency2R**2)
    gnssR = gnssR.dropna(subset = ['Ionosphere_Free'])
    gnssR["Travel_time"] = gnssR["Ionosphere_Free"] / _CLIGHT
    gnssR["X_Reception"],gnssR["Y_Reception"],gnssR["Z_Reception"] = _reception_coord(gnssR.X, gnssR.Y, gnssR.Z, gnssR.Vx, gnssR.Vy, gnssR.Vz, gnssR.Travel_time)

    epochList =gnss.index.get_level_values("Epoch").unique().sort_values()
    epoch_start = epochList[0]
    epochListR =gnssR.index.get_level_values("Epoch").unique().sort_values()
    epoch_startR = epochListR[0]

    #-----------------------------------------------------------------------------
    if epoch_start == epoch_startR:
        print("epochs are same")
    else:
        raise Warning("Start epochs are different")
    # ----------------------------------------------------------------------------

    epoch_offset= _timedelta(seconds=300)
    epoch_interval = _timedelta(seconds=station.interval-0.000001)
    epoch_stop  = epochList[-1] + _timedelta(seconds=0.000001)

    approx_position = [station.approx_position[0], station.approx_position[1], station.approx_position[2]]
    receiver_clock = station.receiver_clock
    position_list = []

    approx_positionR = [stationR.approx_position[0], stationR.approx_position[1], stationR.approx_position[2]]
    receiver_clockR = stationR.receiver_clock
    position_listR = []

    approx_positionD = [station.approx_position[0]-stationR.approx_position[0], station.approx_position[1]-stationR.approx_position[1], station.approx_position[2]-stationR.approx_position[2]]
    receiver_clockD = station.receiver_clock - stationR.receiver_clock
    position_listD = []

    [lat0, lon0, h0] = pm.ecef2geodetic(approx_positionR[0], approx_positionR[1], approx_positionR[2])
    userENU = pm.ecef2enu(approx_position[0],approx_position[1],approx_position[2], lat0, lon0, h0)
    userENU = _np.array(userENU)

    while True:

        epoch_step = epoch_start + epoch_interval

        gnss_temp = gnss.xs((slice(epoch_start,epoch_step))).copy()
        gnss_tempR = gnssR.xs((slice(epoch_start,epoch_step))).copy()

        # pick up satellite that are seen from both stations
        satelliteList = gnss_temp.index.get_level_values("SV").unique().sort_values()
        satelliteListR = gnss_tempR.index.get_level_values("SV").unique().sort_values()
        matchSatelliteList = []
        for count in range(len(satelliteList)):
            for count2 in range(len(satelliteListR)):
                if  satelliteList[count] == satelliteListR[count2]:
                    matchSatelliteList.append(satelliteList[count])

        gnss_temp2 = gnss_temp.loc[matchSatelliteList]
        gnss_temp2R = gnss_tempR.loc[matchSatelliteList]
        gnss_temp = gnss_temp2
        gnss_tempR = gnss_temp2R

        # wrong satellite clock for debug
        for count in range(len(gnss_temp)):
            gnss_temp.Relativistic_clock[count] = 0.0000002*count
        for count in range(len(gnss_tempR)):
            gnss_tempR.Relativistic_clock[count] = 0.0000002*count

        for iter in range(6):

            distance = _distance_euclidean(approx_position[0],approx_position[1],approx_position[2], gnss_temp.X_Reception, gnss_temp.Y_Reception, gnss_temp.Z_Reception)
            gnss_temp["Distance"] = distance + _sagnac(approx_position[0],approx_position[1],approx_position[2], gnss_temp.X_Reception, gnss_temp.Y_Reception, gnss_temp.Z_Reception)
            gnss_temp["Azimuth"], gnss_temp["Elevation"], gnss_temp["Zenith"] = _azel(station.approx_position[0], station.approx_position[1], station.approx_position[2], gnss_temp.X, gnss_temp.Y, gnss_temp.Z, gnss_temp.Distance)
            gnss_temp["Tropo"] = tropospheric_delay(station.approx_position[0],station.approx_position[1],station.approx_position[2], gnss_temp.Elevation, station.epoch)

            distanceR = _distance_euclidean(approx_positionR[0],approx_positionR[1],approx_positionR[2], gnss_tempR.X_Reception, gnss_tempR.Y_Reception, gnss_tempR.Z_Reception)
            gnss_tempR["Distance"] = distanceR + _sagnac(approx_positionR[0],approx_positionR[1],approx_positionR[2], gnss_tempR.X_Reception, gnss_tempR.Y_Reception, gnss_tempR.Z_Reception)
            gnss_tempR["Azimuth"], gnss_tempR["Elevation"], gnss_tempR["Zenith"] = _azel(stationR.approx_position[0], stationR.approx_position[1], stationR.approx_position[2], gnss_tempR.X, gnss_tempR.Y, gnss_tempR.Z, gnss_tempR.Distance)
            gnss_tempR["Tropo"] = tropospheric_delay(stationR.approx_position[0],stationR.approx_position[1],stationR.approx_position[2], gnss_tempR.Elevation, stationR.epoch)

            enuSatX, enuSatY, enuSatZ  = pm.ecef2enu(gnss_temp.X_Reception, gnss_temp.Y_Reception, gnss_temp.Z_Reception, lat0, lon0, h0)
            enuSatX = _np.array(enuSatX)
            enuSatY = _np.array(enuSatY)
            enuSatZ = _np.array(enuSatZ)

            coeffMatrixD = _np.zeros([len(gnss_temp)-1,2])

            for count in range(len(gnss_temp)-1):
                coeffMatrixD[count,0] = (userENU[0] - enuSatX[count]) / gnss_temp.Distance[count] - ((userENU[0] - enuSatX[count+1]) / gnss_temp.Distance[count+1])
                coeffMatrixD[count,1] = (userENU[1] - enuSatY[count]) / gnss_temp.Distance[count] - ((userENU[1] - enuSatY[count+1]) / gnss_temp.Distance[count+1])
                # coeffMatrixD[count,2] = (userENU[2] - enuSatZ[count]) / gnss_temp.Distance[count] - ((userENU[2] - enuSatZ[count+1]) / gnss_temp.Distance[count+1])

            lMatrix = _np.zeros([len(gnss_temp)-1,1])
            lMatrixR = _np.zeros([len(gnss_temp)-1,1])
            lMatrixD = _np.zeros([len(gnss_temp)-1,1])

            for count in range(len(gnss_temp)-1):
                lMatrix[count] = gnss_temp.Ionosphere_Free[count] - gnss_temp.Distance[count] + _CLIGHT * (gnss_temp.DeltaTSV[count] + gnss_temp.Relativistic_clock[count] - receiver_clock) - gnss_temp.Tropo[count] - (gnss_temp.Ionosphere_Free[count+1] - gnss_temp.Distance[count+1] + _CLIGHT * (gnss_temp.DeltaTSV[count+1] + gnss_temp.Relativistic_clock[count+1] - receiver_clock) - gnss_temp.Tropo[count+1])
                lMatrixR[count] = gnss_tempR.Ionosphere_Free[count] - gnss_tempR.Distance[count] + _CLIGHT * (gnss_tempR.DeltaTSV[count] + gnss_tempR.Relativistic_clock[count] - receiver_clockR) - gnss_tempR.Tropo[count] - (gnss_tempR.Ionosphere_Free[count+1] - gnss_tempR.Distance[count+1] + _CLIGHT * (gnss_tempR.DeltaTSV[count+1] + gnss_tempR.Relativistic_clock[count+1] - receiver_clockR) - gnss_tempR.Tropo[count+1])

            lMatrixD = lMatrix - lMatrixR

            if iter == 0:
                # DOP in NED coordinate
                GTG = _np.dot(coeffMatrixD.T, coeffMatrixD)
                DOP = _np.linalg.inv(GTG)
                HDOP = (DOP[0][0]+DOP[1][1])**0.5
                XDOP = (DOP[0][0])**0.5
                YDOP = (DOP[1][1])**0.5

            try:
                linearEquationSolutionD = _np.linalg.lstsq(coeffMatrixD,lMatrixD,rcond=None)
                xMatrixD = linearEquationSolutionD[0]
                # posD = [userENU[0] + xMatrixD[0], userENU[1] + xMatrixD[1], userENU[2] + xMatrixD[2]]
                posD = [userENU[0] + xMatrixD[0], userENU[1] + xMatrixD[1], userENU[2]] # 2D est
                userENU[0], userENU[1], userENU[2] = posD[0], posD[1], posD[2]
                approx_position[0],approx_position[1],approx_position[2] = pm.enu2ecef(userENU[0], userENU[1], userENU[2], lat0, lon0, h0)
                approx_position = _np.array(approx_position)
            except:
                print("Cannot solve normal equations for epoch", epoch_start,"| Skipping...")

        position_listD.append(posD)
        print(float(posD[0]), float(posD[1]), float(posD[2]), XDOP, YDOP, HDOP)

        epoch_start += epoch_offset
        epoch_step  += epoch_offset
        if (epoch_step - epoch_stop) > _timedelta(seconds=station.interval):
            break

    x_coordinateD = _np.mean([posD[0] for posD in position_listD])
    y_coordinateD = _np.mean([posD[1] for posD in position_listD])
    z_coordinateD = _np.mean([posD[2] for posD in position_listD])

    finish = time.time()     # Time of finish
    # print("Pseudorange calculation is done in", "{0:.2f}".format(finish-start), "seconds.")
    # print ("\n")

    # return (x_coordinate, y_coordinate, z_coordinate, rec_clock, x_coordinateR, y_coordinateR, z_coordinateR, rec_clockR)
    return (x_coordinateD, y_coordinateD, z_coordinateD)

def spp(station, orbit, system="G", cut_off=7.0):
    start = time.time() # Time of start
    if len(system)>1:
        raise Warning("SPP does not support multiple satellite system | This feature will be implemented in the next version")
    observation_list = _observation_picker(station, system)
    gnss = gnssDataframe(station, orbit, system, cut_off)

    #-----------------------------------------------------------------------------
    if len(observation_list) >=2:
        carrierPhase1 = getattr(gnss,observation_list[0][2])
        carrierPhase2 = getattr(gnss,observation_list[1][2])
        pseudorange1  = getattr(gnss,observation_list[0][3])
        pseudorange2  = getattr(gnss,observation_list[1][3])
        frequency1 = observation_list[0][4]
        frequency2 = observation_list[1][4]
    else:
        raise Warning("Ionosphere-free combination is not available")
    # ----------------------------------------------------------------------------

    gnss["Ionosphere_Free"] = (frequency1**2*pseudorange1-frequency2**2*pseudorange2)/(frequency1**2-frequency2**2)
    gnss = gnss.dropna(subset = ['Ionosphere_Free'])
    gnss["Travel_time"] = gnss["Ionosphere_Free"] / _CLIGHT
    gnss["X_Reception"],gnss["Y_Reception"],gnss["Z_Reception"] = _reception_coord(gnss.X, gnss.Y, gnss.Z, gnss.Vx, gnss.Vy, gnss.Vz, gnss.Travel_time)

    epochList =gnss.index.get_level_values("Epoch").unique().sort_values()
    epoch_start = epochList[0]
    epoch_offset= _timedelta(seconds=300)
    epoch_interval = _timedelta(seconds=station.interval-0.000001)
    epoch_stop  = epochList[-1] + _timedelta(seconds=0.000001)
    approx_position = [station.approx_position[0], station.approx_position[1], station.approx_position[2]]
    receiver_clock = station.receiver_clock
    position_list = []

    # [lat0, lon0, h0] = pm.ecef2geodetic(-3947764.0793, 3364399.9344, 3699430.4794) # MTKA
    [lat0, lon0, h0] = pm.ecef2geodetic(approx_position[0], approx_position[1], approx_position[2])
    userENU = pm.ecef2enu(approx_position[0],approx_position[1],approx_position[2], lat0, lon0, h0)
    userENU = _np.array(userENU)
    # print(approx_position)
    # print(lat0, lon0, h0)
    # print(userENU)

    userECFE = pm.enu2ecef(userENU[0],userENU[1],userENU[2], lat0, lon0, h0)
    userECFE = _np.array(userECFE)
    # print(userECFE)

    # test rotation matrix converting ECEF to ENU
    # lat0d = _np.radians(lat0)
    # lon0d = _np.radians(lon0)
    # rotationMat1 = [[_np.cos(lon0d),-1*_np.sin(lon0d),0],[_np.sin(lon0d),_np.cos(lon0d),0],[0,0,1]]
    # rotationMat2 = [[_np.cos(-lat0d),0,_np.sin(-lat0d)],[0,1,0],[-1*_np.sin(-lat0d),0,_np.cos(-lat0d)]]
    # rotationMat3 = [[0,0,1],[1,0,0],[0,1,0]]
    # rotationXYZ2ENU = _np.dot(_np.dot(rotationMat1,rotationMat2),rotationMat3)
    # userENU2 = _np.dot(approx_position,rotationXYZ2ENU)
    # userENU2 = _np.array(userENU2)
    # print(userENU2)

    while True:
        epoch_step = epoch_start + epoch_interval
        gnss_temp = gnss.xs((slice(epoch_start,epoch_step))).copy()

        for iter in range(6):

            distance = _distance_euclidean(approx_position[0],approx_position[1],approx_position[2], gnss_temp.X_Reception, gnss_temp.Y_Reception, gnss_temp.Z_Reception)
            gnss_temp["Distance"] = distance + _sagnac(approx_position[0],approx_position[1],approx_position[2], gnss_temp.X_Reception, gnss_temp.Y_Reception, gnss_temp.Z_Reception)
            gnss_temp["Azimuth"], gnss_temp["Elevation"], gnss_temp["Zenith"] = _azel(station.approx_position[0], station.approx_position[1], station.approx_position[2], gnss_temp.X, gnss_temp.Y, gnss_temp.Z, gnss_temp.Distance)
            gnss_temp["Tropo"] = tropospheric_delay(station.approx_position[0],station.approx_position[1],station.approx_position[2], gnss_temp.Elevation, station.epoch)

            # coeffMatrix = _np.zeros([len(gnss_temp),4])
            # coeffMatrix[:,0] =  (approx_position[0] - gnss_temp.X_Reception) / gnss_temp.Distance
            # coeffMatrix[:,1] =  (approx_position[1] - gnss_temp.Y_Reception) / gnss_temp.Distance
            # coeffMatrix[:,2] =  (approx_position[2] - gnss_temp.Z_Reception) / gnss_temp.Distance
            # coeffMatrix[:,3] =  1

            # from observer to target, ECEF => enu
            enuSatX, enuSatY, enuSatZ = pm.ecef2enu(gnss_temp.X_Reception, gnss_temp.Y_Reception, gnss_temp.Z_Reception, lat0, lon0, h0)
            enuSatX = _np.array(enuSatX)
            enuSatY = _np.array(enuSatY)
            enuSatZ = _np.array(enuSatZ)

            coeffMatrix = _np.zeros([len(gnss_temp),4])
            coeffMatrix[:,0] =  (userENU[0] - enuSatX) / gnss_temp.Distance
            coeffMatrix[:,1] =  (userENU[1] - enuSatY) / gnss_temp.Distance
            coeffMatrix[:,2] =  (userENU[2] - enuSatZ) / gnss_temp.Distance
            coeffMatrix[:,3] =  1

            lMatrix = gnss_temp.Ionosphere_Free - gnss_temp.Distance + _CLIGHT * (gnss_temp.DeltaTSV + gnss_temp.Relativistic_clock - receiver_clock) - gnss_temp.Tropo
            lMatrix = _np.array(lMatrix)

            # G = coeffMatrix
            # GTG = _np.dot(G.T, G)
            # GTG_inv = _np.linalg.inv(GTG)
            # GTG_inv_GT = _np.dot(GTG_inv, G.T)
            # xMatrix = _np.dot(GTG_inv_GT,lMatrix)
            #
            # posenu = [userENU[0] + xMatrix[0], userENU[1] + xMatrix[1], userENU[2] + xMatrix[2], receiver_clock + xMatrix[3] / _CLIGHT]
            # userENU[0] = userENU[0] + xMatrix[0]
            # userENU[1] = userENU[1] + xMatrix[1]
            # userENU[2] = userENU[2] + xMatrix[2]
            # receiver_clock = receiver_clock + xMatrix[3] / _CLIGHT
            # approx_position = pm.enu2ecef(userENU[0], userENU[1], userENU[2], lat0, lon0, h0)
            # approx_position = _np.array(approx_position)

            if iter == 0:
                # DOP in NED coordinate
                GTG = _np.dot(coeffMatrix.T, coeffMatrix)
                DOP = _np.linalg.inv(GTG)
                HDOP = (DOP[0][0]+DOP[1][1])**0.5
                XDOP = (DOP[0][0])**0.5
                YDOP = (DOP[1][1])**0.5
                TDOP = (DOP[3][3])**0.5
                # print (XDOP, YDOP, HDOP)

            try:
                linearEquationSolution = _np.linalg.lstsq(coeffMatrix,lMatrix,rcond=None)
                xMatrix = linearEquationSolution[0]
                posenu = [userENU[0] + xMatrix[0], userENU[1] + xMatrix[1], userENU[2] + xMatrix[2], receiver_clock + xMatrix[3] / _CLIGHT]
                userENU[0], userENU[1], userENU[2], receiver_clock = posenu[0], posenu[1], posenu[2], posenu[3]
                approx_position[0],approx_position[1],approx_position[2] = pm.enu2ecef(userENU[0], userENU[1], userENU[2], lat0, lon0, h0)
                approx_position = _np.array(approx_position)

            except:
                print("Cannot solve normal equations for epoch", epoch_start,"| Skipping...")

        position_list.append(posenu)
        print(posenu, XDOP, YDOP, HDOP)

        # [lat, lon, h] = pm.ecef2geodetic(pos[0],pos[1],pos[2])
        # positionNED = []
        # positionNED = pm.geodetic2enu(lat, lon, h, lat0, lon0, h0)
        # print(positionNED)

        # print(pos)
        epoch_start += epoch_offset
        epoch_step  += epoch_offset
        if (epoch_step - epoch_stop) > _timedelta(seconds=station.interval):
            break

    x_coordinate = _np.mean([posenu[0] for pos in position_list])
    y_coordinate = _np.mean([posenu[1] for pos in position_list])
    z_coordinate = _np.mean([posenu[2] for pos in position_list])
    rec_clock    = _np.mean([posenu[3] for pos in position_list])
    finish = time.time()     # Time of finish
    print("Pseudorange calculation is done in", "{0:.2f}".format(finish-start), "seconds.")
    return (x_coordinate, y_coordinate, z_coordinate, rec_clock)

def gnssDataframe(station, orbit, system="G+R+E+C+J+I+S", cut_off=7.0):
    try:
        system = _itemgetter(*system.split("+"))(_SYSTEM_NAME)
        if type(system)==str: system = tuple([system])
    except KeyError:
        raise Warning("Unknown Satellite System:", system, "OPTIONS: G-R-E-C-J-R-I-S")
    epochMatch = station.observation.index.intersection(orbit.index)
    gnss = _pd.concat([station.observation.loc[epochMatch].copy(), orbit.loc[epochMatch]], axis=1)
    gnss = gnss[gnss['SYSTEM'].isin(system)]
    gnss["Distance"] = _distance_euclidean(station.approx_position[0], station.approx_position[1], station.approx_position[2], gnss.X, gnss.Y, gnss.Z)
    gnss["Relativistic_clock"] = _relativistic_clock(gnss.X, gnss.Y, gnss.Z, gnss.Vx, gnss.Vy, gnss.Vz)
    gnss['Azimuth'], gnss['Elevation'], gnss['Zenith'] = _azel(station.approx_position[0], station.approx_position[1], station.approx_position[2], gnss.X, gnss.Y, gnss.Z, gnss.Distance)
    gnss = gnss.loc[gnss['Elevation'] > cut_off]
    gnss["Tropo"] = tropospheric_delay(station.approx_position[0],station.approx_position[1],station.approx_position[2], gnss.Elevation, station.epoch)
    return gnss

def multipath(station, system="G"):
    if len(system)>1:
        raise Warning("Multiple satellite system is not applicable for multipath | This feature will be implemented in next version.")
    observation_list = _observation_picker(station, system=system)
    observation = station.observation.dropna(subset=[observation_list[0][2],observation_list[1][2],observation_list[0][3],observation_list[1][3]])
    observation = observation.loc[observation.SYSTEM==_SYSTEM_NAME[system]].copy(deep=True)
    carrierPhase1 = getattr(observation,observation_list[0][2])
    carrierPhase2 = getattr(observation,observation_list[1][2])
    pseudorange1  = getattr(observation,observation_list[0][3])
    pseudorange2  = getattr(observation,observation_list[1][3])
    frequency1 = observation_list[0][4]
    frequency2 = observation_list[1][4]
    lam1 = _CLIGHT/frequency1
    lam2 = _CLIGHT/frequency2
    ioncoeff = (frequency1/frequency2)**2
    observation["Multipath1"] = pseudorange1 - (2/(ioncoeff-1)+1)*(carrierPhase1*lam1) + (2/(ioncoeff-1))*(carrierPhase2*lam2)
    observation["Multipath2"] = pseudorange2 - (2*ioncoeff/(ioncoeff-1))*(carrierPhase1*lam1) + (2*ioncoeff/(ioncoeff-1)-1)*(carrierPhase2*lam2)
    observation = observation.reorder_levels(['SV','Epoch'])
    observation = observation.sort_index()
    sv_list = observation.index.get_level_values('SV').unique()
    # ----------------------------------------------------------------------------
    Multipath1 = []
    Multipath2 = []
    for sv in sv_list:
        ObsSV = observation.loc[sv]
        multipathSV1 = []
        multipathSV2 = []
        j = 0
        for i in range(1, len(ObsSV)):
            if (ObsSV.iloc[i].epoch - ObsSV.iloc[i-1].epoch) > _pd.Timedelta('0 days 00:15:00'):
                multipath1 = ObsSV.iloc[j:i].Multipath1.values - _np.nanmean(ObsSV.iloc[j:i].Multipath1.values)
                multipath2 = ObsSV.iloc[j:i].Multipath1.values - _np.nanmean(ObsSV.iloc[j:i].Multipath1.values)
                multipathSV1.extend(multipath1)
                multipathSV2.extend(multipath2)
                j=i
        multipath1 = ObsSV.iloc[j:].Multipath1.values - _np.nanmean(ObsSV.iloc[j:].Multipath1.values)
        multipath2 = ObsSV.iloc[j:].Multipath1.values - _np.nanmean(ObsSV.iloc[j:].Multipath1.values)
        multipathSV1.extend(multipath1)
        multipathSV2.extend(multipath2)
        Multipath1.extend(multipathSV1)
        Multipath2.extend(multipathSV2)
    # Re-assign multipath values
    observation["Multipath1"] = Multipath1
    observation["Multipath2"] = Multipath2
    return observation

def _adjustment(coeffMatrix,LMatrix):
    NMatrix = _np.linalg.inv(_np.dot(_np.transpose(coeffMatrix), coeffMatrix))
    nMatrix = _np.matmul(_np.transpose(coeffMatrix), LMatrix)
    XMatrix = _np.dot(NMatrix, nMatrix)
    vMatrix = _np.dot(coeffMatrix, XMatrix) - LMatrix
    m0 = _np.sqrt(_np.dot(_np.transpose(vMatrix), vMatrix)/(len(LMatrix)-len(NMatrix)))
    diagN = _np.diag(NMatrix)
    rmse = m0*_np.sqrt(diagN)
    mp = _np.sqrt(rmse[0]**2+rmse[1]**2+rmse[2]**2)
    return XMatrix, rmse

def _observation_picker(station, system="G"):
    try:
        system = _SYSTEM_NAME[system.upper()]
    except KeyError:
        raise Warning("Unknown Satellite System:", system, "OPTIONS: G-R-E-C-J-R-I-S")
    #-------------------------------------------------------------------
    # RINEX-3
    if station.version.startswith("3"):
        observation_codes = station.observation.columns.tolist()
        system_observations = getattr(station.observation_types, system)
        band_list         = set("L" + code[1] for code in observation_codes if len(code)==3)
        channel_list      = set([code[2] for code in observation_codes if len(code)==3])
        obs_codes = []
        for band in band_list:
            if band in _SYSTEM_RNX3[system]:
                for channel in channel_list:
                    if (band+channel) in _SYSTEM_RNX3[system][band]["Carrierphase"] and (band+channel) in system_observations:
                        obs_codes.append([system,band,(band+channel),("C"+band[1]+channel),_SYSTEM_RNX3[system][band]["Frequency"]])
                        break
    # RINEX-2
    elif station.version.startswith("2"):
        observation_codes = station.observation.columns.tolist()
        system_observations = station.observation_types
        band_list         = set(code for code in observation_codes if code.startswith(("L")))
        obs_codes = []
        for band in band_list:
            if band in _SYSTEM_RNX2[system].keys():
                for code in _SYSTEM_RNX2[system][band]["Pseudorange"]:
                    if code in system_observations:
                        obs_codes.append([system,band,band,code,_SYSTEM_RNX2[system][band]["Frequency"]])
                        break
    obs_codes = sorted(obs_codes, key=_itemgetter(1))
    return (obs_codes[0],obs_codes[1])

def _observation_picker_by_band(station, system="G", band="L1"):
    #-------------------------------------------------------------------
    try:
        system = _SYSTEM_NAME[system.upper()]
        if band not in _SYSTEM_RNX3[system].keys():
            raise Warning(band,"band cannot be found in",system,"satellite system! Band options for",system,"system:",tuple(_SYSTEM_RNX3[system].keys()))
    except KeyError:
        raise Warning("Unknown Satellite System:", system, "OPTIONS: G-R-E-C-J-R-I-S")
    #-------------------------------------------------------------------

    # RINEX-3
    if station.version.startswith("3"):
        observation_codes = station.observation.columns.tolist()
        system_observations = getattr(station.observation_types, system)
        channel_list      = set([code[2] for code in observation_codes if len(code)==3])
        obs_codes = []
        if band in _SYSTEM_RNX3[system]:
            for channel in channel_list:
                if (band+channel) in _SYSTEM_RNX3[system][band]["Carrierphase"] and (band+channel) in system_observations:
                    obs_codes.append([system,band,(band+channel),("C"+band[1]+channel),_SYSTEM_RNX3[system][band]["Frequency"],("D"+band[1]+channel),("S"+band[1]+channel)])
                    break
    # RINEX-2
    elif station.version.startswith("2"):
        observation_codes = station.observation.columns.tolist()
        system_observations = station.observation_types
        obs_codes = []
        if band in _SYSTEM_RNX2[system].keys():
            for code in _SYSTEM_RNX2[system][band]["Pseudorange"]:
                if code in system_observations:
                    obs_codes.append([system,band,band,code,_SYSTEM_RNX2[system][band]["Frequency"],("D"+band[1]),("S"+band[1])])
                    break
    return (obs_codes[0])
