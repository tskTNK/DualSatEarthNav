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
# ===========================================================
__all__ = ["spp","multipath"]

def mdpo(station, stationR, orbit, system="G", cut_off=7.0):
    start = time.time() # Time of start
    if len(system)>1:
        raise Warning("SPP does not support multiple satellite system | This feature will be implemented in the next version")
    observation_list = _observation_picker(station, system)
    observation_listR = _observation_picker(stationR, system)

    # print("observation_list")
    # print(observation_list)
    # print(observation_listR)

    gnss = gnssDataframe(station, orbit, system, cut_off)
    gnssR = gnssDataframe(stationR, orbit, system, cut_off)

    # print("gnss")
    # print(gnss)

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

    # print("pseudorange1")
    # print(pseudorange1)

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

    coeffMatrixD = _np.zeros([100,2])
    lMatrixD = _np.zeros([100,1])

    numOfEpochs = 0

    previous_matchSatelliteList_select = []

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

        # adding satellite clock bias for debug
        for count in range(len(gnss_temp)):
            gnss_temp.Relativistic_clock[count] = 0.0000002*count
        for count in range(len(gnss_tempR)):
            gnss_tempR.Relativistic_clock[count] = 0.0000002*count

        # pick up two satellites to use
        indices = [0, 1]
        matchSatelliteList_select = []
        for index in indices:
            matchSatelliteList_select.append(matchSatelliteList[index])

        if previous_matchSatelliteList_select == matchSatelliteList_select:
            # print ("keizoku")
            # print ("epoch num")
            # print (epoch_step)

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

            coeffMatrixD[numOfEpochs,0] = (approx_position[0] - gnss_temp.X_Reception[sat1]) / gnss_temp.Distance[sat1] - ((approx_position[0] - gnss_temp.X_Reception[sat2]) / gnss_temp.Distance[sat2])
            coeffMatrixD[numOfEpochs,1] = (approx_position[1] - gnss_temp.Y_Reception[sat1]) / gnss_temp.Distance[sat1] - ((approx_position[1] - gnss_temp.Y_Reception[sat2]) / gnss_temp.Distance[sat2])

            lMatrix = gnss_temp.Ionosphere_Free[sat1] - gnss_temp.Distance[sat1] + _CLIGHT * (gnss_temp.DeltaTSV[sat1] + gnss_temp.Relativistic_clock[sat1] - receiver_clock) - gnss_temp.Tropo[sat1] - (gnss_temp.Ionosphere_Free[sat2] - gnss_temp.Distance[sat2] + _CLIGHT * (gnss_temp.DeltaTSV[sat2] + gnss_temp.Relativistic_clock[sat2] - receiver_clock) - gnss_temp.Tropo[sat2])
            lMatrixR = gnss_tempR.Ionosphere_Free[sat1] - gnss_tempR.Distance[sat1] + _CLIGHT * (gnss_tempR.DeltaTSV[sat1] + gnss_tempR.Relativistic_clock[sat1] - receiver_clockR) - gnss_tempR.Tropo[sat1] - (gnss_tempR.Ionosphere_Free[sat2] - gnss_tempR.Distance[sat2] + _CLIGHT * (gnss_tempR.DeltaTSV[sat2] + gnss_tempR.Relativistic_clock[sat2] - receiver_clockR) - gnss_tempR.Tropo[sat2])
            lMatrixD[numOfEpochs,0] = lMatrix - lMatrixR

            numOfEpochs = numOfEpochs + 1

        else:
            # print ("stop at epoch")
            # print (numOfEpochs)

            if numOfEpochs > 10:
                # DOP Calculation
                # print (coeffMatrixD[0:numOfEpochs])
                # print (lMatrixD[0:numOfEpochs])
                GTG = _np.dot(coeffMatrixD.T, coeffMatrixD)
                DOP = _np.linalg.inv(GTG)
                GDOP = (DOP[0][0]+DOP[1][1])**0.5
                XDOP = (DOP[0][0])**0.5
                YDOP = (DOP[1][1])**0.5
                # print("GDOP, XDOP, YDOP")
                # print([GDOP, XDOP, YDOP])

                if GDOP < 10:
                    try:
                        linearEquationSolutionD = _np.linalg.lstsq(coeffMatrixD,lMatrixD,rcond=None)
                        xMatrixD = linearEquationSolutionD[0]
                        # print (xMatrixD[0], xMatrixD[1])
                        approx_position[0], approx_position[1], approx_position[2] = approx_position[0] + xMatrixD[0], approx_position[1] + xMatrixD[1], approx_position[2]
                        posD = [approx_position[0]-approx_positionR[0], approx_position[1]-approx_positionR[1], approx_position[2]-approx_positionR[2]]
                    except:
                        print("Cannot solve normal equations for epoch", epoch_start,"| Skipping...")

                    # print("posD")
                    # print([posD[0], posD[1], posD[2]])
                    position_listD.append(posD)

            numOfEpochs = 0
            coeffMatrixD = _np.zeros([100,2])
            lMatrixD = _np.zeros([100,1])

        previous_matchSatelliteList_select = matchSatelliteList_select

        epoch_start += epoch_offset
        epoch_step  += epoch_offset
        if (epoch_step - epoch_stop) > _timedelta(seconds=station.interval):
            break

    # for count in range(len(position_listD)):
    #     print (position_listD[count])

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

        # adding satellite clock bias for debug
        for count in range(len(gnss_temp)):
            gnss_temp.Relativistic_clock[count] = 0.0000002*count
        for count in range(len(gnss_tempR)):
            gnss_tempR.Relativistic_clock[count] = 0.0000002*count

        for iter in range(6):
            distance = _distance_euclidean(approx_position[0],approx_position[1],approx_position[2], gnss_temp.X_Reception, gnss_temp.Y_Reception, gnss_temp.Z_Reception)
            # why distance is calculated based on the satellite position at the 'reception' timing instead 'emission' timing?
            gnss_temp["Distance"] = distance + _sagnac(approx_position[0],approx_position[1],approx_position[2], gnss_temp.X_Reception, gnss_temp.Y_Reception, gnss_temp.Z_Reception)
            gnss_temp["Azimuth"], gnss_temp["Elevation"], gnss_temp["Zenith"] = _azel(station.approx_position[0], station.approx_position[1], station.approx_position[2], gnss_temp.X, gnss_temp.Y, gnss_temp.Z, gnss_temp.Distance)
            gnss_temp["Tropo"] = tropospheric_delay(station.approx_position[0],station.approx_position[1],station.approx_position[2], gnss_temp.Elevation, station.epoch)
            coeffMatrix = _np.zeros([len(gnss_temp),4])
            coeffMatrix[:,0] = (approx_position[0] - gnss_temp.X_Reception) / gnss_temp.Distance
            coeffMatrix[:,1] = (approx_position[1] - gnss_temp.Y_Reception) / gnss_temp.Distance
            coeffMatrix[:,2] = (approx_position[2] - gnss_temp.Z_Reception) / gnss_temp.Distance
            coeffMatrix[:,3] =  1
            lMatrix = gnss_temp.Ionosphere_Free - gnss_temp.Distance + _CLIGHT * (gnss_temp.DeltaTSV + gnss_temp.Relativistic_clock - receiver_clock) - gnss_temp.Tropo
            lMatrix = _np.array(lMatrix)

            try:
                linearEquationSolution = _np.linalg.lstsq(coeffMatrix,lMatrix,rcond=None)
                xMatrix = linearEquationSolution[0]
                pos = [approx_position[0] + xMatrix[0], approx_position[1] + xMatrix[1], approx_position[2] + xMatrix[2], receiver_clock + xMatrix[3] / _CLIGHT]
                approx_position[0], approx_position[1] , approx_position[2], receiver_clock = pos[0], pos[1], pos[2], pos[3]
            except:
                print("Cannot solve normal equations for epoch", epoch_start,"| Skipping...")

        position_list.append(pos)

        for iter in range(6):
            distanceR = _distance_euclidean(approx_positionR[0],approx_positionR[1],approx_positionR[2], gnss_tempR.X_Reception, gnss_tempR.Y_Reception, gnss_tempR.Z_Reception)
            # why distance is calculated based on the satellite position at the 'reception' timing instead 'emission' timing?
            gnss_tempR["Distance"] = distanceR + _sagnac(approx_positionR[0],approx_positionR[1],approx_positionR[2], gnss_tempR.X_Reception, gnss_tempR.Y_Reception, gnss_tempR.Z_Reception)
            gnss_tempR["Azimuth"], gnss_tempR["Elevation"], gnss_tempR["Zenith"] = _azel(stationR.approx_position[0], stationR.approx_position[1], stationR.approx_position[2], gnss_tempR.X, gnss_tempR.Y, gnss_tempR.Z, gnss_tempR.Distance)
            gnss_tempR["Tropo"] = tropospheric_delay(stationR.approx_position[0],stationR.approx_position[1],stationR.approx_position[2], gnss_tempR.Elevation, stationR.epoch)
            coeffMatrixR = _np.zeros([len(gnss_tempR),4])
            coeffMatrixR[:,0] = (approx_positionR[0] - gnss_tempR.X_Reception) / gnss_tempR.Distance
            coeffMatrixR[:,1] = (approx_positionR[1] - gnss_tempR.Y_Reception) / gnss_tempR.Distance
            coeffMatrixR[:,2] = (approx_positionR[2] - gnss_tempR.Z_Reception) / gnss_tempR.Distance
            coeffMatrixR[:,3] =  1
            lMatrixR = gnss_tempR.Ionosphere_Free - gnss_tempR.Distance + _CLIGHT * (gnss_tempR.DeltaTSV + gnss_tempR.Relativistic_clock - receiver_clockR) - gnss_tempR.Tropo
            lMatrixR = _np.array(lMatrixR)
            try:
                linearEquationSolutionR = _np.linalg.lstsq(coeffMatrixR,lMatrixR,rcond=None)
                xMatrixR = linearEquationSolutionR[0]
                posR = [approx_positionR[0] + xMatrixR[0], approx_positionR[1] + xMatrixR[1], approx_positionR[2] + xMatrixR[2], receiver_clockR + xMatrixR[3] / _CLIGHT]
                approx_positionR[0], approx_positionR[1] , approx_positionR[2], receiver_clockR = posR[0], posR[1], posR[2], posR[3]
            except:
                print("Cannot solve normal equations for epoch", epoch_start,"| Skipping...")

        position_listR.append(posR)

        epoch_start += epoch_offset
        epoch_step  += epoch_offset
        if (epoch_step - epoch_stop) > _timedelta(seconds=station.interval):
            break

    x_coordinate = _np.mean([pos[0] for pos in position_list])
    y_coordinate = _np.mean([pos[1] for pos in position_list])
    z_coordinate = _np.mean([pos[2] for pos in position_list])
    rec_clock    = _np.mean([pos[3] for pos in position_list])

    x_coordinateR = _np.mean([posR[0] for posR in position_listR])
    y_coordinateR = _np.mean([posR[1] for posR in position_listR])
    z_coordinateR = _np.mean([posR[2] for posR in position_listR])
    rec_clockR    = _np.mean([posR[3] for posR in position_listR])

    finish = time.time()     # Time of finish
    print("Pseudorange calculation is done in", "{0:.2f}".format(finish-start), "seconds.")
    print ("\n")

    return (x_coordinate, y_coordinate, z_coordinate, rec_clock, x_coordinateR, y_coordinateR, z_coordinateR, rec_clockR)

def spp(station, orbit, system="G", cut_off=7.0):
    start = time.time() # Time of start
    if len(system)>1:
        raise Warning("SPP does not support multiple satellite system | This feature will be implemented in the next version")
    observation_list = _observation_picker(station, system)

    # print("observation_list")
    # print(observation_list)

    gnss = gnssDataframe(station, orbit, system, cut_off)

    # print("gnss")
    # print(gnss)

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

    # print("pseudorange1")
    # print(pseudorange1)

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
    while True:
        epoch_step = epoch_start + epoch_interval
        gnss_temp = gnss.xs((slice(epoch_start,epoch_step))).copy()

        # print("epoch_step")
        # print(epoch_step)
        # print("gnss_temp")
        # print(gnss_temp)

        for iter in range(6):
            distance = _distance_euclidean(approx_position[0],approx_position[1],approx_position[2], gnss_temp.X_Reception, gnss_temp.Y_Reception, gnss_temp.Z_Reception)
            # why distance is calculated based on the satellite position at the 'reception' timing instead 'emission' timing?
            gnss_temp["Distance"] = distance + _sagnac(approx_position[0],approx_position[1],approx_position[2], gnss_temp.X_Reception, gnss_temp.Y_Reception, gnss_temp.Z_Reception)
            gnss_temp["Azimuth"], gnss_temp["Elevation"], gnss_temp["Zenith"] = _azel(station.approx_position[0], station.approx_position[1], station.approx_position[2], gnss_temp.X, gnss_temp.Y, gnss_temp.Z, gnss_temp.Distance)
            gnss_temp["Tropo"] = tropospheric_delay(station.approx_position[0],station.approx_position[1],station.approx_position[2], gnss_temp.Elevation, station.epoch)
            coeffMatrix = _np.zeros([len(gnss_temp),4])
            coeffMatrix[:,0] = (approx_position[0] - gnss_temp.X_Reception) / gnss_temp.Distance
            coeffMatrix[:,1] = (approx_position[1] - gnss_temp.Y_Reception) / gnss_temp.Distance
            coeffMatrix[:,2] = (approx_position[2] - gnss_temp.Z_Reception) / gnss_temp.Distance
            coeffMatrix[:,3] =  1
            lMatrix = gnss_temp.Ionosphere_Free - gnss_temp.Distance + _CLIGHT * (gnss_temp.DeltaTSV + gnss_temp.Relativistic_clock - receiver_clock) - gnss_temp.Tropo
            lMatrix = _np.array(lMatrix)
            try:
                linearEquationSolution = _np.linalg.lstsq(coeffMatrix,lMatrix,rcond=None)
                xMatrix = linearEquationSolution[0]
                pos = [approx_position[0] + xMatrix[0], approx_position[1] + xMatrix[1], approx_position[2] + xMatrix[2], receiver_clock + xMatrix[3] / _CLIGHT]
                approx_position[0], approx_position[1] , approx_position[2], receiver_clock = pos[0], pos[1], pos[2], pos[3]
            except:
                print("Cannot solve normal equations for epoch", epoch_start,"| Skipping...")
        position_list.append(pos)
        epoch_start += epoch_offset
        epoch_step  += epoch_offset
        if (epoch_step - epoch_stop) > _timedelta(seconds=station.interval):
            break
    x_coordinate = _np.mean([pos[0] for pos in position_list])
    y_coordinate = _np.mean([pos[1] for pos in position_list])
    z_coordinate = _np.mean([pos[2] for pos in position_list])
    rec_clock    = _np.mean([pos[3] for pos in position_list])
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
