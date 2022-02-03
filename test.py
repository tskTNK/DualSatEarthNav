# import gnsspy as gp
# station = gp.read_obsFile("kgni0320.21o")
# orbit = gp.sp3_interp(station.epoch, interval=station.interval, poly_degree=16, sp3_product="gfz", clock_product="gfz")
# spp_result = gp.spp(station, orbit, system="G", cut_off=7.0) # return (x_coordinate, y_coordinate, z_coordinate, rec_clock)

from gnsspy.position.position import spp, dpp, mdpo
from gnsspy.position.interpolation import sp3_interp
from gnsspy.io.readFile import read_obsFile

# Standard Pseudorange Calculation

stationR = read_obsFile("mtka0320.21o")
orbitR = sp3_interp(stationR.epoch, interval=stationR.interval, poly_degree=16, sp3_product="gfz", clock_product="gfz")
#
# spp_resultR = spp(stationR, orbitR, system="G", cut_off=7.0) # return (x_coordinate, y_coordinate, z_coordinate, rec_clock)
# print (stationR.approx_position)
# print (spp_resultR)

station = read_obsFile("kgni0320.21o")
orbit = sp3_interp(station.epoch, interval=station.interval, poly_degree=16, sp3_product="gfz", clock_product="gfz")

# spp_result = spp(station, orbit, system="G", cut_off=7.0) # return (x_coordinate, y_coordinate, z_coordinate, rec_clock)
# print (station.approx_position)
# print (spp_result)

# #-----------------------------------------------------------------------------
# if station.epoch == stationR.epoch:
#     print("epochs are same")
# else:
#     raise Warning("Start epochs are different")
# # ----------------------------------------------------------------------------
# #-----------------------------------------------------------------------------
# if station.interval == stationR.interval:
#     print("intervals are same")
# else:
#     raise Warning("intervals are different")
# # ----------------------------------------------------------------------------
#
# initialDifference = [station.approx_position[0]-stationR.approx_position[0], station.approx_position[1]-stationR.approx_position[1], station.approx_position[2]-stationR.approx_position[2]]


spp_resultdpp = dpp(station, stationR, orbitR, system="G", cut_off=7.0) # return (x_coordinate, y_coordinate, z_coordinate, rec_clock)

# spp_resultmdpo = mdpo(station, stationR, orbitR, system="G", cut_off=7.0)

# python setup.py install
