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

# spp_resultR = spp(stationR, orbitR, system="G", cut_off=7.0) # return (x_coordinate, y_coordinate, z_coordinate, rec_clock)
# print (stationR.approx_position)
# print (spp_resultR)

station = read_obsFile("kgni0320.21o")
orbit = sp3_interp(station.epoch, interval=station.interval, poly_degree=16, sp3_product="gfz", clock_product="gfz")

# spp_result = spp(station, orbit, system="G", cut_off=7.0) # return (x_coordinate, y_coordinate, z_coordinate, rec_clock)
# print (station.approx_position)
# print (spp_result)

#-----------------------------------------------------------------------------
if station.epoch == stationR.epoch:
    print("epochs are same")
else:
    raise Warning("Start epochs are different")
# ----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
if station.interval == stationR.interval:
    print("intervals are same")
else:
    raise Warning("intervals are different")
# ----------------------------------------------------------------------------

initialDifference = [station.approx_position[0]-stationR.approx_position[0], station.approx_position[1]-stationR.approx_position[1], station.approx_position[2]-stationR.approx_position[2]]
print ("\n Difference: initial")
print (initialDifference[0], initialDifference[1], initialDifference[2])
print ("\n")

spp_resultmdpo = mdpo(station, stationR, orbitR, system="G", cut_off=7.0)
print ("\n MDPO estimation")
print (spp_resultmdpo[0], spp_resultmdpo[1], spp_resultmdpo[2])
print ("\n Difference: MDPO estimation")
print (initialDifference[0]-spp_resultmdpo[0], initialDifference[1]-spp_resultmdpo[1], initialDifference[2]-spp_resultmdpo[2])
print ("\n")

spp_resultdpp = dpp(station, stationR, orbitR, system="G", cut_off=7.0) # return (x_coordinate, y_coordinate, z_coordinate, rec_clock)
# print ("\n")
# print ("single estimation: station")
# print (station.approx_position)
# print (spp_resultdpp[0],spp_resultdpp[1],spp_resultdpp[2],spp_resultdpp[3])
# print ("\n")
# print ("difference")
# print (station.approx_position[0]-spp_resultdpp[0],station.approx_position[1]-spp_resultdpp[1],station.approx_position[2]-spp_resultdpp[2])
# print ("\n")
# print ("single estimation: stationR")
# print (stationR.approx_position)
# print (spp_resultdpp[4],spp_resultdpp[5],spp_resultdpp[6],spp_resultdpp[7])
# print ("\n")
# print ("difference")
# print (stationR.approx_position[0]-spp_resultdpp[4],stationR.approx_position[1]-spp_resultdpp[5],stationR.approx_position[2]-spp_resultdpp[6])
print ("\n Differential GPS with all possible satellites")
print (spp_resultdpp[0]-spp_resultdpp[4], spp_resultdpp[1]-spp_resultdpp[5], spp_resultdpp[2]-spp_resultdpp[6])
print ("\n Difference: Differential GPS with all possible satellites")
print (initialDifference[0]-(spp_resultdpp[0]-spp_resultdpp[4]), initialDifference[1]-(spp_resultdpp[1]-spp_resultdpp[5]), initialDifference[2]-(spp_resultdpp[2]-spp_resultdpp[6]))

# python setup.py install
