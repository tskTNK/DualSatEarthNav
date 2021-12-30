# import gnsspy as gp
from gnsspy.position.position import spp
from gnsspy.position.interpolation import sp3_interp
from gnsspy.io.readFile import read_obsFile

# station = gp.read_obsFile("kgni0320.21o")
# orbit = gp.sp3_interp(station.epoch, interval=station.interval, poly_degree=16, sp3_product="gfz", clock_product="gfz")
# spp_result = gp.spp(station, orbit, system="G", cut_off=7.0) # return (x_coordinate, y_coordinate, z_coordinate, rec_clock)
station = read_obsFile("kgni0320.21o")
orbit = sp3_interp(station.epoch, interval=station.interval, poly_degree=16, sp3_product="gfz", clock_product="gfz")
spp_result = spp(station, orbit, system="G", cut_off=7.0) # return (x_coordinate, y_coordinate, z_coordinate, rec_clock)

# initial values
print (station.approx_position)
# estimated position after averaged
print (spp_result)
