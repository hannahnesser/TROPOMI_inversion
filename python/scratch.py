import sys

sat_data_dir = sys.argv[1]
GC_data_dir = sys.argv[2]
output_dir = sys.argv[3]

LON_MIN = sys.argv[4]
LON_MAX = sys.argv[5]
LON_DELTA = sys.argv[6]

LAT_MIN = sys.argv[7]
LAT_MAX = sys.argv[8]
LAT_DELTA = sys.argv[9]

BUFFER = sys.argv[10:14]

YEAR = sys.argv[14]
MONTH = sys.argv[15]

print(sat_data_dir)
print(GC_data_dir)
print(output_dir)
print(LON_MIN)
print(LON_MAX)
print(LON_DELTA)
print(LAT_MIN)
print(LAT_MAX)
print(LAT_DELTA)
print(BUFFER)
print(YEAR)
print(MONTH)
