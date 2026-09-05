import rdata

parsed = rdata.parser.parse_file("yg821jf8611_ny_statewide_2020_04_01.rds")
data = rdata.conversion.convert(parsed)

print(type(data))
print(data)