#### Heading ####
# Ringworld_analysis.R
#
# This file is used to generate qualitative ringworld data regarding surface vs
# edge atmospheric losses from particle numbers recaptured and escaped generated
# using the main Python codes for the simulation. It outputs the ringworld size
# expected to generate 100% losses and the loss rate associated with the face.
#
# V2.0.1, Eric Comstock, 16/07/2025
# V2.0, Eric Comstock, 12/07/2025
# V1.2.1, Eric Comstock, 09/06/2025
# V1.2, Eric Comstock, 09/06/2025
# V1.1, Eric Comstock, 02/06/2025
# V1.0, Eric Comstock, 02/06/2025

#### Data input ####

# Freitas ringworld is an outlier - figure out why this is

#First the width variation tests, then Edwin's gravity tests, then the general tests are added

# Ringworld test widths (km)
worldWidth  = c(1600000,100000,10000,1000,1000,1000,1000,1000,1000,1000,
                200, 2000, 1000000, 200, 3957.98762536996)

# Ringworld gravity (g)
worldGrav  = c(1,1,1,1,0.25, 0.5, 1, 1.5, 2, 3,
               1, 1, 0.00956122530571083/9.81, 1, 1)

# Ringworld radius (km)
worldRad = c(149598000,149598000,149598000,149598000,149598000,149598000,149598000,149598000,149598000,149598000,
             1000,1854977.7, 9.4e13, 1854977.7, 395798.762536996)

# Number of particles recaptured
recaptured             = c(953897 , 94555 , 94414 , 93713,
                           30106, 37635, 46811, 49349, 49701, 49799,
                           90559, 94174, 43002, 90855, 94362)
# Number of particles escaped
escaped                = c(4, 16, 78, 851,682, 560, 422, 327, 275, 201,
                           4307, 392, 27, 4251, 209)

#### Process data ####

escape_frac_list       = c()                         # Fraction of particles escaped
escape_frac_std_list   = c()                         # Std. dev. of the above fraction

fudge_factor = 0# Should be zero in the final code - prevents infinity from showing up

for (i in 1: length(worldGrav))                 # Data processing
{
  rec_p                = integer(recaptured[i]) # Vector of zeros with length recaptured[i]
  esc_p                = integer(escaped[i]) + 1# Vector of ones with length escaped[i]
  
  alldata              = c(rec_p, esc_p)        # Concatenated vector with a 1 for
                                                #each escape, and a 0 for each recapture
  
  avg_escape           = mean(alldata)          # Expected value of the binary escape data
  std_escape           = sqrt(var(alldata))     # Std. dev. of the binary escape data
  
  std_avg_escape       = std_escape / sqrt(length(alldata))# Std. dev. of the Expected 
                                                #value of the binary escape data
  
  # Printing results for clarity
  print(paste0('Trial ', as.character(i)))
  print(paste0('Expected escape chance: ', as.character(avg_escape)))
  print(paste0('Expected escape chance std. dev.: ', as.character(std_avg_escape)))
  print('')
  
  escape_frac_list     = c(escape_frac_list, avg_escape)
  escape_frac_std_list = c(escape_frac_std_list, std_avg_escape) * (1 - fudge_factor)
}

# Calculating linear regressions for average, +1s.d., and -1s.d. escape fraction data
l.normal               = lm(log10(escape_frac_list) ~ log10(worldGrav) + log10(worldWidth) + log10(worldRad))
l.high                 = lm(log10(escape_frac_list+escape_frac_std_list) ~ log10(worldGrav) + log10(worldWidth) + log10(worldRad))
l.low                  = lm(log10(escape_frac_list-escape_frac_std_list) ~ log10(worldGrav) + log10(worldWidth) + log10(worldRad))

# No plotting this time

#### Linear regression and final results ####

# Print linear regression statistics
l.normal
l.high
l.low

# (Intercept) coefficient shows log10 of leakage rate per particle for 1g ringworlds
#that are 1 km in width and 1 km in radius. Right now we can predict this with a
#30% error, an it is around 10 km

# log10(worldGrav) coefficient shows polynomial relation between leakage and world
#gravity. Leakage ~ gravity ^ -0.62 +- 0.03, so since atmo mass is inversely proportional
#to gravity, lifespan ~ gravity ^ -0.38 +- 0.03

# log10(worldWidth) coefficient shows a statistically linear correlation between
#width and lifetime (which is proportional to the inverse of escape rate)

# log10(worldRad) coefficient shows no significant correlation at all
