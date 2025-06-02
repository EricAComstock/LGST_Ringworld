#### Heading ####
# This file is used to generate qualitative ringworld data regarding surface vs
# edge atmospheric losses from particle numbers recaptured and escaped generated
# using the main Python codes for the simulation. It outputs the ringworld size
# expected to generate 100% losses and the loss rate associated with the face.
#
# V1.1, Eric Comstock, 02/06/2025
# V1.0, Eric Comstock, 02/06/2025

#### Data input ####

worldWidth             = c(1600000,100000,10000,1000)# Ringworld test widths
recaptured             = c(1000000-4-1048-1058-5378-5411-5439-5469-5542-5572-5439-5743
               ,9452,94414,93713)                    # Number of particles recaptured
escaped                = c(4, 1, 78, 851)            # Number of particles escaped

#### Process data ####

escape_frac_list       = c()                         # Fraction of particles escaped
escape_frac_std_list   = c()                         # Std. dev. of the above fraction

fudge_factor = 1e-6# Should be zero in the final code - prevents infinity from showing up

for (i in 1: length(worldWidth))                # Data processing
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
  print(paste0('Trial ', as.character(i), ', ', as.character(worldWidth[i]), ' km width:'))
  print(paste0('Expected escape chance: ', as.character(avg_escape)))
  print(paste0('Expected escape chance std. dev.: ', as.character(std_avg_escape)))
  print('')
  
  escape_frac_list     = c(escape_frac_list, avg_escape)
  escape_frac_std_list = c(escape_frac_std_list, std_avg_escape) * (1 - fudge_factor)
}

inverse.worldWidth     = 1 / worldWidth              # Calculating 1/worldWidth

# Calculating linear regressions for average, +1s.d., and -1s.d. escape fraction data
l.normal               = lm(escape_frac_list ~ inverse.worldWidth)
l.high                 = lm(escape_frac_list+escape_frac_std_list ~ inverse.worldWidth)
l.low                  = lm(escape_frac_list-escape_frac_std_list ~ inverse.worldWidth)

#### Plotting ###

library(Hmisc)# Plotting library

# Linear plot
errbar(inverse.worldWidth, escape_frac_list, escape_frac_list-escape_frac_std_list,
       escape_frac_list+escape_frac_std_list, type='b')

# log-log plot
errbar(log10(inverse.worldWidth), log10(escape_frac_list), log10(escape_frac_list
                                                                 -escape_frac_std_list),
       log10(escape_frac_list+escape_frac_std_list), type='b')

#### Linear regression and final results ####

# Print linear regression statistics
l.normal
l.high
l.low

# (Intercept) coefficient shows the leakage inherent to the faces - in other words,
#the rate when extrapolated to an infinitely wide ringworld without edges.
# This value has a range that includes zero, so thus it is statistically insignificant.

# inverse.worldWidth coefficient shows leakage coming from the edges. The value
#can be interpreted as the width of the ringworld in km that has 100% of particles
#likely to escape. This has a range around 8 to 9, indicating that ringworlds must
#be around that wide or larger to take advantage of lowered escape attempts from
#particles.
