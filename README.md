# A high-resolution inversion of 2019 TROPOMI methane observations over North America

This repository includes the code necessary to recreate an inversion of 2019
TROPOMI observations over North America.

## Inversion procedure
1. Run generate_clusters.py

   **Description**

   This script generates a cluster file that meets HEMCO requirements for use in inversions. The file maps a unique integer key for every grid cell contained in the state vector to the latitude-longitude grid used in the forward model. A grid cell is included in the state vector if it meets either the `emis_threshold` or `land_threshold` criteria.

   **Inputs**

   | Input             | Description                                        |
   | ----------------- | -------------------------------------------------- |
   | emis_file | A file or files containing information on methane emissions from the prior run. This is typically given by HEMCO_diagnostics. The input here can be either a list of monthly files or a single file with an annual average. |
   | land_file | A file containing information on land cover for inversion domain. This can be provided by the land cover file referenced by HEMCO_Config.rc. |
   | emis_threshold | An emission threshold in Mg/km2/yr that gives the minimum anthropogenic emissions needed in a grid cell for its inclusion in the cluster file. The default value is 0.1. |
   | land_threshold | A fractional land threshold that gives the minimum fraction of a grid cell that must be land covered for inclusion in the cluster file. The default value is 0.25. |

   **Outputs**

   | Output            | Description                                        |
   | ----------------- | -------------------------------------------------- |
   | clusters.nc | A HEMCO-ready cluster file that maps a unique key for every grid cell contained in the state vector to the latitude-longitude grid used in the forward model. |

   To do:
   - [ ] Move inputs to one directory
   - [x] Change hardcoded cluster file name

2. Run generate_prior.py

   **Description**

   This script generates netcdfs of the absolute and relative prior emissions and error variances for use in an analytical inversion.

   **Inputs**

   | Input             | Description                                        |
   | ----------------- | -------------------------------------------------- |
   | emis_file | A file or files containing information on methane  emissions from the prior run. This is typically given by HEMCO_diagnostics. The input here can be either a list of monthly files or a single file with an annual average. |
   | clusters | The cluster file generated by generate_clusters.py that maps a unique key for every grid cell contained in the state vector to the latitude - longitude grid used in the forward model. |
   | rel_err | The relative error (standard deviation) value to be used in the relative prior error covariance matrix. The default is 0.5. |

   **Outputs**

   | Output            | Description                                        |
   | ----------------- | -------------------------------------------------- |
   | xa.nc | A netcdf containing the relative prior (all ones) xa. |
   | sa.nc | A netcdf containing the relative prior error (all given by rel_err). |
   | xa_abs.nc | A netcdf containing the absolute prior. |
   | sa_abs.nc | A netcdf containing the absolute prior error. |

   To do:
   - [ ] Move inputs to one directory

4. Run generate_obs.py
   This script generates the observation vector, the prior observation vector (i.e. F(xa)), and the observational error variances. It applies filters on albedo, latitude, and seaason to remove problematic TROPOMI observations. The variances are calcualted using the residual error method.

   **Inputs**

   | Input             | Description                                        |
   | ----------------- | -------------------------------------------------- |
   | prior_run | A file or files containing the processed output of the prior run of the forward model (after applying the satellite averaging kernel. The input here can be either a list of daily files or a single file containing all observations for the year. |
   | filter_on_blended_albedo | A boolean flag indicating whether or not to filter on blended albedo, which should remove snow and ice covered scenes, as recommended by Lorente et al. 2021 and described in Wunch et al. 2011. Lorente et al. find a value of 0.85 and Wunch et al. find a value of about 1. We use 1.1. |
   | filter_on_albedo | A boolean flag indicating whether or not to filter out scenes below the albedo_threshold, following the recommendation of de Gouw et al. 2020. De Gouw et al. use 0.05. We do too. |
   | filter_on_seasonal_latitude | A boolean flag indicating whether or not to remove observations north of 50 degrees N during winter (DJF) months to further remove snow- and ice-covered scenes. |
   | remove_latitudinal_bias | A boolean flag indicating whether or not to remove the latitudinal bias in the model - observation difference with a first order polynomial. |

   **Outputs**

   | Output            | Description                                        |
   | ----------------- | -------------------------------------------------- |
   | y.nc | The observation vector containing bias-corrected TROPOMI observations. |
   | ya.nc | The prior observation vector containing the output of the prior simulation, i.e. F(xa). |
   | so.nc | The observational error variances calculated using the residual error method. |

## To do
- [ ] Rewrite to_dataset() (in inversion.py) to automatically identify elements of object
- [ ] Regenerate
- [ ] Move all inputs for inversion to one directory to allow for consistency across
      various scripts
- [ ] Add great lakes to plotting
- [ ] Fix doc strings at the top of every python file -- read me would be a good source for these
- [ ] Change all scripts to allow iterating through other years
