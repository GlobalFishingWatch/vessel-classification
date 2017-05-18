* List of known fishing vessels.
* Flag state extracted from mmsi parsing.
* Exclude vessels with < 200 positions in a year.
* For 365 days of AIS positions, for each vessel:
  * Find all periods where a vessel is stationary for >= 24 hours.
    * Continuous period where vessel is n ever more than 0.8km from first
      point in period.
* For all stationary periods:
  * Find the S2 cell of the mean of all locations over the stationary period.
    * S2 cell scale 13, min area 0.76 km^2, mean area 1.27 km^2, max area 1.59 km^2.
* Group all stationary periods by their S2 cell and calculate:
  * The mean location across all stationary periods in the cell.
  * The number of unique vessels that visited.
    * Store the mmsis, the # fishing vessels and the distribution of flag states.
  * The mean distance to shore.
  * The mean drift radius of all vessels (distinguishes sea anchorages from ports as their is greater drift in the former).
* Discard S2 cells with < 20 unique vessels stationary within, to give a set of Anchorage points.

Step 2, group adjacent 'Anchorage Points' into 'Anchorages'.

* Anchorages consist of groups of adjacent anchorage points (any S2 cells that are anchorages and are adjacent are merged).