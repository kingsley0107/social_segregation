### Quantitatively calculating social segregation using Signal mobile data

### key functions

| Function                                   | Description                                                              |
| ------------------------------------------ | ------------------------------------------------------------------------ |
| get_prob_matrix                            | calculating individual's location probability in each grid(spatial unit) |
| get_PSI_individual_location_matrix_chunked | calculating all samples' PSI from location possibility matrix            |
| get_PSI_matrix_for_individual              | PSI matrix for result, grouping by individual                            |
| get_PSI_matrix_for_unit                    | PSI matrix grouping by spatial unit(500m)                                |
| get_prob_matrix_in_timewindow              | PSI matrix , in one-hour-long level                                      |
