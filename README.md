# LUMI-CPEtools

This repository contains a number of tools that are useful for testing
allocations etc on LUMI.

As some of the tools rely on MPI and others on the OpenMP run time, they
should be installed for each Cray Programming Environment separately.

These tools were not only tested on LUMI. Several of them may also work on
other clusters to you can rebuild them for your home cluster to compare the
results.


## Some included tools

-   `hybrid_check`: Derived from the `acheck` and `xthi` commands used by HPE Cray in their
    tutorials.

-   `gpu_Check`: Derived from the ORNL `hello_jobstep` program.

