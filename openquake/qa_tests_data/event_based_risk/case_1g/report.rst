event based risk with PMF
=========================

================================================ ========================
tstation.gem.lan:/mnt/ssd/oqdata/calc_21315.hdf5 Fri May 12 10:45:44 2017
engine_version                                   2.4.0-git59713b5        
hazardlib_version                                0.24.0-git0596dd3       
================================================ ========================

num_sites = 1, sitecol = 809 B

Parameters
----------
=============================== ==================
calculation_mode                'event_based_risk'
number_of_logic_tree_samples    0                 
maximum_distance                {'default': 200.0}
investigation_time              1000.0            
ses_per_logic_tree_path         1                 
truncation_level                3.0               
rupture_mesh_spacing            2.0               
complex_fault_mesh_spacing      2.0               
width_of_mfd_bin                0.1               
area_source_discretization      10.0              
ground_motion_correlation_model None              
random_seed                     42                
master_seed                     0                 
avg_losses                      True              
=============================== ==================

Input files
-----------
======================== ============================================================
Name                     File                                                        
======================== ============================================================
exposure                 `exposure_model.xml <exposure_model.xml>`_                  
gsim_logic_tree          `gsim_logic_tree.xml <gsim_logic_tree.xml>`_                
job_ini                  `job.ini <job.ini>`_                                        
source                   `source_model.xml <source_model.xml>`_                      
source_model_logic_tree  `source_model_logic_tree.xml <source_model_logic_tree.xml>`_
structural_vulnerability `vulnerability_model.xml <vulnerability_model.xml>`_        
======================== ============================================================

Composite source model
----------------------
========= ====== ====================================== =============== ================
smlt_path weight source_model_file                      gsim_logic_tree num_realizations
========= ====== ====================================== =============== ================
b1        1.000  `source_model.xml <source_model.xml>`_ trivial(1)      1/1             
========= ====== ====================================== =============== ================

Required parameters per tectonic region type
--------------------------------------------
====== =================== ========= ========== ==========
grp_id gsims               distances siteparams ruptparams
====== =================== ========= ========== ==========
0      BooreAtkinson2008() rjb       vs30       mag rake  
====== =================== ========= ========== ==========

Realizations per (TRT, GSIM)
----------------------------

::

  <RlzsAssoc(size=1, rlzs=1)
  0,BooreAtkinson2008(): ['<0,b1~b1,w=1.0>']>

Number of ruptures per tectonic region type
-------------------------------------------
================ ====== ==================== =========== ============ ============
source_model     grp_id trt                  num_sources eff_ruptures tot_ruptures
================ ====== ==================== =========== ============ ============
source_model.xml 0      Active Shallow Crust 1           31           482         
================ ====== ==================== =========== ============ ============

Informational data
------------------
============================ =========================================================================
compute_ruptures.received    tot 25.88 KB, max_per_task 25.88 KB                                      
compute_ruptures.sent        sources 3.09 KB, monitor 890 B, src_filter 684 B, gsims 102 B, param 65 B
hazard.input_weight          482                                                                      
hazard.n_imts                1 B                                                                      
hazard.n_levels              8 B                                                                      
hazard.n_realizations        1 B                                                                      
hazard.n_sites               1 B                                                                      
hazard.n_sources             1 B                                                                      
hazard.output_weight         8.000                                                                    
hostname                     tstation.gem.lan                                                         
require_epsilons             0 B                                                                      
============================ =========================================================================

Estimated data transfer for the avglosses
-----------------------------------------
1 asset(s) x 1 realization(s) x 1 loss type(s) x 1 losses x 8 bytes x 50 tasks = 400 B

Exposure model
--------------
=============== ========
#assets         1       
#taxonomies     1       
deductibile     absolute
insurance_limit absolute
=============== ========

======== ===== ====== === === ========= ==========
taxonomy mean  stddev min max num_sites num_assets
tax1     1.000 NaN    1   1   1         1         
======== ===== ====== === === ========= ==========

Slowest sources
---------------
====== ========= ================= ============ ========= ========= =========
grp_id source_id source_class      num_ruptures calc_time num_sites num_split
====== ========= ================= ============ ========= ========= =========
0      1         SimpleFaultSource 482          0.0       0         0        
====== ========= ================= ============ ========= ========= =========

Computation times by source typology
------------------------------------
================= ========= ======
source_class      calc_time counts
================= ========= ======
SimpleFaultSource 0.0       1     
================= ========= ======

Information about the tasks
---------------------------
================== ===== ====== ===== ===== =========
operation-duration mean  stddev min   max   num_tasks
compute_ruptures   0.071 NaN    0.071 0.071 1        
================== ===== ====== ===== ===== =========

Slowest operations
------------------
================================ ========= ========= ======
operation                        time_sec  memory_mb counts
================================ ========= ========= ======
total compute_ruptures           0.071     0.023     1     
saving ruptures                  0.007     0.0       1     
reading exposure                 0.006     0.0       1     
filtering ruptures               0.005     0.0       31    
reading composite source model   0.004     0.0       1     
setting event years              0.003     0.0       1     
store source_info                0.001     0.0       1     
managing sources                 9.909E-04 0.0       1     
filtering composite source model 4.649E-05 0.0       1     
reading site collection          7.868E-06 0.0       1     
================================ ========= ========= ======