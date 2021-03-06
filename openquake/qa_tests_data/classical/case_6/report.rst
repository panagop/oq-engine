Classical Hazard QA Test, Case 6
================================

================================================ ========================
tstation.gem.lan:/mnt/ssd/oqdata/calc_21321.hdf5 Fri May 12 10:45:45 2017
engine_version                                   2.4.0-git59713b5        
hazardlib_version                                0.24.0-git0596dd3       
================================================ ========================

num_sites = 1, sitecol = 809 B

Parameters
----------
=============================== ==================
calculation_mode                'classical'       
number_of_logic_tree_samples    0                 
maximum_distance                {'default': 200.0}
investigation_time              1.0               
ses_per_logic_tree_path         1                 
truncation_level                0.0               
rupture_mesh_spacing            0.1               
complex_fault_mesh_spacing      0.1               
width_of_mfd_bin                1.0               
area_source_discretization      10.0              
ground_motion_correlation_model None              
random_seed                     1066              
master_seed                     0                 
=============================== ==================

Input files
-----------
======================= ============================================================
Name                    File                                                        
======================= ============================================================
gsim_logic_tree         `gsim_logic_tree.xml <gsim_logic_tree.xml>`_                
job_ini                 `job.ini <job.ini>`_                                        
source                  `source_model.xml <source_model.xml>`_                      
source_model_logic_tree `source_model_logic_tree.xml <source_model_logic_tree.xml>`_
======================= ============================================================

Composite source model
----------------------
========= ====== ====================================== =============== ================
smlt_path weight source_model_file                      gsim_logic_tree num_realizations
========= ====== ====================================== =============== ================
b1        1.000  `source_model.xml <source_model.xml>`_ trivial(1)      1/1             
========= ====== ====================================== =============== ================

Required parameters per tectonic region type
--------------------------------------------
====== ================ ========= ========== ==========
grp_id gsims            distances siteparams ruptparams
====== ================ ========= ========== ==========
0      SadighEtAl1997() rrup      vs30       mag rake  
====== ================ ========= ========== ==========

Realizations per (TRT, GSIM)
----------------------------

::

  <RlzsAssoc(size=1, rlzs=1)
  0,SadighEtAl1997(): ['<0,b1~b1,w=1.0>']>

Number of ruptures per tectonic region type
-------------------------------------------
================ ====== ==================== =========== ============ ============
source_model     grp_id trt                  num_sources eff_ruptures tot_ruptures
================ ====== ==================== =========== ============ ============
source_model.xml 0      Active Shallow Crust 2           140          140         
================ ====== ==================== =========== ============ ============

Informational data
------------------
============================== =============================================================================
count_eff_ruptures.received    tot 2.13 KB, max_per_task 1.07 KB                                            
count_eff_ruptures.sent        sources 2.15 KB, monitor 1.66 KB, srcfilter 1.34 KB, gsims 182 B, param 130 B
hazard.input_weight            287                                                                          
hazard.n_imts                  1 B                                                                          
hazard.n_levels                3 B                                                                          
hazard.n_realizations          1 B                                                                          
hazard.n_sites                 1 B                                                                          
hazard.n_sources               2 B                                                                          
hazard.output_weight           3.000                                                                        
hostname                       tstation.gem.lan                                                             
require_epsilons               0 B                                                                          
============================== =============================================================================

Slowest sources
---------------
====== ========= ================== ============ ========= ========= =========
grp_id source_id source_class       num_ruptures calc_time num_sites num_split
====== ========= ================== ============ ========= ========= =========
0      2         ComplexFaultSource 49           0.002     1         1        
0      1         SimpleFaultSource  91           0.002     1         1        
====== ========= ================== ============ ========= ========= =========

Computation times by source typology
------------------------------------
================== ========= ======
source_class       calc_time counts
================== ========= ======
ComplexFaultSource 0.002     1     
SimpleFaultSource  0.002     1     
================== ========= ======

Information about the tasks
---------------------------
================== ===== ========= ===== ===== =========
operation-duration mean  stddev    min   max   num_tasks
count_eff_ruptures 0.002 4.196E-04 0.002 0.003 2        
================== ===== ========= ===== ===== =========

Slowest operations
------------------
================================ ========= ========= ======
operation                        time_sec  memory_mb counts
================================ ========= ========= ======
reading composite source model   0.096     0.0       1     
total count_eff_ruptures         0.005     0.0       2     
managing sources                 0.002     0.0       1     
store source_info                5.274E-04 0.0       1     
reading site collection          4.172E-05 0.0       1     
filtering composite source model 4.148E-05 0.0       1     
aggregate curves                 3.886E-05 0.0       2     
saving probability maps          2.456E-05 0.0       1     
================================ ========= ========= ======