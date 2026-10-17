# Preserved primitive-control correction

The first synthetic correctness-face control used competitor(0,1), so its selected margin z0=0 with positive s already violated final support; its passing exclusion did not discriminate the extra correctness constraints. Inspection replaced it with competitor(1,0), selected margin z1=0 and z0=s>0.

An overly strong control then asserted both rank sides had broad feasible faces. This failed before any trained application: with first gates active, x>=0, z1=x+1+y=0 can meet only rank side10 (x>=1,y<=1), not side01. The corrected exact expectation is broad feasibility[false,true], with full correctness feasibility[false,false] because competitor(2,2) beats target(1,1) when z0=s>0. This is a synthetic-control expectation defect, not a trained result or face algorithm change. Failed CPU was not separately recorded.
