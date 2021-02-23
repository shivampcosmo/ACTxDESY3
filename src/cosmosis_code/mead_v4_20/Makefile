include ${COSMOSIS_SRC_DIR}/config/compilers.mk

all: libmhmcamb_mod.a mead_interface_mod.so

mead_interface_mod.so: libmhmcamb_mod.a mead_interface_mod.f90
	$(FC) $(FFLAGS) -shared -o $@ $+ -L. -lmhmcamb_mod $(LDFLAGS) -lcosmosis_fortran -lcosmosis



test: test.f90 libmhmcamb_mod.a $(FC) $(FFLAGS) -o $@ $< -L. -lmhmcamb_mod $(LDFLAGS)


clean:
	rm -f mhmcamb_mod
	rm -f libmhmcamb.a
	rm -f mhmcamb.o
	rm -f mhmcamb_mod.o
	rm -f mead_interface.so
	rm -f mead_interface_mod.so
	rm -rf test
	rm -rf power.dat
	rm -rf *.dSYM/
	rm -rf *.mod

libmhmcamb_mod.a: mhmcamb_mod.f90
	$(FC) $(FFLAGS) -c  $+ $(LDFLAGS)
	$(AR) rc $@ mhmcamb_mod.o


#include ${COSMOSIS_SRC_DIR}/config/compilers.mk
#
#all: libmhmcamb_mod.a  mead_interface_mod.so
#
#mead_interface_mod.so: mead_interface_mod.f90
#	$(FC) $(FFLAGS) -shared -o $@ -L. -lmhmcamb_mod $(LDFLAGS) -lcosmosis_fortran -lcosmosis
#
#test: test.f90 libmhmcamb.so
#	$(FC) $(FFLAGS) -o $@ $< -L. -lmhmcamb $(LDFLAGS)
#
#clean:
#	rm -f mhmcamb
#	rm -f libmhmcamb.a
#	rm -f libmhmcamb.so
#	rm -f mhmcamb.o
#	rm -f mead_interface.so
#	rm -rf test
#	rm -rf power.dat
#	rm -rf *.dSYM/
#	rm -rf *.mod
#
##libmhmcamb_mod.so: mhmcamb_mod.f90
##	$(FC) $(FFLAGS) -shared -o $@ -c  $+ $(LDFLAGS)
#
#libmhmcamb_mod.a: mhmcamb_mod.f90
#	$(FC) $(FFLAGS) -c  $+ $(LDFLAGS)
#	$(AR) rc $@ mhmcamb_mod.o
