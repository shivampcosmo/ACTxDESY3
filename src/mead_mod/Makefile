include ${COSMOSIS_SRC_DIR}/config/compilers.mk

all: libhmcode_mod.so mead_interface_mod.so

mead_interface_mod.so: libhmcode_mod.so mead_interface_mod.f90
	$(FC) $(FFLAGS) -shared -o $@ $+ -L. -lhmcode_mod $(LDFLAGS) -lcosmosis_fortran -lcosmosis



test: test.f90 libhmcode_mod.so
	$(FC) $(FFLAGS) -o $@ $< -L. -lhmcode_mod $(LDFLAGS)


clean:
	rm -f hmcode_mod
	rm -f libhmcode_mod.so
	rm -f hmcode_mod.o
	rm -f mead_interface_mod.so
	rm -rf test
	rm -rf power.dat
	rm -rf *.dSYM/
	rm -rf *.mod

libhmcode_mod.so: HMcode_mod.f90
	$(FC) $(FFLAGS) -c  $+ $(LDFLAGS)
	$(FC) -shared -o $(CURDIR)/$@ HMcode_mod.o
