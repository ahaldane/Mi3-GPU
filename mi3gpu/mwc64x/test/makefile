########################################
## Stuff you'll want to modify to point to opencl in your environment

ifeq ($(OS),Linux)
CL_DIR = /vol/cc/opt/cuda
LDFLAGS += -L$(CL_DIR)/lib64
else
CL_DIR = /C/ATI/Stream2p3
LDFLAGS += -L$(CL_DIR)/lib/x86
endif

# This is for OpenCL distributions that don't have cl.hpp
CPPFLAGS += -ICLEXT

CPPFLAGS += -I$(CL_DIR)/include -g -DNDEBUG=1 -O3 -msse2
LDLIBS += -lOpenCL

########################################
## Various macros for performing tests
##
## run "make check" to run the autotests

define rng_template_rule

output/$(1)_wi$(2).txt : dump_samples test_$(1).cl
	@mkdir -p output
	./dump_samples test_$(1).cl $(2) > $$@
	
OUTPUTS += output/$(1)_wi$(2).txt

endef

WIS = 1 2 4 8 16
RNGS = $(subst .cl,,$(subst test_,,$(wildcard test_*.cl)))

$(foreach rng,$(RNGS),$(foreach i,$(WIS),$(eval $(call rng_template_rule,$(rng),$(i)))))

.INTERMEDIATE : $(OUTPUTS)

clean :
	-rm test_rng dump_samples estimate_pi
	-rm test_rng.exe dump_samples.exe estimate_pi.exe
	-rm -rf output/*.txt

check : test_rng estimate_pi $(OUTPUTS)
	@rngs="$(RNGS)"; \
	echo "Running basic sanity checks..."; \
	for i in $$rngs; do \
		./test_rng "test_$${i}.cl"; \
	done; \
	echo "Checking that the stream skipping works (there should be no file differences)..."; \
	for i in output/*.txt; do \
		diff --brief output/mwc64x_wi1.txt $$i; \
	done; \
	echo "... if diff didn't complain, it succeeded."; \
	wis="1 2 4 8 16 32 64 512"; \
	echo "Checking pi estimator (all versions should report exactly the same answer)"; \
	for i in $$rngs; do \
		for j in $$wis; do \
			./estimate_pi test_$$i.cl $$j 16; \
		done; \
	done; \
	echo "... finished pi test"; \
	echo "All tests done."
