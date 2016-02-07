# Makefile - prepared for MC3
#
# `make` - Build and compile the MC3 executable and python extension.
# `make clean` - Remove all compiled (non-source) files that are created.
#
# If you are interested in the commands being run by this makefile, you may add
# "VERBOSE=1" to the end of any `make` command, i.e.:
#
# 		make VERBOSE=1
#
# This will display the exact commands being used for building, etc.
#

LIBDIR = MCcubed/lib/

# Set verbosity
#
Q = @
O = > /dev/null

ifdef VERBOSE
	ifeq ("$(origin VERBOSE)", "command line")
		Q =
		O =
	endif
endif

all:
	@echo "Building MC3 package."
	$(Q) python setup.py build $(O)
	@mv -f build/lib.*/*.so $(LIBDIR)
	@rm -rf build/
	@echo "\nSuccessful compilation."
clean:
	@rm -rf $(LIBDIR)*.so
