# Makefile - prepared for MC3
#
# `make` - Build and compile the MC3 executable and python extension..
# `make clean` - Remove all compiled (non-source) files that are created.

# If you are interested in the commands being run by this makefile, you may add
# "VERBOSE=1" to the end of any `make` command, i.e.:
#      make VERBOSE=1
# This will display the exact commands being used for building, etc.

# To enforce compilation with Python3 append "PY3=1" to the make command:
#      make PY3=1


LIBDIR = mc3/lib/

# Set verbosity
#
Q = @
O = > /dev/null

ifdef VERBOSE
	ifeq ("$(origin VERBOSE)", "command line")
		Q =
		O =
	endif
else
	MAKEFLAGS += --no-print-directory
endif

DIRECTIVE = 
ifdef PY3
	ifeq ("$(origin PY3)", "command line")
		DIRECTIVE = 3
	endif
endif

all:
	@echo "Building MC3 package."
	$(Q) python$(DIRECTIVE) setup.py build $(O)
	@mv -f build/lib.*/$(LIBDIR)*.so ./$(LIBDIR)
	@rm -rf build/
	@echo "Successful compilation."
	@echo ""
clean:
	@rm -rf $(LIBDIR)*.so
