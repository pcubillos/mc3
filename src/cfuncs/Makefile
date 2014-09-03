SRCDIR = src/
LIBDIR = lib/

all:
	python setup.py build_ext --inplace
	@mv -f *.so $(LIBDIR)
	@rm -rf build/
	@echo "\nSuccessful compilation"
clean:
	@rm -rf $(LIBDIR)*.so
