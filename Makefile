all: modules

clean:
	rm -rf build pyest/mixtures/_algorithms.so pyest/acor/_acor.so

modules: pyest/mixtures/_algorithms.c
	python setup.py build_ext --inplace
	python setup_acor.py build_ext --inplace

