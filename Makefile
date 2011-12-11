all: modules

clean:
	rm -rf build pyest/mixtures/_algorithms.so

modules: pyest/mixtures/_algorithms.c
	python setup.py build_ext --inplace

