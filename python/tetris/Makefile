PYTHON=python3

all:
	rm -rf build
	$(PYTHON) setup.py build_ext --inplace
	chmod 755 *.so

clean:
	rm -rf build src *.so
