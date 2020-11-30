.PHONY: run test test_functions test_networks

test_modules = tests.test_functions test_networks

run:
	python3 main.py

test:
	python3 -m unittest -v ${test_modules}

test_functions:
	python3 -m unittest -v tests.test_functions

test_networks:
	python3 -m unittest -v tests.test_networks
