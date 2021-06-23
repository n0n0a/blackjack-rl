.PHONY: clean install uninstall pytest

clean:
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info

uninstall:
	python3 -m pip uninstall blackjack-rl

install:
	python3 setup.py install

pytest:
	python3 -m pytest --doctest-modules

