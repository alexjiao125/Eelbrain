UNAME := $(shell uname -m)

clean:
	rm -rf coverage .coverage build dist

clean-py:
	find . -type f -name *.pyc -delete
	find . -type f -name *.pyo -delete

doc:
	rm -rf doc/build doc/generated
	pythonw setup.py build_sphinx

style-check:
	pydocstyle eelbrain

flake:
	flake8 --count eelbrain examples scripts

test:
	pytest eelbrain

testw:
	pythonw $(shell which pytest) eelbrain

pypi:
	rm -rf build dist
	python setup.py sdist bdist_wheel bdist_egg upload

.PHONY: clean clean-py doc testw pypi
